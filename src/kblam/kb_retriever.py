import numpy as np
import torch
from typing import List, Dict, Optional
import random
from kblam.kb_encoder import KBEncoder
from kblam.utils.train_utils import context_set_size_scheduler, get_kb_embd

import hnswlib
import json
from pathlib import Path
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer 
import time


def build_path_string(t1: dict, t2: dict) -> str:
    """
    Build a 2-hop path string with structural awareness.
    """

    name1 = t1["name"]
    type1 = t1["description_type"]
    type2 = t2["description_type"]

    is_attr1 = t1["key_string"].lower().startswith("the ") and " of " in t1["key_string"].lower()
    is_attr2 = t2["key_string"].lower().startswith("the ") and " of " in t2["key_string"].lower()

    # ---------- attribute + attribute ----------
    if is_attr1 and is_attr2:
        # the mother of the director of Film
        return f"the {type2} of the {type1} of {name1}"

    # ---------- relation + attribute ----------
    if not is_attr1 and is_attr2:
        # the mother of which X directed Film
        return f"the {type2} of which {name1} {type1}"

    # ---------- attribute + relation ----------
    if is_attr1 and not is_attr2:
        # the sequel related to the director of Film
        return f"the {type2} related to the {type1} of {name1}"

    # ---------- relation + relation ----------
    # X directed sequel
    return f"{name1} {type1} {type2}"

class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray] = None,
        value_embds: Optional[np.ndarray] = None,
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[str] = None,
        hnsw_index_path:Optional[str] = None,
        base_embeder_path:Optional[str] = None,
    ):

        self.encoder = encoder
        self.dataset = dataset
        
        if precomputed_embed_keys_path is not None and precomputed_embed_values_path is not None:
            self._load_cached_embd(precomputed_embed_keys_path, precomputed_embed_values_path)
        else:
            self.key_embds = key_embds
            self.value_embds = value_embds
        
        self.hnsw_index_path=hnsw_index_path
        self.hnsw_index = None
        self.hnsw_ready = False
        self.hnsw_dim = None
        self.hnsw_space = None
        if hnsw_index_path is not None:
            print(f"[HNSW] ENABLE")
            if not self.load_hnsw_index():
                print(f"Load HNSW index failed: {hnsw_index_path}, create a new one.")
                self.build_and_save_hnsw_index()
            if base_embeder_path is None:
                raise ValueError("base_embeder_path is None, but hnsw_index_path is not None.")
            # self.base_embeder = SentenceTransformer(base_embeder_path, device="cpu")
            self.base_embeder = SentenceTransformer(base_embeder_path, device="cuda")
            
        
        # metrics
        self.metrics_2hop_recall_1=0
        self.metrics_2hop_recall_topk=0

        self.retrieval_time=[]
        self.embedding_time=[]

    
    def reset_metrics(self):
        self.metrics_2hop_recall_1=0
        self.metrics_2hop_recall_topk=0
        self.retrieval_time=[]
        self.embedding_time=[]


    def _load_cached_embd(self, precomputed_embed_keys_path, precomputed_embed_values_path):
        self.key_embds = np.load(precomputed_embed_keys_path).astype("float32")
        self.value_embds = np.load(precomputed_embed_values_path).astype("float32")

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices:List[int], batch_size:Optional[int]=None, step:Optional[int]=None, kb_size:Optional[int]=None):

        if self._use_cached_embd():
            train_set_key, train_set_val = get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            train_set_key, train_set_val = get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)

        if kb_size is None:
            # during inference, kb_size is None, return (sample_size, embedding_dim)
            return train_set_key, train_set_val

        # (batch_size, 1, embedding_dim)
        if len(train_set_key.shape) == 2:
            train_set_key = train_set_key.unsqueeze(0).transpose(0, 1)
            train_set_val = train_set_val.unsqueeze(0).transpose(0, 1)

        context_set_size = context_set_size_scheduler(step, kb_size)
        context_set_index = np.random.choice(len(self.dataset), context_set_size, replace=False)  # type: ignore
        if self._use_cached_embd():
            context_set_key, context_set_val = get_kb_embd(
                self.encoder,
                context_set_index,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            context_set_key, context_set_val = get_kb_embd(self.encoder, context_set_index, kb_dict=self.dataset)
        
        # (batch_size, context_set_size, embedding_dim)
        context_set_key = context_set_key.unsqueeze(0).expand(batch_size, *context_set_key.shape)
        context_set_val = context_set_val.unsqueeze(0).expand(batch_size, *context_set_val.shape)

        true_kb_copy = 1
        # kb_embedding = (
        #     torch.concat([*([train_set_key] * true_kb_copy), context_set_key], 1),
        #     torch.concat([*([train_set_val] * true_kb_copy), context_set_val], 1),
        # )

        # return kb_embedding

        B = batch_size
        C = context_set_key.size(1)
        insert_pos = torch.randint(0, C + 1, (B,), device=context_set_key.device)

        new_keys = []
        new_vals = []
        for b in range(B):
            keys = torch.cat(
                [context_set_key[b, :insert_pos[b]], train_set_key[b], context_set_key[b, insert_pos[b]:]],
                dim=0,
            )
            vals = torch.cat(
                [context_set_val[b, :insert_pos[b]], train_set_val[b], context_set_val[b, insert_pos[b]:]],
                dim=0,
            )
            new_keys.append(keys.unsqueeze(0))
            new_vals.append(vals.unsqueeze(0))

        train_set_key = torch.cat(new_keys, dim=0)
        train_set_val = torch.cat(new_vals, dim=0)

        kb_embedding = (train_set_key, train_set_val)
        return kb_embedding

    def get_embeddings_with_adj_2wiki(
        self,
        batch_indices,
        kb_size: int | None = None,
        step: int | None = None,
        device: torch.device | None = None,
        hop_num: int = 2,
    ):
        if not self._use_cached_embd():
            raise RuntimeError("get_embeddings_with_adj_2wiki only support cached KB embedding")

        if hop_num != 2:
            raise ValueError("get_embeddings_with_adj_2wiki only support hop_num=2")

        B = len(batch_indices)

        # ------------------------------------------------
        # 1) 取出每个样本的“真 KB triple”索引
        #    假设每个样本的两条 triple 在全局 embedding 中是连续的：
        #    sample i → idx = 2*i, 2*i+1
        #    如果你有单独的 triple_offset 数组，也可以改成：
        #      base = triple_offset[idx]
        #      true_indices.extend([base, base+1])
        # ------------------------------------------------
        true_triple_indices = []
        for idx in batch_indices:
            base = idx * hop_num
            true_triple_indices.extend([base, base + 1])
        true_triple_indices = np.array(true_triple_indices, dtype=np.int64)

        # ------------------------------------------------
        # 2) 取出真 triple 的 base embedding，并通过 KBEncoder 编码
        # ------------------------------------------------
        key_true_np = self.key_embds[true_triple_indices]
        val_true_np = self.value_embds[true_triple_indices]

        key_true = self.encoder.encode_key(base_emb=key_true_np)   # (B*2, d)
        val_true = self.encoder.encode_val(base_emb=val_true_np)   # (B*2, d)


        if kb_size is None:
            # 推理模式
            # 一次性传入指定的样本ID，构建成一张KB图
            kb_size=len(batch_indices)
            kb_len = hop_num*kb_size
            # ------------------------------------------
            # given: kb_size=100, hop_num=2, kb_len=200
            # row_idx = [0, 2, 4, ..., 198]
            # col_idx = [1, 3, 5, ..., 199]
            # indices = [[0, 2, 4, ..., 198], [1, 3, 5, ..., 199]], (2*100)
            # values = [1.0, 1.0, 1.0, ..., 1.0], (100)
            # sparse_adj = (200,200)的稀疏矩阵，100条边[(0,1), (2,3), ..., (198,199)], 非零元素为1.0
            # ------------------------------------------
            adj_device = device if device is not None else key_true.device
            row_idx = torch.arange(0, kb_len, hop_num, device=adj_device)
            col_idx = row_idx + 1
            indices = torch.stack([row_idx, col_idx])
            values = torch.ones(row_idx.size(0), dtype=key_true.dtype, device=adj_device)
            sparse_adj = torch.sparse_coo_tensor(indices, values, (kb_len, kb_len), device=adj_device).coalesce()
            # kb_adjs.append(sparse_adj)
            return key_true, val_true, sparse_adj

        # 形状变为(B, 2, d)
        key_true = key_true.view(B, hop_num, -1)
        val_true = val_true.view(B, hop_num, -1)

        # ------------------------------------------------
        # 4) 有 kb_size 时：仿照 get_key_embeddings，随机采样 context KB
        #    注意：kb_size 控制“最大 context 数量”，最终每个样本 KB 总长度为：
        #      kb_len = hop_num + context_set_size
        # ------------------------------------------------
        total_samples = int(len(self.key_embds) / hop_num)
        # 这里用 context_set_size_scheduler 让 context_set_size 随 step 变化
        context_set_size = context_set_size_scheduler(step, kb_size)
        context_set_size = min(context_set_size, total_samples)
        #print("[kb_size]:", context_set_size * 2 + 2)

        # 从全局 KB 中随机采样 context_set_size 个 triple（所有样本共享）
        random_sample_ids = np.random.choice(total_samples, context_set_size, replace=False)

        random_triple_indices=[]
        for idx in random_sample_ids:
            base = idx * hop_num
            random_triple_indices.extend([base, base + 1])
        random_triple_indices = np.array(random_triple_indices, dtype=np.int64)

        # 随机选择的 context triple 形状： （C*2, d）
        key_ctx_np = self.key_embds[random_triple_indices]
        val_ctx_np = self.value_embds[random_triple_indices]

        key_ctx = self.encoder.encode_key(base_emb=key_ctx_np)   # (C*2, d)
        val_ctx = self.encoder.encode_val(base_emb=val_ctx_np)   # (C*2, d)

        # 扩展成 (B, C*2, d)，每个样本共享同一批 context KB
        key_ctx = key_ctx.unsqueeze(0).expand(B, *key_ctx.shape).contiguous()
        val_ctx = val_ctx.unsqueeze(0).expand(B, *val_ctx.shape).contiguous()


        key_emb = torch.cat([key_true, key_ctx], dim=1)  # (B, 2 + C*2, d)
        val_emb = torch.cat([val_true, val_ctx], dim=1)  # (B, 2 + C*2, d)

        kb_len = hop_num*(1+context_set_size)

        # ------------------------------------------------
        # 6) 构造路径邻接矩阵 kb_adj
        #    - 只保留 0->1 的边，其余 context 均视作“孤立节点”
        # ------------------------------------------------
        # kb_adj = torch.zeros(B, kb_len, kb_len, dtype=torch.float32, device=device)
        # for b in range(B):
        #     for i in range(0, kb_len, hop_num):
        #         kb_adj[b, i, i+1] = 1.0
        adj_device = device if device is not None else key_emb.device
        batch_entries = []
        value_entries = []
        for b in range(B):
            rows = torch.arange(0, kb_len, hop_num, device=adj_device)
            cols = rows + 1
            batch_entries.append(torch.stack([torch.full_like(rows, b), rows, cols]))
            value_entries.append(torch.ones(rows.size(0), dtype=key_emb.dtype, device=adj_device))
        if batch_entries:
            indices = torch.cat(batch_entries, dim=1)
            values = torch.cat(value_entries, dim=0)
        else:
            indices = torch.empty((3, 0), dtype=torch.long, device=adj_device)
            values = torch.empty((0,), dtype=key_emb.dtype, device=adj_device)
        kb_adj = torch.sparse_coo_tensor(
            indices, values, (B, kb_len, kb_len), device=adj_device, dtype=key_emb.dtype
        ).coalesce()

        return key_emb, val_emb, kb_adj

    def build_and_save_hnsw_index(
        self,
        space: str = "cosine",
        ef_construction: int = 200,
        M: int = 32,
    ):
        """
        Build HNSW index and save to disk.
        """
        assert self._use_cached_embd(), "HNSW requires cached embeddings"

        if not os.path.isdir(self.hnsw_index_path):
            print(f"[HNSW] index path {self.hnsw_index_path} does not exist, use current directory {os.getcwd()} to save index")
            self.hnsw_index_path = os.getcwd()

        index_path = os.path.join(self.hnsw_index_path, "hnsw.index")
        meta_path = os.path.join(self.hnsw_index_path, "hnsw.meta.json")

        key_np = self.key_embds.astype("float32")
        N, D = key_np.shape

        index = hnswlib.Index(space=space, dim=D)
        index.init_index(
            max_elements=N,
            ef_construction=ef_construction,
            M=M,
        )

        batch_size = int(N / 100)
        for i in tqdm(range(0, N, batch_size), desc="Building HNSW index"):
            end_idx = min(i + batch_size, N)
            index.add_items(key_np[i:end_idx], np.arange(i, end_idx))
        index.set_ef(ef_construction)

        # ---- save ----
        index.save_index(str(index_path))
        meta = {
            "space": space,
            "dim": D,
            "num_elements": N,
            "ef_construction": ef_construction,
            "M": M,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # ---- register ----
        self.hnsw_index = index
        self.hnsw_ready = True
        self.hnsw_dim = D
        self.hnsw_space = space

        print(f"[HNSW] index built & saved to {self.hnsw_index_path}")


    def load_hnsw_index(
        self,
        ef_search: int = 200,
    ):
        """
        Load prebuilt HNSW index from disk.
        """
        index_path = os.path.join(self.hnsw_index_path, "hnsw.index")
        meta_path = os.path.join(self.hnsw_index_path, "hnsw.meta.json")

        if not os.path.exists(index_path):
            print(f"HNSW index not found: {index_path}")
            return False
        if not os.path.exists(meta_path):
            print(f"HNSW meta not found: {meta_path}")
            return False

        with open(meta_path) as f:
            meta = json.load(f)

        space = meta["space"]
        dim = meta["dim"]
        num_elements = meta["num_elements"]

        index = hnswlib.Index(space=space, dim=dim)
        index.load_index(str(index_path), max_elements=num_elements)
        index.set_ef(ef_search)

        self.hnsw_index = index
        self.hnsw_ready = True
        self.hnsw_dim = dim
        self.hnsw_space = space

        print(f"[HNSW] index loaded from {self.hnsw_index_path}, data size: {num_elements}")
        return True


    def create_query_embeddings(self, questions: list[str]):
        """
        Encode question into the SAME embedding space as key_embds.
        """
        # return self.encoder.embedding_query_cpu(questions)  # (B, D)
        start_time = time.time()
        q_embs = self.base_embeder.encode( questions, convert_to_numpy=True)  # (B, D)
        end_time = time.time()
        self.embedding_time.append(end_time - start_time)
        return q_embs  # (B, D)


    def retrieve_topk(
        self,
        query_emb: np.ndarray,
        topk: int,
    ):
        assert self.hnsw_ready, "HNSW index not initialized"

        start_time = time.time()
        labels, distances = self.hnsw_index.knn_query(
            query_emb.reshape(1, -1),
            k=topk,
        )
        end_time = time.time()
        self.retrieval_time.append(end_time - start_time)
        return labels[0]   # shape: (topk,)

    def retrieve_topk_batch(
        self,
        query_embs: np.ndarray,
        topk: int,
    ):
        """
        Batch HNSW retrieval.

        Args:
            query_embs: np.ndarray, shape (B, D)
            topk: int

        Returns:
            labels: np.ndarray, shape (B, topk)
        """
        assert self.hnsw_ready, "HNSW index not initialized"
        assert query_embs.ndim == 2, "query_embs must be 2D (B, D)"

        start_time = time.time()
        labels, distances = self.hnsw_index.knn_query(
            query_embs,
            k=topk,
        )
        end_time = time.time()

        self.retrieval_time.append(end_time - start_time)

        return labels   # (B, topk)



    def get_kb_batch_by_hnsw(
        self,
        questions: list[str],
        topk: int,
        device = "cuda",
        true_indices: Optional[list[int]] = None,
        random_sample: Optional[int] = None,
    )->tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        For inference only.
        Returns:
            kb_keys: (B, topk, d)
            kb_vals: (B, topk, d)
        """
        assert self._use_cached_embd()
        assert self.hnsw_ready

        B = len(questions)

        all_keys, all_vals = [], []
        q_embs = self.create_query_embeddings(questions)  # (B, D)
        for i, q in enumerate(questions):
            q_emb = q_embs[i]                              # (D,)
            idxs = self.retrieve_topk(q_emb, topk)        # (topk,)
            if random_sample is not None:
                random_idxs = np.random.choice(self.key_embds.shape[0], random_sample, replace=False)
                idxs = np.concatenate([idxs, random_idxs]).astype(int).tolist()

            # collect retrieval results
            if true_indices is not None:
                assert len(true_indices) == B, "true_indices must have the same length as questions"
                true_idx = true_indices[i]
                pos = np.where(idxs == true_idx)[0]
                if len(pos) > 0:
                    rank_true = int(pos[0]) + 1
                else:
                    rank_true = None   # or topk + 1
                print(f"[HNSW] Retrieval Accuracy: top-{topk}={rank_true}")
                print(f"[HNSW] Retrieval Indexes (true={true_idx}): {idxs}")

            train_set_key, train_set_val=self.get_key_embeddings(idxs)

            all_keys.append(train_set_key)
            all_vals.append(train_set_val)

        return all_keys, all_vals

    def is_hnsw_ready(self):
        return self.hnsw_ready

    def get_kb_adj_batch_by_hnsw(
        self,
        questions: list[str],
        ann_topk: int = 10,
        rerank_policy: int = 2,
        rerank_topk: int = 1,
        device = "cuda",
        true_indices: Optional[list[int]] = None,
        random_sample: Optional[int] = None,
        hop_num: int =2,
    ):
        """
        For inference only.
        Returns:
            kb_keys: (B, topk, d)
            kb_vals: (B, topk, d)
            kb_adj: (B, topk, topk)
        """
        assert self._use_cached_embd()
        assert self.hnsw_ready

        B = len(questions)

        all_keys, all_vals, all_adj = [], [], []
        q_embs = self.create_query_embeddings(questions)  # (B, D)
        for i, q in enumerate(questions):
            q_emb = q_embs[i]                              # (D,)
            if rerank_policy == 1:
                idxs = self.get_retrieve_idx_v1(q_emb, topk=ann_topk)        # (topk,)
            elif rerank_policy == 2:
                idxs = self.get_retrieve_idx_v2(q_emb, topk=ann_topk)        # (topk,)
            elif rerank_policy == 3:
                idxs = self.get_retrieve_idx_v3(q_emb, topk=ann_topk)        # (topk,)
            else:
                raise ValueError(f"Unknown rerank_policy: {rerank_policy}")

            # 取前rerank_topk个索引
            idxs = idxs[:rerank_topk]

            # 统一索引
            idxs = (np.asarray(idxs, dtype=np.int64) // hop_num).tolist()
            if random_sample is not None:
                total_samples = int(self.key_embds.shape[0] // hop_num)
                random_idxs = np.random.choice(total_samples, random_sample, replace=False).astype(np.int64).tolist()
                idxs = idxs + random_idxs
            # collect retrieval results
            if true_indices is not None:
                assert len(true_indices) == B, "true_indices must have the same length as questions"
                true_idx = int(true_indices[i])
                if len(idxs) > 0 and true_idx == int(idxs[0]):
                    self.metrics_2hop_recall_1 += 1

            kb_k, kb_v, kb_adj=self.get_embeddings_with_adj_2wiki(idxs, hop_num=hop_num)
            all_keys.append(kb_k)
            all_vals.append(kb_v)
            all_adj.append(kb_adj)
        return all_keys, all_vals, all_adj

    def get_retrieve_idx_v1(self, q_embd, topk: int = 10):
        idxs = self.retrieve_topk(q_embd, topk)
        return idxs

    def get_retrieve_idx_v2(self, q_emb, topk: int = 10, alpha: float = 1.0, beta: float = 0.25, hop_num: int = 2):
        # ---------- 2) first-hop retrieval ----------
        first_hop_idxs = self.retrieve_topk(q_emb, topk=topk)  # (topk,)
        first_hop_idxs = (first_hop_idxs // hop_num) * hop_num

        # ---------- 3) collect first-hop scores ----------
        first_keys = self.key_embds[first_hop_idxs]          # (K, D)
        first_scores = first_keys @ q_emb                    # (K,)

        # ---------- 4) collect second-hop scores ----------
        second_hops = []
        valid_first = []
        valid_first_scores = []

        for c, s1 in zip(first_hop_idxs, first_scores):
            c = int(c)
            second_hop = c + 1
            if second_hop < len(self.key_embds):
                second_hops.append(second_hop)
                valid_first.append(c)
                valid_first_scores.append(s1)

        if not second_hops:
            raise ValueError("No second-hop candidates found.")

        second_keys = self.key_embds[second_hops]             # (K', D)
        second_scores = second_keys @ q_emb                   # (K',)

        valid_first_scores = np.asarray(valid_first_scores)

        # ---------- 5) combined score ----------
        combined_scores = alpha * valid_first_scores + beta * second_scores

        # ---------- 6) reorder ----------
        sorted_idxs = np.argsort(combined_scores)[::-1]
        first_hop_idxs = first_hop_idxs[sorted_idxs]
        return first_hop_idxs
    
    def get_retrieve_idx_v3(self, q_emb, topk: int = 10, hop_num: int = 2):
        # ---------- 2) first-hop Top-K retrieval ----------
        first_hop_idxs = self.retrieve_topk(q_emb, topk=topk)

        # ---------- 3) batch prepare path components ----------
        # 对齐到 first hop
        first_hops = (first_hop_idxs // hop_num) * hop_num
        first_hops = first_hops.astype(int)

        # 过滤非法 second hop
        valid_mask = (first_hops + 1) < len(self.key_embds)
        first_hops = first_hops[valid_mask]

        if len(first_hops) == 0:
            raise ValueError("No valid second-hop candidates found.")

        # 映射到 dataset 行
        sample_ids = first_hops // hop_num

        # ---------- 4) build path strings (single tight loop) ----------
        path_strings = [
            build_path_string(
                self.dataset[sid]["triple_lists"][0],
                self.dataset[sid]["triple_lists"][1],
            )
            for sid in sample_ids
        ]
        
        # ---------- 5) embed paths ----------
        path_embs = self.create_query_embeddings(path_strings)  # (K, D)

        # ---------- 6) path-level similarity ----------
        scores = path_embs @ q_emb

        # ---------- 7) reorder ----------
        sorted_idxs = np.argsort(scores)[::-1]
        first_hop_idxs = first_hops[sorted_idxs]
        return first_hop_idxs



    def collect_recall_v1(self, topk: int = 1):
        hop_num=2
        self.metrics_2hop_recall_1 = 0
        self.metrics_2hop_recall_topk = 0
        for idx, row in enumerate(tqdm(self.dataset, desc="Calculating Recall@1")):
            q=row["Q"]
            q_embd=self.create_query_embeddings([q])[0]
            true_idxs=idx*2
            retrieval_idxs=self.retrieve_topk(q_embd, topk)
            if true_idxs == retrieval_idxs[0]:
                self.metrics_2hop_recall_1 += 1
            if true_idxs in retrieval_idxs:
                self.metrics_2hop_recall_topk += 1
            

        self.print_metrics()


    def collect_recall_v2(self, topk: int = 10, alpha: float = 1.0, beta: float = 0.25):
        """
        Key-level rerank Recall@1 with combined first-hop and second-hop scores.
        """

        assert self.hnsw_ready
        assert self._use_cached_embd()

        hop_num = 2
        self.metrics_2hop_recall_1 = 0
        self.metrics_2hop_recall_topk = 0

        rerank_time=[]

        for idx, row in enumerate(tqdm(self.dataset, desc="Calculating Recall@1 (Key-level rerank, combined)")):
            q = row["Q"]

            # ---------- 1) query embedding ----------
            q_emb = self.create_query_embeddings([q])[0]  # (D,)

            # ---------- 2) first-hop retrieval ----------
            first_hop_idxs = self.retrieve_topk(q_emb, topk=topk)  # (topk,)
            first_hop_idxs = (first_hop_idxs // hop_num) * hop_num

            true_first_hop = idx * hop_num
            start_rerank=time.time()
            # ---------- 3) collect first-hop scores ----------
            first_keys = self.key_embds[first_hop_idxs]          # (K, D)
            first_scores = first_keys @ q_emb                    # (K,)

            # ---------- 4) collect second-hop scores ----------
            second_hops = []
            valid_first = []
            valid_first_scores = []

            for c, s1 in zip(first_hop_idxs, first_scores):
                c = int(c)
                second_hop = c + 1
                if second_hop < len(self.key_embds):
                    second_hops.append(second_hop)
                    valid_first.append(c)
                    valid_first_scores.append(s1)

            if not second_hops:
                continue

            second_keys = self.key_embds[second_hops]             # (K', D)
            second_scores = second_keys @ q_emb                   # (K',)

            valid_first_scores = np.asarray(valid_first_scores)

            # ---------- 5) combined score ----------
            combined_scores = alpha * valid_first_scores + beta * second_scores

            best_idx = int(np.argmax(combined_scores))
            best_first_hop = valid_first[best_idx]
            rerank_time.append(time.time()-start_rerank)
            
            if best_first_hop == true_first_hop:
                self.metrics_2hop_recall_1 += 1
            if true_first_hop in first_hop_idxs:
                self.metrics_2hop_recall_topk += 1

        print(f"[HNSW] Rerank Time Distribution:")
        print(f"  - Count: {len(rerank_time)}")
        print(f"  - Mean: {np.mean(rerank_time):.4f}")
        print(f"  - 95% Tail: {np.percentile(rerank_time, 95):.4f}")
        self.print_metrics()

    def collect_recall_v3_1(self, topk: int = 10):
        assert self.hnsw_ready
        assert self._use_cached_embd()

        hop_num = 2
        self.metrics_2hop_recall_1 = 0
        self.metrics_2hop_recall_topk = 0
        max_hop_idx = len(self.dataset) * hop_num

        path_str_create_time=[]
        path_str_embedding_time=[]
        path_str_rerank_time=[]

        for idx, row in enumerate(tqdm(self.dataset, desc="Calculating Recall@1 (Path-level rerank)")):
            q = row["Q"]

            # ---------- 1) query embedding ----------
            q_emb = self.create_query_embeddings([q])[0]  # (D,)

            # ---------- 2) first-hop Top-K retrieval ----------
            first_hop_idxs = self.retrieve_topk(q_emb, topk=topk)

            true_first_hop = idx * hop_num

            # ---------- 3) batch prepare path components ----------
            # 对齐到 first hop
            first_hops = (first_hop_idxs // hop_num) * hop_num
            first_hops = first_hops.astype(int)

            # 过滤非法 second hop
            valid_mask = (first_hops + 1) < max_hop_idx
            first_hops = first_hops[valid_mask]

            if len(first_hops) == 0:
                continue

            # 映射到 dataset 行
            sample_ids = first_hops // hop_num

            # ---------- 4) build path strings (single tight loop) ----------
            t0 = time.time()
            path_strings = [
                build_path_string(
                    self.dataset[sid]["triple_lists"][0],
                    self.dataset[sid]["triple_lists"][1],
                )
                for sid in sample_ids
            ]
            t1 = time.time()
            path_str_create_time.append(t1-t0)

            # ---------- 5) embed paths ----------
            path_embs = self.create_query_embeddings(path_strings)  # (K, D)
            t2 = time.time()
            path_str_embedding_time.append(t2-t1)

            # ---------- 6) path-level similarity ----------
            scores = path_embs @ q_emb
            best_idx = int(np.argmax(scores))
            best_first_hop = first_hops[best_idx]

            if best_first_hop == true_first_hop:
                self.metrics_2hop_recall_1 += 1
            if true_first_hop in first_hop_idxs:
                self.metrics_2hop_recall_topk += 1
            t3 = time.time()
            path_str_rerank_time.append(t3-t2)

        self.print_metrics()
        print(f"[Path-level Rerank] Create Time: {np.mean(path_str_create_time):.4f}")
        print(f"[Path-level Rerank] Embedding Time: {np.mean(path_str_embedding_time):.4f}")
        print(f"[Path-level Rerank] Rerank Time: {np.mean(path_str_rerank_time):.4f}")


    def collect_recall1_batch(self):
        """
        Compute Recall@1 using batch HNSW retrieval.
        Assumes 2-hop KB layout: true index = idx * 2
        """
        hop_num = 2
        topk = 10

        # 1) batch encode all queries
        questions = [row["Q"] for row in self.dataset]
        q_embds = self.create_query_embeddings(questions)  # (B, D)

        # 2) batch retrieve
        labels = self.retrieve_topk_batch(q_embds, topk=topk)  # (B, 1)

        # 3) compute recall@1
        for idx in range(len(self.dataset)):
            true_idx = idx * hop_num
            if labels[idx, 0] == true_idx:
                self.metrics_2hop_recall_1 += 1
            t=10 if topk>10 else topk
            if true_idx in labels[idx, :t]:
                self.metrics_2hop_recall_10 += 1

        self.print_metrics()



    
    def get_avg_retrieval_time(self):
        return np.mean(self.retrieval_time)

    def print_metrics(self):
        print(f"[HNSW] Recall@1: {self.metrics_2hop_recall_1}/{len(self.dataset)}={self.metrics_2hop_recall_1/len(self.dataset):.4f}")
        print(f"[HNSW] Recall@top-k: {self.metrics_2hop_recall_topk}/{len(self.dataset)}={self.metrics_2hop_recall_topk/len(self.dataset):.4f}")
        print(f"[HNSW] Retrieval Time Distribution:")
        print(f"  - Count: {len(self.retrieval_time)}")
        print(f"  - Mean: {np.mean(self.retrieval_time):.4f}")
        print(f"  - 95% Tail: {np.percentile(self.retrieval_time, 95):.4f}")
        print(f"[HNSW] Embedding Time Distribution:")
        print(f"  - Count: {len(self.embedding_time)}")
        print(f"  - Mean: {np.mean(self.embedding_time):.4f}")
        print(f"  - 95% Tail: {np.percentile(self.embedding_time, 95):.4f}")
        