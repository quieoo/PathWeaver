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

class StageRetriever:
    def __init__(self, max_kb_paths=16, hop_num=2):
        self.max_kb_paths=max_kb_paths
        self.hop_num=hop_num
        # 四个阶段
        # self.stage_config=[0.5, 0.7, 0.85, 1]

        # 五个阶段: 0.1, 0.4, 0.3, 0.15, 0.15
        self.stage_config=[0.1, 0.4, 0.7, 0.85, 1]
        self.q_fit_stage=1

        self.path_nums_stage = [
            [1, 0, self.max_kb_paths-1],
            [1, 0, self.max_kb_paths-1],
            [1, int((self.max_kb_paths-1)/2), self.max_kb_paths-1-int((self.max_kb_paths-1)/2)],
            [1, self.max_kb_paths-1, 0],
            [0, self.max_kb_paths, 0]
        ]   # [gold, neg, random_neg]
            
    def get_stage(self, current_step ,total_steps):
        r = current_step / total_steps
        for s in range(len(self.stage_config)):
            if r < self.stage_config[s]:
                return s

    def get_path_stage(self, stage, sample, all_samples, max_kb_paths=16, shuffle=True):
        kb_paths=[]
        gold_path = sample.get("gold_path", None)
        neg_paths = sample.get("triple_lists", [])
        def random_paths(k):
            res = []
            while len(res) < k:
                s = random.choice(all_samples)
                if "gold_path" in s:
                    res.append(s["gold_path"])
            return res

        if stage <= 1:
            kb_paths=[gold_path]
            kb_paths += random_paths(max_kb_paths - 1)

        elif stage == 2:
            kb_paths = [gold_path]
            n_neg = min(len(neg_paths), max_kb_paths - 1)
            if n_neg > 0:
                kb_paths += random.sample(neg_paths, n_neg)
            n_random = max_kb_paths - 1 - n_neg
            if n_random > 0:
                kb_paths += random_paths(n_random)
        else:
            n_neg = min(len(neg_paths), max_kb_paths - 1)
            if n_neg > 0:
                kb_paths += random.sample(neg_paths, n_neg)
            n_random = max_kb_paths - 1 - n_neg
            if n_random > 0:
                kb_paths += random_paths(n_random)
        
        if shuffle:
            random.shuffle(kb_paths)
        
        return kb_paths


    def get_triple_ids_stage(self, stage, sample, all_samples, shuffle=True, verbose=False):
        sample_base_offset=sample["start_id"]

        # decide the num of paths
        acutal_gold_path_num=1 if sample.get("gold_path", None) is not None else 0
        acutal_neg_path_num=len(sample.get("triple_lists", []))

        stage_gold_num, stage_neg_num, stage_random_neg_num = self.path_nums_stage[stage]


        stage_gold_num = min(stage_gold_num, acutal_gold_path_num)
        stage_neg_num = min(stage_neg_num, acutal_neg_path_num)
        stage_random_num=self.max_kb_paths-stage_gold_num-stage_neg_num

        if verbose:
            print(f"[DEBUG] S{stage} gold-nega-random {stage_gold_num}-{stage_neg_num}-{stage_random_num}")
            
        path_triple_base=[] # 每个path的第一个三元组的id
        if stage_gold_num > 0:
            path_triple_base.append(sample_base_offset)
        
        if stage_neg_num > 0:
            for p in range(stage_neg_num):
                path_triple_base.append(sample_base_offset + self.hop_num + p * self.hop_num)

        if stage_random_num > 0:
            random_samples=random.sample(all_samples, stage_random_num)
            for s in random_samples:
                s_base_offset=s["start_id"]
                path_triple_base.append(s_base_offset)
        if shuffle:
            # triples_ids应该能够分成多个hop_num的组，将组间顺序打乱，保持组内顺序
            random.shuffle(path_triple_base)
        
        triples_ids=[]
        for p in path_triple_base:
            for i in range(self.hop_num):
                triples_ids.append(p + i)

        return triples_ids
            

    def get_question_type_stage(self, stage):
        if stage < self.q_fit_stage:
            return "gold_Q"
        else:
            return "Q"
    
def get_question_type_sampled_T1(current_step, total_steps, batch_size, verbose=False):
    # curriculum settings
    gold_question_ratio_per_stage = [1.0, 0.7, 0.3, 0.1]
    stage_num = len(gold_question_ratio_per_stage)

    progress = current_step / float(total_steps)
    cur_stage = min(int(progress * stage_num), stage_num - 1)
    gold_ratio = gold_question_ratio_per_stage[cur_stage]

    question_type_array = [
        "gold_Q" if random.random() < gold_ratio else "Q"
        for _ in range(batch_size)
    ]
    if verbose:
        print(f"[DEBUG] T1-Q-S{cur_stage} gold-ratio {gold_ratio} batch size {batch_size}, actual gold num {question_type_array.count('gold_Q')}")

    return question_type_array


# T1: 保持1*gold path + 随机(max_kb_paths-1)个neg path
def get_triple_ids_T1(sample, all_samples, max_kb_paths=16, hop_num=2, shuffle=True, verbose=False):
    sample_base_offset=sample["start_id"]
    stage_gold_num, stage_neg_num, stage_random_num = 1, 0, max_kb_paths-1

    if verbose:
        print(f"[DEBUG] T1-Path gold-nega-random {stage_gold_num}-{stage_neg_num}-{stage_random_num}")

    path_triple_base=[] # 每个path的第一个三元组的id
    if stage_gold_num > 0:
        path_triple_base.append(sample_base_offset)
    
    if stage_neg_num > 0:
        for p in range(stage_neg_num):
            path_triple_base.append(sample_base_offset + hop_num + p * hop_num)

    if stage_random_num > 0:
        random_samples=random.sample(all_samples, stage_random_num)
        for s in random_samples:
            s_base_offset=s["start_id"]
            path_triple_base.append(s_base_offset)
    if shuffle:
        # triples_ids应该能够分成多个hop_num的组，将组间顺序打乱，保持组内顺序
        random.shuffle(path_triple_base)
    
    triples_ids=[]
    for p in path_triple_base:
        for i in range(hop_num):
            triples_ids.append(p + i)

    return triples_ids


def get_question_type_sampled_T2(current_step, total_steps, batch_size, verbose=False):
    # curriculum settings
    gold_question_ratio = 0

    question_type_array = [
        "Q"
        for _ in range(batch_size)
    ]
    if verbose:
        print(f"[DEBUG] T2-Q gold-ratio {gold_question_ratio} batch size {batch_size}, actual gold num {question_type_array.count('gold_Q')}")

    return question_type_array

def get_triple_ids_T2(sample, all_samples, current_step, total_steps, max_kb_paths=16, hop_num=2, shuffle=True, verbose=False):
    path_compositions = [
        # gold path, random path, negative path
        [1, 3, max_kb_paths],
        [1, 0, max_kb_paths],
        [0, 0, max_kb_paths]
    ]
    step_ratio=[0.3, 0.5, 0.7, 1.0]
    stage_num=len(step_ratio)

    progress = current_step / float(total_steps)
    cur_stage = stage_num - 1
    for i in range(stage_num):
        if progress < step_ratio[i]:
            cur_stage = i
            break

    
    def get_path_composition(stage):
        if stage == 0:
            return path_compositions[0]
        elif stage == 1:
            return path_compositions[1] if random.random() < 0.7 else path_compositions[2]
        elif stage == 2:
            return path_compositions[1] if random.random() < 0.3 else path_compositions[2]
        else:
            return path_compositions[2]

    sample_base_offset=sample["start_id"]
    acutal_gold_path_num=1 if sample.get("gold_path", None) is not None else 0
    acutal_neg_path_num=len(sample.get("triple_lists", []))
    stage_gold_num, stage_random_num, stage_neg_num = get_path_composition(cur_stage)

    stage_gold_num=min(stage_gold_num, acutal_gold_path_num)
    stage_neg_num=min(stage_neg_num, acutal_neg_path_num)

    if verbose:
        print(f"[DEBUG] T2-Path-S{cur_stage} gold-nega-random {stage_gold_num}-{stage_neg_num}-{stage_random_num}")

    path_triple_base=[] # 每个path的第一个三元组的id
    if stage_gold_num > 0:
        path_triple_base.append(sample_base_offset)
    
    if stage_neg_num > 0:
        for p in range(stage_neg_num):
            path_triple_base.append(sample_base_offset + hop_num + p * hop_num)

    if stage_random_num > 0:
        random_samples=random.sample(all_samples, stage_random_num)
        for s in random_samples:
            s_base_offset=s["start_id"]
            path_triple_base.append(s_base_offset)
            
    if shuffle:
        # triples_ids应该能够分成多个hop_num的组，将组间顺序打乱，保持组内顺序
        random.shuffle(path_triple_base)
    
    triples_ids=[]
    for p in path_triple_base:
        for i in range(hop_num):
            triples_ids.append(p + i)

    return triples_ids

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
        if base_embeder_path is not None:
            self.base_embeder = SentenceTransformer(base_embeder_path)
            self.base_embeder.to("cuda")
        
            
        
        # metrics
        self.metrics_2hop_recall_1=0
        self.metrics_2hop_recall_topk=0

        self.retrieval_time=[]
        self.embedding_time=[]
        self.stage_starts=[False, False, False, False] 

    
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


    def build_kb_embedding_and_adj_for_at2qa(
        self,
        sample_id,
        step: int,
        total_steps: int,
        max_kb_paths: int = 16,
        device: torch.device | None = None,
        hop_num: int = 2,
    ):
        sample=self.dataset[sample_id]
        all_samples=self.dataset

        sr=StageRetriever()
        # decide current stage
        stage = sr.get_stage(step, total_steps)
        kb_paths = sr.get_path_stage(stage, sample, all_samples, max_kb_paths)

        
        for s in range(len(self.stage_starts)):
            if not self.stage_starts[s] and stage >= s:
                self.stage_starts[s] = True
                print(f"[KBRetriever] Stage {s} starts")
                print(f"Q: {sample['Q']}")
                print(f"gold_Q: {sample['gold_Q']}")
                print(f"A: {sample['A']}")
                print("-------------paths-------------")
                print(kb_paths)
                print("-------------------------------")

        # create kb embeddings
        key_texts = []
        val_texts = []

        for path in kb_paths:
            # NOTE: 每个路径最多 hop_num 个 hop
            path=path[:hop_num]
            for tri in path:
                key_texts.append(tri["key_string"])
                # value 用 description（与你现有 KBEncoder 设计一致）
                val_texts.append(tri["description"])

        concat=key_texts+val_texts
        print(concat)
        concat_emb = self.base_embeder.encode(concat, convert_to_numpy=True, normalize_embeddings=True)  # (B, D)
        key_emb_base=concat_emb[:len(key_texts)]
        val_emb_base=concat_emb[len(key_texts):]

        # key_emb_base = self.base_embeder.encode(key_texts, convert_to_numpy=True, normalize_embeddings=True)  # (B, D)
        # val_emb_base = self.base_embeder.encode(val_texts, convert_to_numpy=True, normalize_embeddings=True)  # (B, D)
        
        kb_keys=self.encoder.encode_key(base_emb=key_emb_base)
        kb_vals=self.encoder.encode_val(base_emb=val_emb_base)


        if device is not None:
            kb_keys = kb_keys.to(device)
            kb_vals = kb_vals.to(device)

        N = kb_keys.size(0)

        # ------------------------------------------------
        # 3. Build sparse adjacency (path-local only)
        # ------------------------------------------------
        # Edges: (0->1), (2->3), (4->5), ...
        adj_device = device if device is not None else kb_keys.device

        rows = torch.arange(0, N, hop_num, device=adj_device)
        cols = rows + 1


        indices = torch.stack([rows, cols])   # (3, num_edges)
        values = torch.ones(rows.size(0), dtype=kb_keys.dtype, device=adj_device)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=(N, N),
            dtype=kb_keys.dtype,
            device=adj_device,
        ).coalesce()

        return kb_keys, kb_vals, kb_adj


    # load QA datasets, but compute embeddings online
    def get_embeddings_with_adj_2wiki_at2qa(
        self,
        batch_indices,
        kb_size: int | None = None,
        step: int | None = None,
        device: torch.device | None = None,
        hop_num: int = 2,
    ):
        key_texts = []
        val_texts = []

        for sample_id in batch_indices:
            sample=self.dataset[sample_id]
            if len(sample["triple_lists"]) != 2:
                raise ValueError(f"Sample {sample['id']} has {len(sample['triple_lists'])} triple lists, expected 2.")
            for triple in sample["triple_lists"]:
                key_texts.append(triple["key_string"])
                val_texts.append(triple["description"])

        print(f"key_texts: {key_texts}")
        print(f"val_texts: {val_texts}")

        concat=key_texts+val_texts
        concat_emb = self.base_embeder.encode(concat, convert_to_numpy=True, normalize_embeddings=True)  # (B, D)
        key_emb_base=concat_emb[:len(key_texts)]
        val_emb_base=concat_emb[len(key_texts):]

        print(f"key_emb_base: {key_emb_base}")
        print(f"val_emb_base: {val_emb_base}")
        # 统计key_emb_base的均值和方差
        print(f"key_emb_base mean: {np.mean(key_emb_base)}")
        print(f"key_emb_base std: {np.std(key_emb_base)}")
        # 统计val_emb_base的均值和方差
        print(f"val_emb_base mean: {np.mean(val_emb_base)}")
        print(f"val_emb_base std: {np.std(val_emb_base)}")

        # key_emb_base = self.base_embeder.encode(key_texts, convert_to_numpy=True, normalize_embeddings=True)  # (B, D)
        # val_emb_base = self.base_embeder.encode(val_texts, convert_to_numpy=True, normalize_embeddings=True)  # (B, D)
        
        kb_keys=self.encoder.encode_key(base_emb=key_emb_base)
        kb_vals=self.encoder.encode_val(base_emb=val_emb_base)
        
        print(f"kb_keys: {kb_keys}")
        print(f"kb_vals: {kb_vals}")


        if device is not None:
            kb_keys = kb_keys.to(device)
            kb_vals = kb_vals.to(device)

        N = kb_keys.size(0)

        # ------------------------------------------------
        # 3. Build sparse adjacency (path-local only)
        # ------------------------------------------------
        # Edges: (0->1), (2->3), (4->5), ...
        adj_device = device if device is not None else kb_keys.device

        rows = torch.arange(0, N, hop_num, device=adj_device)
        cols = rows + 1


        indices = torch.stack([rows, cols])   # (3, num_edges)
        values = torch.ones(rows.size(0), dtype=kb_keys.dtype, device=adj_device)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=(N, N),
            dtype=kb_keys.dtype,
            device=adj_device,
        ).coalesce()
        print(f"kb_adj: {kb_adj}")
        return kb_keys, kb_vals, kb_adj


    def get_embeddings_at2qa_from_precompute(
        self,
        sample_id,
        step: int,
        total_steps: int,
        max_kb_paths: int = 16,
        device: torch.device | None = None,
        hop_num: int = 2,
    ):
        sample=self.dataset[sample_id]
        all_samples=self.dataset

        sr=StageRetriever()
        # decide current stage
        stage = sr.get_stage(step, total_steps)
        
        # instead of get kb_path, get triple_idx
        true_triple_indices=sr.get_triple_ids_stage(stage, sample, all_samples, verbose=True if step % 100 == 0 else False)

        key_emb_base=self.key_embds[true_triple_indices]
        val_emb_base=self.value_embds[true_triple_indices]

        kb_keys=self.encoder.encode_key(base_emb=key_emb_base)
        kb_vals=self.encoder.encode_val(base_emb=val_emb_base)


        if device is not None:
            kb_keys = kb_keys.to(device)
            kb_vals = kb_vals.to(device)

        N = kb_keys.size(0)

        # ------------------------------------------------
        # 3. Build sparse adjacency (path-local only)
        # ------------------------------------------------
        # Edges: (0->1), (2->3), (4->5), ...
        adj_device = device if device is not None else kb_keys.device

        rows = torch.arange(0, N, hop_num, device=adj_device)
        cols = rows + 1


        indices = torch.stack([rows, cols])   # (3, num_edges)
        values = torch.ones(rows.size(0), dtype=kb_keys.dtype, device=adj_device)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=(N, N),
            dtype=kb_keys.dtype,
            device=adj_device,
        ).coalesce()

        return kb_keys, kb_vals, kb_adj

    def build_kb_embedding_and_adj_for_at2qa_batch(
        self,
        sample_ids: list[int],
        step: int,
        total_steps: int,
        max_kb_paths: int = 16,
        hop_num: int = 2,
        device: torch.device | None = None,
        verbose: bool = False,
    ):

        device = device or torch.device("cuda")

        # -----------------------------
        # 1. decide current stage
        # -----------------------------
        sr=StageRetriever()
        stage=sr.get_stage(step, total_steps)
        all_samples = self.dataset
        all_kb_paths = []          # flattened list of paths


        # -----------------------------
        # 3. collect kb_paths for each sample
        # -----------------------------
        for sid in sample_ids:
            sample = all_samples[sid]
            kb_paths=sr.get_path_stage(stage, sample, all_samples, max_kb_paths)

            # ---- flatten ----
            for pid, path in enumerate(kb_paths):
                if path is None:
                    continue
                all_kb_paths.append(path)

        if verbose:
            print(f"[AT2QA-Batch] stage={stage}, "
                f"samples={len(sample_ids)}, "
                f"paths={len(all_kb_paths)}")

        key_texts = []
        val_texts = []

        for path in all_kb_paths:
            for hop in range(hop_num):
                key_texts.append(path[hop]["key_string"])
                val_texts.append(path[hop]["description"])
        
        concat=key_texts+val_texts
        concat_emb = self.base_embeder.encode(concat, convert_to_numpy=True, normalize_embeddings=True)  # (B, D)
        key_emb_base=concat_emb[:len(key_texts)]
        val_emb_base=concat_emb[len(key_texts):]

        # key_emb_base = self.base_embeder.encode(key_texts, convert_to_numpy=True)  # (B, D)
        # val_emb_base = self.base_embeder.encode(val_texts, convert_to_numpy=True)  # (B, D)
        kb_keys=self.encoder.encode_key(base_emb=key_emb_base)
        kb_vals=self.encoder.encode_val(base_emb=val_emb_base)


        # -----------------------------
        # 5. build global sparse adj
        # -----------------------------
        # for each path: connect hop0 -> hop1
        num_paths = len(all_kb_paths)
        total_nodes = num_paths * hop_num

        rows = torch.arange(0, total_nodes, hop_num, device=device)
        cols = rows + 1

        indices = torch.stack([rows, cols])
        values = torch.ones(rows.size(0), device=device)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            (total_nodes, total_nodes),
            device=device,
        ).coalesce()

        return kb_keys, kb_vals, kb_adj
    def get_embeddings_at2qa_from_precompute_batch(
        self,
        sample_ids: list[int],
        step: int,
        total_steps: int,
        max_kb_paths: int = 16,
        device: torch.device | None = None,
        hop_num: int = 2,
    ):
        all_samples=self.dataset
        # sr=StageRetriever()
        # stage = sr.get_stage(step, total_steps)

        all_true_triple_indices=[]
        for sid in sample_ids:
            sample = all_samples[sid]
            # true_triple_indices=sr.get_triple_ids_stage(stage, sample, all_samples, verbose=True if step % 100 == 0 and sid == sample_ids[0] else False)
            # true_triple_indices = get_triple_ids_T1(sample, all_samples, verbose=True if step % 100 == 0 and sid == sample_ids[0] else False)
            true_triple_indices = get_triple_ids_T2(sample, all_samples, step, total_steps, max_kb_paths, verbose=True if step % 100 == 0 and sid == sample_ids[0] else False)
            all_true_triple_indices.extend(true_triple_indices)
                
        key_emb_base=self.key_embds[all_true_triple_indices]
        val_emb_base=self.value_embds[all_true_triple_indices]

        kb_keys=self.encoder.encode_key(base_emb=key_emb_base)
        kb_vals=self.encoder.encode_val(base_emb=val_emb_base)


        if device is not None:
            kb_keys = kb_keys.to(device)
            kb_vals = kb_vals.to(device)

        N = kb_keys.size(0)

        # ------------------------------------------------
        # 3. Build sparse adjacency (path-local only)
        # ------------------------------------------------
        # Edges: (0->1), (2->3), (4->5), ...
        adj_device = device if device is not None else kb_keys.device

        rows = torch.arange(0, N, hop_num, device=adj_device)
        cols = rows + 1


        indices = torch.stack([rows, cols])   # (3, num_edges)
        values = torch.ones(rows.size(0), dtype=kb_keys.dtype, device=adj_device)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=(N, N),
            dtype=kb_keys.dtype,
            device=adj_device,
        ).coalesce()

        return kb_keys, kb_vals, kb_adj 