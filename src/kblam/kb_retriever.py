import numpy as np
import torch
from typing import List, Dict, Optional
import random
from kblam.kb_encoder import KBEncoder
from kblam.utils.train_utils import context_set_size_scheduler, get_kb_embd


class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray] = None,
        value_embds: Optional[np.ndarray] = None,
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[str] = None,
    ):

        self.encoder = encoder
        self.dataset = dataset
        
        if precomputed_embed_keys_path is not None and precomputed_embed_values_path is not None:
            self._load_cached_embd(precomputed_embed_keys_path, precomputed_embed_values_path)
        else:
            self.key_embds = key_embds
            self.value_embds = value_embds

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

        # 为每个样本随机插入位置
        B = batch_size
        C = context_set_key.size(1)
        insert_pos = torch.randint(0, C + 1, (B,), device=context_set_key.device)

        new_keys = []
        new_vals = []
        for b in range(B):
            # 在随机位置插入 train_set_key[b]
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
        # === 🟩 新增部分结束 ===

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


# DATASET_SUPPORT
    def get_key_embeddings_document(self, start_ids, num_triples, batch_size, step, kb_size):
        if not self._use_cached_embd():
            print("Current only supports cached KB embedding")
            return None
        if len(start_ids) != batch_size:
            print("Batch size mismatch")
            return None

        # 先收集三元组并执行编码
        key_embeddings = [[] for _ in range(batch_size)]
        value_embeddings = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            start_id = start_ids[i]
            for j in range(num_triples[i]):
                k_embed = self.encoder.encode_key(base_emb=self.key_embds[start_id + j])  # pyright: ignore[reportOptionalSubscript]
                v_embed = self.encoder.encode_val(base_emb=self.value_embds[start_id + j])  # pyright: ignore[reportOptionalSubscript]
                key_embeddings[i].append(k_embed)
                value_embeddings[i].append(v_embed)

        # 处理变长序列：首先将每个样本的嵌入堆叠成张量
        # 然后进行填充使所有样本具有相同的序列长度
        key_tensor_list = []
        value_tensor_list = []
        
        for i in range(batch_size):
            # 将每个样本的嵌入列表转换为张量
            key_tensor_list.append(torch.stack(key_embeddings[i]))
            value_tensor_list.append(torch.stack(value_embeddings[i]))
            
        # 获取最大序列长度
        max_seq_len = max([t.size(0) for t in key_tensor_list])
        
        # 对所有张量进行填充以匹配最大序列长度
        padded_key_tensors = []
        padded_value_tensors = []
        
        for i in range(batch_size):
            current_seq_len = key_tensor_list[i].size(0)
            if current_seq_len < max_seq_len:
                # 计算需要填充的数量
                padding_size = max_seq_len - current_seq_len
                # 创建填充张量（使用0填充）
                key_padding = torch.zeros(padding_size, key_tensor_list[i].size(1), 
                                          dtype=key_tensor_list[i].dtype, 
                                          device=key_tensor_list[i].device)
                value_padding = torch.zeros(padding_size, value_tensor_list[i].size(1), 
                                            dtype=value_tensor_list[i].dtype, 
                                            device=value_tensor_list[i].device)
                # 拼接原始张量和填充张量
                padded_key = torch.cat([key_tensor_list[i], key_padding], dim=0)
                padded_value = torch.cat([value_tensor_list[i], value_padding], dim=0)
            else:
                # 如果已经是最大长度，则不需要填充
                padded_key = key_tensor_list[i]
                padded_value = value_tensor_list[i]
                
            padded_key_tensors.append(padded_key)
            padded_value_tensors.append(padded_value)
        
        # 最后堆叠所有样本的张量
        key_embeddings = torch.stack(padded_key_tensors, dim=0)
        value_embeddings = torch.stack(padded_value_tensors, dim=0)

        # print(f"----shape of key embeddings: {key_embeddings.shape}")
        return (key_embeddings, value_embeddings)
    def get_embeddings(self, start_id_lists, num_triples_lists, batch_size, is_inference=False):
        import random
        if not self._use_cached_embd():
            raise RuntimeError("Only supports cached KB embeddings")

        if len(start_id_lists) != batch_size or len(num_triples_lists) != batch_size:
            raise ValueError("Batch size mismatch")

        all_indices, seq_lengths = [], []
        total_keys = len(self.key_embds)

        for i in range(batch_size):
            starts = start_id_lists[i] if isinstance(start_id_lists[i], list) else [start_id_lists[i]]
            nums = num_triples_lists[i] if isinstance(num_triples_lists[i], list) else [num_triples_lists[i]]

            sample_indices = []
            for start, num in zip(starts, nums):
                if num > 0:
                    if start < 0 or start + num > total_keys:
                        raise IndexError(f"Triple index out of range: {start}-{start+num}, total={total_keys}")
                    sample_indices.extend(range(start, start + num))

            if not is_inference and len(sample_indices) > 1:
                random.shuffle(sample_indices)

            all_indices.extend(sample_indices)
            seq_lengths.append(len(sample_indices))

        total_triples = len(all_indices)
        if total_triples == 0:
            dummy_key = self.encoder.encode_key(base_emb=np.zeros_like(self.key_embds[0:1]))
            dummy_val = self.encoder.encode_val(base_emb=np.zeros_like(self.value_embds[0:1]))
            dim_k, dim_v = dummy_key.shape[1], dummy_val.shape[1]
            return (
                torch.zeros(batch_size, 1, dim_k, device=dummy_key.device, dtype=dummy_key.dtype),
                torch.zeros(batch_size, 1, dim_v, device=dummy_val.device, dtype=dummy_val.dtype),
            )

        key_batch_np = self.key_embds[all_indices]
        val_batch_np = self.value_embds[all_indices]
        key_encoded = self.encoder.encode_key(base_emb=key_batch_np)
        val_encoded = self.encoder.encode_val(base_emb=val_batch_np)

        key_seq_list, val_seq_list = [], []
        offset = 0
        for length in seq_lengths:
            k = key_encoded[offset:offset + length]
            v = val_encoded[offset:offset + length]
            offset += length
            key_seq_list.append(k)
            val_seq_list.append(v)

        if not is_inference:
            for i, (k, v) in enumerate(zip(key_seq_list, val_seq_list)):
                if k.size(0) > 1:
                    perm = torch.randperm(k.size(0), device=k.device)
                    key_seq_list[i] = k.index_select(0, perm)
                    val_seq_list[i] = v.index_select(0, perm)

        max_len = max(k.size(0) for k in key_seq_list) or 1
        padded_keys, padded_vals = [], []
        for k, v in zip(key_seq_list, val_seq_list):
            pad_k = torch.zeros(max_len - k.size(0), k.size(1), dtype=k.dtype, device=k.device)
            pad_v = torch.zeros(max_len - v.size(0), v.size(1), dtype=v.dtype, device=v.device)
            padded_keys.append(torch.cat([k, pad_k], dim=0))
            padded_vals.append(torch.cat([v, pad_v], dim=0))

        return torch.stack(padded_keys, dim=0), torch.stack(padded_vals, dim=0)

