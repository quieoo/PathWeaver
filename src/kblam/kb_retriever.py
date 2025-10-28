import numpy as np
import torch
from typing import List, Dict, Optional

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
        # context_set_val = torch.randn_like(context_set_val)
        # Idea: Try torch.randn here context_set_tokens??

        true_kb_copy = 1
        kb_embedding = (
            torch.concat([*([train_set_key] * true_kb_copy), context_set_key], 1),
            torch.concat([*([train_set_val] * true_kb_copy), context_set_val], 1),
        )
        # (batch_size, 1+context_set_size, embedding_dim)
        return kb_embedding


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
        if not self._use_cached_embd():
            print("Currently only supports cached KB embedding")
            return None

        if len(start_id_lists) != batch_size or len(num_triples_lists) != batch_size:
            print("Batch size mismatch")
            return None

        # Step 1: Collect all indices needed across the batch
        all_indices = []
        seq_lengths = []  # number of triples per sample

        for i in range(batch_size):
            starts = start_id_lists[i] if isinstance(start_id_lists[i], list) else [start_id_lists[i]]
            nums = num_triples_lists[i] if isinstance(num_triples_lists[i], list) else [num_triples_lists[i]]

            sample_indices = []
            for start, num in zip(starts, nums):
                if num > 0:
                    # Ensure we don't go out of bounds (optional safety check)
                    if start + num > len(self.key_embds):
                        raise IndexError(f"Index out of range: start={start}, num={num}, total={len(self.key_embds)}")
                    sample_indices.extend(range(start, start + num))
            all_indices.extend(sample_indices)
            seq_lengths.append(len(sample_indices))

        total_triples = len(all_indices)
        if total_triples == 0:
            print("WARNING: No triples found in batch.")
            # Edge case: no triples in entire batch
            # Create dummy tensors with correct embedding dim
            dummy_key = self.encoder.encode_key(base_emb=np.zeros_like(self.key_embds[0:1]))  # (1, Dk)
            dummy_val = self.encoder.encode_val(base_emb=np.zeros_like(self.value_embds[0:1]))  # (1, Dv)
            dim_k = dummy_key.shape[1]
            dim_v = dummy_val.shape[1]
            max_len = 1
            padded_keys = torch.zeros(batch_size, max_len, dim_k, dtype=dummy_key.dtype, device=dummy_key.device)
            padded_vals = torch.zeros(batch_size, max_len, dim_v, dtype=dummy_val.dtype, device=dummy_val.device)
            return padded_keys, padded_vals

        # Step 2: Batch extract embeddings (as numpy)
        key_batch_np = self.key_embds[all_indices]      # (total_triples, Dk)
        val_batch_np = self.value_embds[all_indices]    # (total_triples, Dv)

        # Step 3: Batch encode via encoder (only 2 calls!)
        key_encoded = self.encoder.encode_key(base_emb=key_batch_np)    # (total_triples, Dk')
        val_encoded = self.encoder.encode_val(base_emb=val_batch_np)  # (total_triples, Dv')

        # Step 4: Split into per-sample sequences
        key_seq_list = []
        val_seq_list = []
        start = 0
        for length in seq_lengths:
            if length == 0:
                # Create empty tensor with correct feature dim
                k = torch.empty(0, key_encoded.shape[1], device=key_encoded.device, dtype=key_encoded.dtype)
                v = torch.empty(0, val_encoded.shape[1], device=val_encoded.device, dtype=val_encoded.dtype)
            else:
                k = key_encoded[start:start + length]
                v = val_encoded[start:start + length]
                start += length
            key_seq_list.append(k)
            val_seq_list.append(v)

        # if is_inference:
        #     kv_list = list(zip(key_seq_list, val_seq_list))
        #     return kv_list

        # Step 5: Post-padding (pad at the end of sequence)
        max_seq_len = max(t.size(0) for t in key_seq_list)
        padded_keys = []
        padded_vals = []

        for k, v in zip(key_seq_list, val_seq_list):
            cur_len = k.size(0)
            if cur_len < max_seq_len:
                pad_len = max_seq_len - cur_len
                k_pad = torch.zeros(pad_len, k.size(1), dtype=k.dtype, device=k.device)
                v_pad = torch.zeros(pad_len, v.size(1), dtype=v.dtype, device=v.device)
                k = torch.cat([k, k_pad], dim=0)
                v = torch.cat([v, v_pad], dim=0)
            padded_keys.append(k)
            padded_vals.append(v)

        return torch.stack(padded_keys, dim=0), torch.stack(padded_vals, dim=0)