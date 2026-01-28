import numpy as np
import torch
import random
from typing import List, Dict, Optional
from tqdm import tqdm

from kblam.kb_encoder import KBEncoder


class AutoSchemaKGKBRetriever:
    """
    KB Retriever for AutoSchemaKG-generated KBLaM Path data.
    Training / Validation only.
    """

    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        base_embedder_path: Optional[str] = None,
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[str] = None,
        key_embds: Optional[np.ndarray] = None,
        value_embds: Optional[np.ndarray] = None,
        device: str = "cuda",
    ):
        self.encoder = encoder
        self.dataset = dataset
        self.base_embedder_path = base_embedder_path
        self.base_embedder = None

        self.device = device

        # ---- global storage ----
        self.key_strings: List[str] = []
        self.value_strings: List[str] = []

        # sample_id -> List[(path_start, path_len)]
        self.sample_path_offsets: List[List[tuple]] = []

        # ---- build flat KB table ----
        self._build_flat_kb_table()

        # ---- embeddings ----
        if key_embds is not None and value_embds is not None:
            self.key_embeds = key_embds.astype("float32")
            self.val_embeds = value_embds.astype("float32")
        else: 
            if precomputed_embed_keys_path and precomputed_embed_values_path:
                self.key_embeds = np.load(precomputed_embed_keys_path).astype("float32")
                self.val_embeds = np.load(precomputed_embed_values_path).astype("float32")
            else:
                self._compute_all_embeddings()

            assert len(self.key_strings) == self.key_embeds.shape[0]
            assert len(self.value_strings) == self.val_embeds.shape[0]

    # ------------------------------------------------
    # Step 1: flatten paths & record offsets
    # ------------------------------------------------
    def _build_flat_kb_table(self):
        """
        Flatten all paths across all samples into a single KB table,
        while recording per-sample path offsets.
        """
        cursor = 0

        for sample in self.dataset:
            path_offsets = []

            paths = sample["AutoSchemaKG"]["Paths"]
            for path in paths:
                start = cursor
                for kv in path:
                    self.key_strings.append(kv["key_string"])
                    self.value_strings.append(kv["description"])
                    cursor += 1
                path_len = len(path)
                path_offsets.append((start, path_len))

            self.sample_path_offsets.append(path_offsets)

    # ------------------------------------------------
    # Step 2: embedding computation
    # ------------------------------------------------
    def _compute_all_embeddings(self, batch_size: int = 256):
        """
        Compute embeddings for all key_string / description pairs.
        """
        print("[AutoSchemaKGKBRetriever] Computing KB embeddings...")

        self.base_embedder = SentenceTransformer(self.base_embedder_path, device="cuda")

        key_embs = []
        val_embs = []

        for i in tqdm(range(0, len(self.key_strings), batch_size)):
            ks = self.key_strings[i : i + batch_size]
            vs = self.value_strings[i : i + batch_size]

            key_embs.append(
                self.base_embedder.encode(
                    ks, convert_to_numpy=True, normalize_embeddings=True
                )
            )
            val_embs.append(
                self.base_embedder.encode(
                    vs, convert_to_numpy=True, normalize_embeddings=True
                )
            )

        self.key_embeds = np.vstack(key_embs).astype("float32")
        self.val_embeds = np.vstack(val_embs).astype("float32")

    # ------------------------------------------------
    # Core API
    # ------------------------------------------------
    def get_kb_embedding(self, sample_id: int):
        """
        For a given sample_id:
          - select 1 gold path
          - select 10 random paths
          - concatenate their embeddings
          - build path-level adjacency matrix

        Returns:
            kb_keys:  (N, d)
            kb_vals:  (N, d)
            kb_adj:   sparse (N, N)
        """
        path_offsets = self.sample_path_offsets[sample_id]
        gold_num = self.dataset[sample_id]["AutoSchemaKG"]["Gold-Path-Num"]

        # some samples have no gold path
        # assert gold_num > 0, "Sample has no gold path"

        # -------- 1️⃣ sample paths --------
        # select gold path-0
        # select all random paths
        gold_path_idx=0
        random_paths=list(range(gold_num, len(path_offsets)))
        selected_paths = [gold_path_idx] + random_paths

        # IMPORTANT: shuffle the path order
        random.shuffle(selected_paths)
        

    # -------- 2️⃣ collect BASE embeddings --------
        base_key_list = []
        base_val_list = []
        adj_edges = []

        node_cursor = 0

        for pid in selected_paths:
            start, length = path_offsets[pid]

            base_key_list.append(self.key_embeds[start : start + length])
            base_val_list.append(self.val_embeds[start : start + length])

            for i in range(length - 1):
                adj_edges.append((node_cursor + i, node_cursor + i + 1))

            node_cursor += length

        # (N, D_base)
        base_key_emb = np.vstack(base_key_list)
        base_val_emb = np.vstack(base_val_list)

        # -------- 3️⃣ 🔥 对齐到模型隐藏空间（关键） --------
        kb_keys = self.encoder.encode_key(
            base_emb=torch.from_numpy(base_key_emb).to(self.device)
        )
        kb_vals = self.encoder.encode_val(
            base_emb=torch.from_numpy(base_val_emb).to(self.device)
        )

        # -------- 4️⃣ adjacency --------
        if adj_edges:
            rows, cols = zip(*adj_edges)
            indices = torch.tensor([rows, cols], device=self.device)
            values = torch.ones(len(rows), device=self.device)
        else:
            indices = torch.empty((2, 0), device=self.device)
            values = torch.empty((0,), device=self.device)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            (kb_keys.size(0), kb_keys.size(0)),
            device=self.device,
        ).coalesce()

        return kb_keys, kb_vals, kb_adj


    def get_kb_embedding_s(
        self,
        sample_ids: list[int],
        n_gold: int = 1,
        *,
        device: torch.device | None = None,
        verbose: bool = False,
    ):
        """
        Build ONE global KB for a batch of samples (shared graph).
        """

        base_key_list = []
        base_val_list = []
        adj_edges = []
        node_cursor = 0

        # 可选但很有用：用于 debug/分析（不影响注入）
        node_sample_ids = []
        node_path_ids = []

        for sid in sample_ids:
            path_offsets = self.sample_path_offsets[sid]
            gold_num = self.dataset[sid]["AutoSchemaKG"]["Gold-Path-Num"]

            selected_paths = []
            if gold_num > 0:
                selected_paths.extend(range(min(n_gold, gold_num)))
            if len(selected_paths) < n_gold:
                need = n_gold - len(selected_paths)
                random_candidates = list(range(gold_num, len(path_offsets)))
                selected_paths.extend(random_candidates[:need])
            if len(selected_paths) == 0 and len(path_offsets) > 0:
                selected_paths.append(0)

            # 打乱选择的路径
            # random.shuffle(selected_paths)

            if verbose:
                # 打印选择的路径
                print(f"Sample {sid}")
                print(f"Q: {self.dataset[sid]['Q']}")
                print(f"A: {self.dataset[sid]['A']}")
                print(f"Selected paths: {selected_paths}")
                def print_path(path):
                    for triple in path:
                        print(f"{triple['key_string']} - {triple['description']}")
                    print("-----")
                for pid in selected_paths:
                    path=self.dataset[sid]["AutoSchemaKG"]["Paths"][pid]
                    print_path(path)
                print("==========")
                
            for pid in selected_paths:
                start, length = path_offsets[pid]

                # ---- base embeddings (numpy) ----
                base_key_list.append(self.key_embeds[start:start + length])
                base_val_list.append(self.val_embeds[start:start + length])

                # ---- shared-graph path-chain edges ----
                # edges: (node_cursor+i) -> (node_cursor+i+1)
                for i in range(length - 1):
                    adj_edges.append((node_cursor + i, node_cursor + i + 1))

                node_sample_ids.extend([sid] * length)
                node_path_ids.extend([pid] * length)

                node_cursor += length

        if not base_key_list:
            # 空 KB：返回空张量和空稀疏图
            dev = device if device is not None else self.device
            empty_k = torch.empty((0, 0), device=dev)
            empty_v = torch.empty((0, 0), device=dev)
            empty_adj = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=dev),
                torch.empty((0,), device=dev),
                (0, 0),
                device=dev,
            ).coalesce()
            return empty_k, empty_v, empty_adj

        base_key_emb = np.vstack(base_key_list)  # (kb_len, P)
        base_val_emb = np.vstack(base_val_list)  # (kb_len, P)

        # ------------------------------------------------------------
        # ✅ 关键：确保 encoder 输出是“slots-packed”的 2D:
        #     (kb_len, num_slots * num_heads * head_dim)
        # 这会被 apply_kblam_attention reshape 成 (kb_len, num_slots, -1) 再按层取 kb_idx
        # ------------------------------------------------------------
        # 建议使用与你 get_embeddings_with_adj_2wiki 一致的调用范式：encode_key(base_emb=...)
        kb_keys = self.encoder.encode_key(base_emb=base_key_emb)   # torch.Tensor, dim=2
        kb_vals = self.encoder.encode_val(base_emb=base_val_emb)   # torch.Tensor, dim=2

        # device 对齐（有些 encoder 可能在 CPU 上产出）
        dev = device if device is not None else self.device
        kb_keys = kb_keys.to(dev)
        kb_vals = kb_vals.to(dev)

        # ------------------------------------------------------------
        # ✅ 关键：shared-graph sparse COO adjacency: (K,K) with indices shape (2,E)
        # ------------------------------------------------------------
        K = kb_keys.size(0)
        if len(adj_edges) == 0:
            kb_adj = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=kb_keys.dtype, device=dev),
                (K, K),
                device=dev,
                dtype=kb_keys.dtype,
            ).coalesce()
        else:
            rows, cols = zip(*adj_edges)
            indices = torch.tensor([rows, cols], dtype=torch.long, device=dev)
            values = torch.ones(indices.size(1), dtype=kb_keys.dtype, device=dev)
            kb_adj = torch.sparse_coo_tensor(indices, values, (K, K), device=dev, dtype=kb_keys.dtype).coalesce()

        return kb_keys, kb_vals, kb_adj


    # current version: not support retrieval
    def is_hnsw_ready(self):
        return False