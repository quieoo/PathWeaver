from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from kblam.kb_encoder import KBEncoder


@dataclass
class DAGSample:
    key_texts: List[str]
    value_texts: List[str]
    adj: np.ndarray
    scores: Optional[np.ndarray]


def _to_numpy_adj(adj_like: Any, n: int) -> np.ndarray:
    adj = np.asarray(adj_like, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"dag.adj must be square, got shape={adj.shape}")
    if adj.shape[0] != n:
        raise ValueError(f"dag.adj size {adj.shape[0]} does not match kv_nodes size {n}")
    return adj


def _longest_path_edges(binary_adj: np.ndarray) -> int:
    n = binary_adj.shape[0]
    if n == 0:
        return 0

    out_neighbors = [np.where(binary_adj[i] > 0)[0].tolist() for i in range(n)]
    indeg = [int(np.sum(binary_adj[:, j] > 0)) for j in range(n)]

    queue = [i for i in range(n) if indeg[i] == 0]
    dp = [0] * n
    visited = 0

    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        visited += 1
        for v in out_neighbors[u]:
            if dp[v] < dp[u] + 1:
                dp[v] = dp[u] + 1
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if visited != n:
        # Not a DAG (or malformed); fallback to 1-hop.
        return 1
    return max(dp)


def _build_multihop_adj(
    adj: np.ndarray,
    max_hops: int,
    decay: float,
    dynamic_hops_by_longest_path: bool = False,
) -> np.ndarray:
    if max_hops <= 1:
        return adj.astype(np.float32)

    a1 = (adj > 0).astype(np.float32)
    if dynamic_hops_by_longest_path:
        longest_edges = _longest_path_edges(a1)
        max_hops = min(max_hops, max(1, longest_edges))

    result = a1.copy()
    power = a1.copy()
    for hop in range(2, max_hops + 1):
        power = power @ a1
        reach = (power > 0).astype(np.float32)
        result += (decay ** (hop - 1)) * reach
    return result.astype(np.float32)


class DAGKVKBRetriever:
    """
    Retriever for DAG_KV dataset format.

    Expected sample format:
      - question / answer (or Q / A)
      - dag.kv_nodes: list of {key, value, ...}
      - dag.adj: square adjacency aligned with kv_nodes
    """

    def __init__(
        self,
        encoder: KBEncoder,
        dataset: Sequence[Dict[str, Any]],
        base_embeder_path: Optional[str] = None,
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[str] = None,
        key_embds: Optional[np.ndarray] = None,
        value_embds: Optional[np.ndarray] = None,
        max_kv_per_sample: Optional[int] = None,
        use_multihop_adj: bool = False,
        max_hops: int = 1,
        hop_decay: float = 0.5,
        dynamic_hops_by_longest_path: bool = False,
        device: str = "cuda",
    ) -> None:
        self.encoder = encoder
        self.dataset = list(dataset)
        self.device = device

        self.max_kv_per_sample = max_kv_per_sample
        self.use_multihop_adj = use_multihop_adj
        self.max_hops = max_hops
        self.hop_decay = hop_decay
        self.dynamic_hops_by_longest_path = dynamic_hops_by_longest_path

        self.samples: List[DAGSample] = [self._parse_sample(row) for row in self.dataset]
        self.sample_sizes = [len(s.key_texts) for s in self.samples]
        self.sample_offsets = np.cumsum([0] + self.sample_sizes[:-1]).astype(np.int64)

        self.key_embds: Optional[np.ndarray] = None
        self.value_embds: Optional[np.ndarray] = None
        if key_embds is not None and value_embds is not None:
            self.key_embds = np.asarray(key_embds, dtype=np.float32)
            self.value_embds = np.asarray(value_embds, dtype=np.float32)
        elif precomputed_embed_keys_path and precomputed_embed_values_path:
            self.key_embds = np.load(precomputed_embed_keys_path).astype(np.float32)
            self.value_embds = np.load(precomputed_embed_values_path).astype(np.float32)

        self.base_embedder: Optional[SentenceTransformer] = None
        if self.key_embds is None or self.value_embds is None:
            if not base_embeder_path:
                raise ValueError(
                    "Either precomputed embeddings or base_embeder_path must be provided."
                )
            self.base_embedder = SentenceTransformer(base_embeder_path)
            self.base_embedder.to(device)
        else:
            total_expected = int(sum(self.sample_sizes))
            if self.key_embds.shape[0] != total_expected or self.value_embds.shape[0] != total_expected:
                raise ValueError(
                    f"Precomputed embedding size mismatch: expected {total_expected}, "
                    f"got key={self.key_embds.shape[0]}, val={self.value_embds.shape[0]}"
                )
    def is_hnsw_ready(self):
        return False
    def _parse_sample(self, row: Dict[str, Any]) -> DAGSample:
        dag = row.get("dag") or {}
        kv_nodes = dag.get("kv_nodes") or []
        adj_like = dag.get("adj") or []

        key_texts: List[str] = []
        value_texts: List[str] = []
        scores: List[float] = []
        for kv in kv_nodes:
            key_texts.append(str(kv.get("key", "")))
            value_texts.append(str(kv.get("value", "")))
            s = kv.get("score")
            scores.append(float(s) if s is not None else 0.0)

        n = len(kv_nodes)
        if n == 0:
            return DAGSample([], [], np.zeros((0, 0), dtype=np.float32), None)

        adj = _to_numpy_adj(adj_like, n)
        return DAGSample(
            key_texts=key_texts,
            value_texts=value_texts,
            adj=adj,
            scores=np.asarray(scores, dtype=np.float32),
        )

    def _select_indices(self, sample: DAGSample) -> np.ndarray:
        n = len(sample.key_texts)
        if n == 0:
            return np.asarray([], dtype=np.int64)
        if self.max_kv_per_sample is None or self.max_kv_per_sample >= n:
            return np.arange(n, dtype=np.int64)

        if sample.scores is None:
            return np.arange(self.max_kv_per_sample, dtype=np.int64)

        # Keep top-scored nodes while preserving original order for stable node ids.
        top = np.argsort(-sample.scores)[: self.max_kv_per_sample]
        return np.sort(top.astype(np.int64))

    def _build_sample_tensors(
        self,
        sample_id: int,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[sample_id]
        select_idx = self._select_indices(sample)
        if select_idx.size == 0:
            dev = device or torch.device(self.device)
            empty = torch.zeros((0, self.encoder.out_dim), dtype=torch.bfloat16, device=dev)
            empty_adj = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.bfloat16, device=dev),
                (0, 0),
                dtype=torch.bfloat16,
                device=dev,
            ).coalesce()
            return empty, empty, empty_adj

        keys = [sample.key_texts[i] for i in select_idx]
        vals = [sample.value_texts[i] for i in select_idx]
        adj = sample.adj[np.ix_(select_idx, select_idx)].astype(np.float32)

        if self.use_multihop_adj and self.max_hops > 1:
            adj = _build_multihop_adj(
                adj=adj,
                max_hops=self.max_hops,
                decay=self.hop_decay,
                dynamic_hops_by_longest_path=self.dynamic_hops_by_longest_path,
            )
        # print("------debug-adj------")
        # print(sample)
        # print(adj)
        # print("---------------------")


        if self.key_embds is not None and self.value_embds is not None:
            offset = int(self.sample_offsets[sample_id])
            base_idx = offset + select_idx
            key_base = self.key_embds[base_idx]
            val_base = self.value_embds[base_idx]
        else:
            assert self.base_embedder is not None
            concat = keys + vals
            base_emb = self.base_embedder.encode(
                concat,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            key_base = base_emb[: len(keys)]
            val_base = base_emb[len(keys) :]

        kb_keys = self.encoder.encode_key(base_emb=key_base)
        kb_vals = self.encoder.encode_val(base_emb=val_base)

        dev = device or kb_keys.device
        kb_keys = kb_keys.to(dev)
        kb_vals = kb_vals.to(dev)

        nz = np.where(adj > 0)
        if nz[0].size == 0:
            indices = torch.empty((2, 0), dtype=torch.long, device=dev)
            values = torch.empty((0,), dtype=kb_keys.dtype, device=dev)
        else:
            indices = torch.tensor(np.vstack(nz), dtype=torch.long, device=dev)
            values = torch.tensor(adj[nz], dtype=kb_keys.dtype, device=dev)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            (adj.shape[0], adj.shape[1]),
            dtype=kb_keys.dtype,
            device=dev,
        ).coalesce()

        return kb_keys, kb_vals, kb_adj

    def get_kb_embedding(
        self,
        sample_id: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._build_sample_tensors(sample_id=sample_id, device=device)

    def get_kb_embedding_s(
        self,
        sample_ids: Sequence[int],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build one shared KB graph for a batch by block-diagonal concatenation.
        """
        dev = device or torch.device(self.device)
        key_blocks: List[torch.Tensor] = []
        val_blocks: List[torch.Tensor] = []
        row_parts: List[torch.Tensor] = []
        col_parts: List[torch.Tensor] = []
        val_parts: List[torch.Tensor] = []

        offset = 0
        dtype = torch.bfloat16
        for sid in sample_ids:
            kb_k, kb_v, kb_adj = self._build_sample_tensors(int(sid), dev)
            key_blocks.append(kb_k)
            val_blocks.append(kb_v)
            dtype = kb_k.dtype

            if kb_adj._nnz() > 0:
                idx = kb_adj.indices()
                row_parts.append(idx[0] + offset)
                col_parts.append(idx[1] + offset)
                val_parts.append(kb_adj.values())
            offset += kb_k.shape[0]

        if key_blocks:
            kb_keys = torch.cat(key_blocks, dim=0)
            kb_vals = torch.cat(val_blocks, dim=0)
        else:
            kb_keys = torch.zeros((0, self.encoder.out_dim), dtype=dtype, device=dev)
            kb_vals = torch.zeros((0, self.encoder.out_dim), dtype=dtype, device=dev)

        if row_parts:
            rows = torch.cat(row_parts, dim=0)
            cols = torch.cat(col_parts, dim=0)
            values = torch.cat(val_parts, dim=0)
            indices = torch.stack([rows, cols], dim=0)
        else:
            indices = torch.empty((2, 0), dtype=torch.long, device=dev)
            values = torch.empty((0,), dtype=dtype, device=dev)

        kb_adj = torch.sparse_coo_tensor(
            indices,
            values,
            (kb_keys.shape[0], kb_keys.shape[0]),
            dtype=dtype,
            device=dev,
        ).coalesce()

        return kb_keys, kb_vals, kb_adj
