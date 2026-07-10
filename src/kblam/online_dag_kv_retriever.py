"""End-to-end online DAG retrieval for KBLaM generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from kblam.dag_kv_retriever import _build_multihop_adj
from kblam.dag_store_retriever import DAGKVStoreRetriever, TextEmbedder, TrainableDAGExtractor
from kblam.dag_store_retriever_v2 import DAGKVStoreRetrieverV2


@dataclass(frozen=True)
class OnlineDAGResult:
    kb_keys: torch.Tensor
    kb_values: torch.Tensor
    kb_adj: torch.Tensor
    dag: dict[str, Any]


class OnlineDAGKBRetriever:
    """Compose entity retrieval, answer-blind DAG extraction, and KV projection."""

    is_online_retriever = True

    def __init__(
        self,
        *,
        encoder,
        store_dir: str,
        entity_embedder: TextEmbedder,
        dag_extractor: TrainableDAGExtractor,
        entity_top_k: int = 1,
        subgraph_hops: int = 2,
        max_triples_per_seed: int | None = None,
        max_incident_triples_per_node: int | None = None,
        search_backend: str = "hnsw",
        query_prompt_name: str | None = None,
        seed_strategy: str = "vector",
        mention_min_chars: int = 8,
        store_version: str = "v1",
        entity_candidate_top_k: int = 64,
        use_multihop_adj: bool = True,
        max_hops: int = 10,
        hop_decay: float = 1.0,
        dynamic_hops_by_longest_path: bool = True,
        require_answer_blind: bool = True,
    ) -> None:
        if require_answer_blind and dag_extractor.backend != "v8-answer-blind":
            raise ValueError("Online retrieval requires the answer-blind v8 DAG backend")
        self.encoder = encoder
        self.dag_extractor = dag_extractor
        if store_version not in {"v1", "v2"}:
            raise ValueError("store_version must be 'v1' or 'v2'")
        if store_version == "v2":
            self.store_retriever = DAGKVStoreRetrieverV2(
                store_dir,
                entity_embedder,
                entity_top_k=entity_top_k,
                entity_candidate_top_k=entity_candidate_top_k,
                subgraph_hops=subgraph_hops,
                max_triples_per_seed=max_triples_per_seed,
                max_incident_triples_per_node=max_incident_triples_per_node,
                search_backend=search_backend,
                query_prompt_name=query_prompt_name,
                seed_strategy=seed_strategy,
                mention_min_chars=mention_min_chars,
            )
        else:
            self.store_retriever = DAGKVStoreRetriever(
                store_dir,
                entity_embedder,
                entity_top_k=entity_top_k,
                subgraph_hops=subgraph_hops,
                max_triples_per_seed=max_triples_per_seed,
                max_incident_triples_per_node=max_incident_triples_per_node,
                search_backend=search_backend,
                query_prompt_name=query_prompt_name,
                seed_strategy=seed_strategy,
                mention_min_chars=mention_min_chars,
            )
        self.store_version = store_version
        self.use_multihop_adj = use_multihop_adj
        self.max_hops = max_hops
        self.hop_decay = hop_decay
        self.dynamic_hops_by_longest_path = dynamic_hops_by_longest_path
        self._samples = 0
        self._empty_dags = 0
        self._stage_seconds = {"candidate": 0.0, "dag": 0.0, "tensor": 0.0, "total": 0.0}
        self._dag_stage_seconds = {
            "build_graph": 0.0,
            "encode": 0.0,
            "feature_prepare": 0.0,
            "model_score": 0.0,
            "select_export": 0.0,
            "total": 0.0,
        }

    def close(self) -> None:
        self.store_retriever.close()

    def is_hnsw_ready(self) -> bool:
        return self.store_retriever.graph_store.hnsw_index_path.exists()

    @torch.no_grad()
    def get_kb_for_queries(
        self,
        questions: Sequence[str],
        *,
        device: torch.device | str,
    ) -> list[OnlineDAGResult]:
        if not questions:
            return []
        batch_started = time.perf_counter()

        stage_started = time.perf_counter()
        candidates = self.store_retriever.retrieve_many([str(question) for question in questions])
        prepared = []
        retrieval_metadata = []
        for index, (question, candidate) in enumerate(zip(questions, candidates)):
            # Gold answers and supporting facts never enter the answer-blind DAG backend.
            sample = {"question": str(question), "__online_index": index}
            candidate_sample = self.store_retriever.build_candidate_sample(sample, candidate)
            prepared.append(candidate_sample)
            retrieval_metadata.append(candidate.metadata())
        candidate_seconds = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        extracted = self.dag_extractor.extract(prepared)
        dag_seconds = time.perf_counter() - stage_started
        dag_profile = getattr(self.dag_extractor, "last_profile", None) or {}
        extracted.sort(key=lambda row: int(row["__online_index"]))

        stage_started = time.perf_counter()
        offsets_by_sample = [self._dag_offsets(row.get("dag") or {}) for row in extracted]
        flat_offsets = [offset for offsets in offsets_by_sample for offset in offsets]
        if flat_offsets:
            key_base, value_base = self.store_retriever.kv_store.get_tensors(flat_offsets)
            projected_keys = self.encoder.encode_key(base_emb=np.asarray(key_base, dtype=np.float32))
            projected_values = self.encoder.encode_val(base_emb=np.asarray(value_base, dtype=np.float32))
            projected_keys = projected_keys.to(device)
            projected_values = projected_values.to(device)
        else:
            projected_keys = projected_values = None

        results = []
        cursor = 0
        for index, (row, offsets) in enumerate(zip(extracted, offsets_by_sample)):
            length = len(offsets)
            if length:
                assert projected_keys is not None and projected_values is not None
                kb_keys = projected_keys[cursor : cursor + length]
                kb_values = projected_values[cursor : cursor + length]
            else:
                kb_keys, kb_values = self._empty_kv_tensors(device)
            cursor += length

            dag = row.get("dag") or {}
            dag.setdefault("meta", {})["retrieval"] = retrieval_metadata[index]
            kb_adj = self._build_sparse_adj(dag.get("adj") or [], length, device, kb_keys.dtype)
            results.append(OnlineDAGResult(kb_keys, kb_values, kb_adj, dag))

        tensor_seconds = time.perf_counter() - stage_started
        total_seconds = time.perf_counter() - batch_started
        self._samples += len(questions)
        self._empty_dags += sum(not result.dag.get("kv_nodes") for result in results)
        self._stage_seconds["candidate"] += candidate_seconds
        self._stage_seconds["dag"] += dag_seconds
        self._stage_seconds["tensor"] += tensor_seconds
        self._stage_seconds["total"] += total_seconds
        for stage in self._dag_stage_seconds:
            self._dag_stage_seconds[stage] += float(dag_profile.get(stage, 0.0))
        return results

    def get_avg_retrieval_time(self) -> float:
        return self._stage_seconds["total"] / max(1, self._samples)

    def stats(self) -> dict[str, Any]:
        denominator = max(1, self._samples)
        return {
            "samples": self._samples,
            "empty_dags": self._empty_dags,
            "average_seconds": {
                stage: elapsed / denominator for stage, elapsed in self._stage_seconds.items()
            },
            "dag_average_seconds": {
                stage: elapsed / denominator for stage, elapsed in self._dag_stage_seconds.items()
            },
        }

    def print_metrics(self) -> None:
        stats = self.stats()
        timings = stats["average_seconds"]
        dag_timings = stats["dag_average_seconds"]
        print(
            "[OnlineDAG] "
            f"samples={stats['samples']} empty_dags={stats['empty_dags']} "
            f"candidate={timings['candidate'] * 1000:.2f}ms "
            f"dag={timings['dag'] * 1000:.2f}ms "
            f"tensor={timings['tensor'] * 1000:.2f}ms "
            f"total={timings['total'] * 1000:.2f}ms "
            f"| dag_substages build_graph={dag_timings['build_graph'] * 1000:.2f}ms "
            f"encode={dag_timings['encode'] * 1000:.2f}ms "
            f"feature_prepare={dag_timings['feature_prepare'] * 1000:.2f}ms "
            f"model_score={dag_timings['model_score'] * 1000:.2f}ms "
            f"select_export={dag_timings['select_export'] * 1000:.2f}ms"
        )

    @staticmethod
    def _dag_offsets(dag: dict[str, Any]) -> list[int]:
        offsets = []
        for index, node in enumerate(dag.get("kv_nodes") or []):
            if node.get("kv_offset") is None:
                raise RuntimeError(f"DAG node {index} is missing kv_offset")
            offsets.append(int(node["kv_offset"]))
        return offsets

    def _empty_kv_tensors(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = next(self.encoder.parameters()).dtype
        shape = (0, int(self.encoder.out_dim))
        empty = torch.empty(shape, dtype=dtype, device=device)
        return empty, empty.clone()

    def _build_sparse_adj(
        self,
        adj_like: Any,
        size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if size == 0:
            indices = torch.empty((2, 0), dtype=torch.long, device=device)
            values = torch.empty((0,), dtype=dtype, device=device)
            return torch.sparse_coo_tensor(indices, values, (0, 0), device=device, dtype=dtype).coalesce()

        adj = np.asarray(adj_like, dtype=np.float32)
        if adj.shape != (size, size):
            raise ValueError(f"DAG adjacency shape {adj.shape} does not match {size} KV nodes")
        if self.use_multihop_adj and self.max_hops > 1:
            adj = _build_multihop_adj(
                adj,
                max_hops=self.max_hops,
                decay=self.hop_decay,
                dynamic_hops_by_longest_path=self.dynamic_hops_by_longest_path,
            )
        rows, columns = np.where(adj > 0)
        indices = torch.as_tensor(np.vstack((rows, columns)), dtype=torch.long, device=device)
        values = torch.as_tensor(adj[rows, columns], dtype=dtype, device=device)
        return torch.sparse_coo_tensor(indices, values, (size, size), device=device, dtype=dtype).coalesce()
