"""Online DAG retrieval backed by Store V2 serving snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from kblam.dag_store_retriever import CandidateGraph, EntityHit
from kblam.stores.common import canonical_entity_key
from kblam.stores.graph_store_v2 import GraphStoreV2
from kblam.stores.kv_store_v2 import KVStoreV2


class TextEmbedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class RetrievalProfile:
    entity_candidate_top_k: int = 64
    mention_min_chars: int = 8


class DAGKVStoreRetrieverV2:
    """Recover query-specific candidate graphs from Store V2."""

    def __init__(
        self,
        store_dir: str | Path,
        embedder: TextEmbedder,
        *,
        entity_top_k: int = 1,
        entity_candidate_top_k: int = 64,
        subgraph_hops: int = 2,
        max_triples_per_seed: int | None = None,
        max_incident_triples_per_node: int | None = None,
        search_backend: str = "auto",
        query_prompt_name: str | None = None,
        seed_strategy: str = "vector",
        mention_min_chars: int = 8,
    ) -> None:
        root = Path(store_dir)
        self.graph_store = GraphStoreV2(root / "graph_v2", create=False)
        self.kv_store = KVStoreV2(root / "kv_v2", create=False)
        self.embedder = embedder
        self.entity_top_k = entity_top_k
        self.entity_candidate_top_k = max(entity_top_k, entity_candidate_top_k)
        self.subgraph_hops = subgraph_hops
        self.max_triples_per_seed = max_triples_per_seed
        self.max_incident_triples_per_node = max_incident_triples_per_node
        self.search_backend = search_backend
        self.query_prompt_name = query_prompt_name
        self.seed_strategy = seed_strategy
        self.mention_min_chars = mention_min_chars

    def __enter__(self) -> "DAGKVStoreRetrieverV2":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.graph_store.close()
        self.kv_store.close()

    def retrieve(self, query: str) -> CandidateGraph:
        return self.retrieve_many([query])[0]

    def retrieve_many(self, queries: Sequence[str]) -> list[CandidateGraph]:
        if not queries:
            return []
        vectors = self._encode_queries(queries)
        return [self._retrieve_from_vector(vector, str(query)) for vector, query in zip(vectors, queries)]

    def build_candidate_sample(
        self,
        sample: dict[str, Any],
        candidate: CandidateGraph,
    ) -> dict[str, Any]:
        prepared = dict(sample)
        prepared["context"] = []
        prepared["triple_list"] = [self._triple_to_raw(triple) for triple in candidate.triples]
        return prepared

    def _encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        vectors = np.asarray(self.embedder.encode(list(queries), **kwargs), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def _retrieve_from_vector(self, query_vector: np.ndarray, query: str | None) -> CandidateGraph:
        hits = tuple(self._entity_hits(query_vector, query))
        node_ids: set[int] = set()
        triples: dict[int, Any] = {}
        for hit in hits:
            local_nodes, local_triples = self.graph_store.get_local_subgraph(
                [hit.node_id],
                hops=self.subgraph_hops,
                max_triples=self.max_triples_per_seed,
                max_incident_triples_per_node=self.max_incident_triples_per_node,
            )
            node_ids.update(local_nodes)
            triples.update((triple.triple_id, triple) for triple in local_triples)
        return CandidateGraph(
            entity_hits=hits,
            node_ids=tuple(sorted(node_ids)),
            triples=tuple(triples[triple_id] for triple_id in sorted(triples)),
        )

    def _entity_hits(self, query_vector: np.ndarray, query: str | None) -> list[EntityHit]:
        if self.seed_strategy == "vector" or not query:
            return [
                EntityHit(node_id=node_id, name=self.graph_store.get_node_name(node_id), score=score, source="vector")
                for node_id, score in self.graph_store.search_entities(
                    query_vector, top_k=self.entity_top_k, backend=self.search_backend
                )
            ]
        candidates = self.graph_store.search_entities(
            query_vector,
            top_k=self.entity_candidate_top_k,
            backend=self.search_backend,
        )
        query_key = f" {canonical_entity_key(query)} "
        shortlisted = self.graph_store.shortlist_entity_hits(
            query_key,
            final_top_k=self.entity_top_k,
            candidate_top_k=self.entity_candidate_top_k,
            min_mention_chars=self.mention_min_chars,
            backend=self.search_backend,
            candidates=candidates,
        )
        return [
            EntityHit(node_id=node_id, name=self.graph_store.get_node_name(node_id), score=score, source=source)
            for node_id, score, source in shortlisted
        ]

    def _triple_to_raw(self, triple) -> dict[str, Any]:
        records = self.kv_store.get_many(triple.kv_offsets)
        return {
            "type": triple.triple_type,
            "name": triple.subject,
            "description_type": triple.predicate,
            "description": triple.object,
            "title": triple.title,
            "kv_lists": [
                {
                    "key_string": record.key_text,
                    "value_string": record.value_text,
                    "kv_offset": record.offset,
                }
                for record in records
            ],
        }
