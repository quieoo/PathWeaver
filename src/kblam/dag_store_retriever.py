"""Online DAG retrieval backed by PathWeaver's persistent stores."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, Sequence

import numpy as np

from kblam.stores import GraphStore, GraphTriple, KVStore


class TextEmbedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EntityHit:
    node_id: int
    name: str
    score: float


@dataclass(frozen=True)
class CandidateGraph:
    """Union of the independently expanded subgraphs for one query."""

    entity_hits: tuple[EntityHit, ...]
    node_ids: tuple[int, ...]
    triples: tuple[GraphTriple, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "entity_hits": [
                {"node_id": hit.node_id, "name": hit.name, "score": hit.score}
                for hit in self.entity_hits
            ],
            "subgraph_nodes": len(self.node_ids),
            "candidate_triples": len(self.triples),
        }


@dataclass(frozen=True)
class DAGExtractionConfig:
    """Arguments consumed by the trainable v5.2 DAG inference functions."""

    infer_batch_size: int = 1024
    topic_top_k: int = 8
    dde_hops: int = 3
    mention_bonus: float = 0.2
    seed_edge_topk: int = 18
    expansion_hops: int = 2
    per_src_cap: int = 3
    max_nodes: int = 30
    max_edges: int = 40
    max_sinks: int = 3
    answer_aware: bool = False
    keep_score: bool = False
    reverse_sink_edge_topk: int = 2
    reverse_sink_hops: int = 4
    reverse_sink_beam_width: int = 4
    end_alpha: float = 0.60
    end_beta: float = 0.35
    end_gamma: float = 0.25
    supporting_only: bool = False

    def as_namespace(self) -> Namespace:
        return Namespace(
            **self.__dict__,
            limit=None,
            profile_online_latency=False,
        )


class DAGKVStoreRetriever:
    """Recover query-specific candidate graphs from GraphStore and KVStore."""

    def __init__(
        self,
        store_dir: str | Path,
        embedder: TextEmbedder,
        *,
        entity_top_k: int = 1,
        subgraph_hops: int = 2,
        max_triples_per_seed: int | None = None,
        search_backend: str = "auto",
        query_prompt_name: str | None = None,
    ) -> None:
        if entity_top_k <= 0:
            raise ValueError("entity_top_k must be positive")
        if subgraph_hops < 0:
            raise ValueError("subgraph_hops must be non-negative")
        if max_triples_per_seed is not None and max_triples_per_seed <= 0:
            raise ValueError("max_triples_per_seed must be positive when provided")

        root = Path(store_dir)
        self.graph_store = GraphStore(root / "graph", create=False)
        self.kv_store = KVStore(root / "kv", create=False)
        self.embedder = embedder
        self.entity_top_k = entity_top_k
        self.subgraph_hops = subgraph_hops
        self.max_triples_per_seed = max_triples_per_seed
        self.search_backend = search_backend
        self.query_prompt_name = query_prompt_name

    def __enter__(self) -> "DAGKVStoreRetriever":
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
        return self.retrieve_embeddings(vectors)

    def retrieve_embeddings(self, embeddings: np.ndarray) -> list[CandidateGraph]:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")
        return [self.retrieve_embedding(vector) for vector in vectors]

    def retrieve_embedding(self, embedding: np.ndarray) -> CandidateGraph:
        return self._retrieve_from_vector(np.asarray(embedding, dtype=np.float32))

    def build_candidate_sample(
        self,
        sample: dict[str, Any],
        candidate: CandidateGraph,
    ) -> dict[str, Any]:
        """Replace only the extractor's triple input; callers retain the original row."""
        prepared = dict(sample)
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
        if vectors.ndim != 2 or vectors.shape[0] != len(queries):
            raise ValueError("Embedder output must be a 2-D array aligned with queries")
        return vectors

    def _retrieve_from_vector(self, query_vector: np.ndarray) -> CandidateGraph:
        hits = tuple(
            EntityHit(node_id=node_id, name=self.graph_store.get_node_name(node_id), score=score)
            for node_id, score in self.graph_store.search_entities(
                query_vector,
                top_k=self.entity_top_k,
                backend=self.search_backend,
            )
        )
        node_ids: set[int] = set()
        triples: dict[int, GraphTriple] = {}

        # Expand each seed independently so one dense neighborhood cannot consume
        # another seed's per-subgraph triple budget.
        for hit in hits:
            local_nodes, local_triples = self.graph_store.get_local_subgraph(
                [hit.node_id],
                hops=self.subgraph_hops,
                max_triples=self.max_triples_per_seed,
            )
            node_ids.update(local_nodes)
            triples.update((triple.triple_id, triple) for triple in local_triples)

        return CandidateGraph(
            entity_hits=hits,
            node_ids=tuple(sorted(node_ids)),
            triples=tuple(triples[triple_id] for triple_id in sorted(triples)),
        )

    def _triple_to_raw(self, triple: GraphTriple) -> dict[str, Any]:
        records = self.kv_store.get_many(triple.kv_offsets)
        return {
            "type": triple.triple_type,
            "name": triple.subject,
            "description_type": triple.predicate,
            "description": triple.object,
            "title": triple.title,
            "kv_lists": [
                {"key_string": record.key_text, "value_string": record.value_text}
                for record in records
            ],
        }


class TrainableDAGExtractor:
    """Thin adapter around the existing v5.2 trainable DAG implementation."""

    def __init__(
        self,
        script_path: str | Path,
        model_checkpoint: str | Path,
        embedder: TextEmbedder,
        *,
        config: DAGExtractionConfig | None = None,
        cpu: bool = False,
    ) -> None:
        self.module = load_dag_module(script_path)
        self.embedder = embedder
        self.config = config or DAGExtractionConfig()
        self.edge_model, self.node_model, self.checkpoint, self.device = self.module.load_model(
            str(model_checkpoint), cpu=cpu
        )

    def extract(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output, _, _ = self.module.create_dag_with_model(
            self.config.as_namespace(),
            samples,
            self.embedder,
            self.edge_model,
            self.node_model,
            self.checkpoint,
            self.device,
        )
        return output


def load_dag_module(script_path: str | Path) -> ModuleType:
    """Load the research script without copying its evolving DAG algorithm."""
    path = Path(script_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = "_pathweaver_trainable_dag_v5_2"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DAG module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def entity_embedding_model_path(store_dir: str | Path) -> str:
    """Read the model used to build entity vectors, for query-side parity."""
    metadata_path = Path(store_dir) / "graph" / GraphStore.ENTITY_VECTOR_META_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Entity vector metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = metadata.get("model_path")
    if not model_path:
        raise ValueError(f"model_path is missing from {metadata_path}")
    return str(model_path)
