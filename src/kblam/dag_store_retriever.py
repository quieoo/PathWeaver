"""Online DAG retrieval backed by PathWeaver's persistent stores."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, Sequence

import numpy as np

from kblam.stores import GraphStore, GraphTriple, KVStore
from kblam.stores.common import canonical_entity_key


class TextEmbedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EntityHit:
    node_id: int
    name: str
    score: float
    source: str = "vector"


@dataclass(frozen=True)
class CandidateGraph:
    """Union of the independently expanded subgraphs for one query."""

    entity_hits: tuple[EntityHit, ...]
    node_ids: tuple[int, ...]
    triples: tuple[GraphTriple, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "entity_hits": [
                {
                    "node_id": hit.node_id,
                    "name": hit.name,
                    "score": hit.score,
                    "source": hit.source,
                }
                for hit in self.entity_hits
            ],
            "subgraph_nodes": len(self.node_ids),
            "candidate_triples": len(self.triples),
        }


@dataclass(frozen=True)
class DAGExtractionConfig:
    """Common arguments consumed by the compatible DAG inference backends."""

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
    embedding_batch_size: int | None = None
    feature_batch_size: int = 4096
    selection_mode: str = "legacy"
    terminal_reranker: str = "joint"
    terminal_end_weight: float = 0.35
    terminal_path_weight: float = 0.25
    terminal_value_weight: float = 0.20
    profile_online_latency: bool = False
    st_prompt_name: str | None = None

    def as_namespace(self) -> Namespace:
        return Namespace(
            **self.__dict__,
            limit=None,
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
        max_incident_triples_per_node: int | None = None,
        search_backend: str = "auto",
        query_prompt_name: str | None = None,
        seed_strategy: str = "vector",
        mention_min_chars: int = 8,
    ) -> None:
        if entity_top_k <= 0:
            raise ValueError("entity_top_k must be positive")
        if subgraph_hops < 0:
            raise ValueError("subgraph_hops must be non-negative")
        if max_triples_per_seed is not None and max_triples_per_seed <= 0:
            raise ValueError("max_triples_per_seed must be positive when provided")
        if max_incident_triples_per_node is not None and max_incident_triples_per_node <= 0:
            raise ValueError("max_incident_triples_per_node must be positive when provided")
        if seed_strategy not in {"vector", "hybrid"}:
            raise ValueError("seed_strategy must be 'vector' or 'hybrid'")
        if mention_min_chars <= 0:
            raise ValueError("mention_min_chars must be positive")

        root = Path(store_dir)
        self.graph_store = GraphStore(root / "graph", create=False)
        self.kv_store = KVStore(root / "kv", create=False)
        self.embedder = embedder
        self.entity_top_k = entity_top_k
        self.subgraph_hops = subgraph_hops
        self.max_triples_per_seed = max_triples_per_seed
        self.max_incident_triples_per_node = max_incident_triples_per_node
        self.search_backend = search_backend
        self.query_prompt_name = query_prompt_name
        self.seed_strategy = seed_strategy
        self.mention_min_chars = mention_min_chars
        self._entity_names = self.graph_store.entity_nodes() if seed_strategy == "hybrid" else []

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
        return [
            self._retrieve_from_vector(vector, query=str(query))
            for vector, query in zip(vectors, queries)
        ]

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
        # Some datasets retain triples inside context. The online extractor must
        # consume only the subgraph recovered from GraphStore, not those gold-row
        # leftovers. The original row is restored after extraction by the caller.
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
        if vectors.ndim != 2 or vectors.shape[0] != len(queries):
            raise ValueError("Embedder output must be a 2-D array aligned with queries")
        return vectors

    def _retrieve_from_vector(
        self,
        query_vector: np.ndarray,
        query: str | None = None,
    ) -> CandidateGraph:
        vector_hits = [
            EntityHit(node_id, self.graph_store.get_node_name(node_id), score)
            for node_id, score in self.graph_store.search_entities(
                query_vector, top_k=self.entity_top_k, backend=self.search_backend
            )
        ]
        hits = tuple(self._merge_entity_hits(query, vector_hits))
        node_ids: set[int] = set()
        triples: dict[int, GraphTriple] = {}

        # Expand each seed independently so one dense neighborhood cannot consume
        # another seed's per-subgraph triple budget.
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

    def _merge_entity_hits(
        self,
        query: str | None,
        vector_hits: list[EntityHit],
    ) -> list[EntityHit]:
        if self.seed_strategy == "vector" or not query:
            return vector_hits

        query_key = f" {canonical_entity_key(query)} "
        mentions = []
        for node_id, name in self._entity_names:
            name_key = canonical_entity_key(name)
            if len(name_key) < self.mention_min_chars or f" {name_key} " not in query_key:
                continue
            mentions.append((len(name_key), EntityHit(node_id, name, 1.0, "mention")))
        mentions.sort(key=lambda item: (-item[0], item[1].node_id))

        merged: list[EntityHit] = []
        seen: set[int] = set()
        for hit in [item[1] for item in mentions] + vector_hits:
            if hit.node_id in seen:
                continue
            seen.add(hit.node_id)
            merged.append(hit)
            if len(merged) == self.entity_top_k:
                break
        return merged

    def _triple_to_raw(self, triple: GraphTriple) -> dict[str, Any]:
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


class TrainableDAGExtractor:
    """Adapter for legacy v5.2 and standalone answer-blind v8 inference."""

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
        self.last_profile: dict[str, float] | None = None
        if hasattr(self.module, "load_models") and hasattr(self.module, "infer"):
            self.backend = "v8-answer-blind"
            loader = self.module.load_models
        elif hasattr(self.module, "load_model") and hasattr(self.module, "create_dag_with_model"):
            self.backend = "v5.2-legacy"
            loader = self.module.load_model
        else:
            raise TypeError(
                "Unsupported DAG script: expected load_models()+infer() or "
                "load_model()+create_dag_with_model()"
            )
        self.edge_model, self.node_model, self.checkpoint, self.device = loader(
            str(model_checkpoint), cpu=cpu
        )

    def extract(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        args = self.config.as_namespace()
        self.last_profile = None
        if self.backend == "v8-answer-blind":
            if hasattr(self.module, "infer_profiled"):
                output, profile = self.module.infer_profiled(
                    args,
                    samples,
                    self.embedder,
                    self.edge_model,
                    self.node_model,
                    self.checkpoint,
                    self.device,
                )
                self.last_profile = dict(profile)
                return output
            return self.module.infer(
                args,
                samples,
                self.embedder,
                self.edge_model,
                self.node_model,
                self.checkpoint,
                self.device,
            )
        output, _, _ = self.module.create_dag_with_model(
            args,
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
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    module_name = f"_pathweaver_dag_{path.stem}_{digest}"
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
