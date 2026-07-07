#!/usr/bin/env python3
"""Compare warm retrieval and space breakdown between PathWeaver Store V1 and V2."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kblam.dag_store_retriever import DAGKVStoreRetriever, EntityHit, entity_embedding_model_path
from kblam.dag_store_retriever_v2 import DAGKVStoreRetrieverV2
from kblam.stores import GraphStore, GraphStoreV2, KVStore, KVStoreV2
from kblam.stores.common import canonical_entity_key

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

try:
    from tools.benchmark_pathweaver_retrieval import load_rows, percentile_summary, value_matches_answer
except ModuleNotFoundError:
    from benchmark_pathweaver_retrieval import load_rows, percentile_summary, value_matches_answer  # type: ignore[no-redef]


@dataclass(frozen=True)
class QuerySample:
    question: str
    answers: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-v1", type=Path, required=True)
    parser.add_argument("--store-v2", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--st-model", type=str, default=None)
    parser.add_argument("--entity-top-k", type=int, default=1)
    parser.add_argument("--entity-candidate-top-k", type=int, default=64)
    parser.add_argument("--subgraph-hops", type=int, default=2)
    parser.add_argument("--seed-strategy", choices=["vector", "hybrid"], default="hybrid")
    parser.add_argument("--mention-min-chars", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--query-batch-size", type=int, default=100)
    parser.add_argument("--search-backend", choices=["auto", "exact", "hnsw"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def file_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def collect_v1_space(store_dir: Path) -> dict[str, Any]:
    graph_dir = store_dir / "graph"
    kv_dir = store_dir / "kv"
    files = {
        "kv_sqlite": file_bytes(kv_dir / KVStore.DB_NAME),
        "kv_key_tensors": file_bytes(kv_dir / KVStore.KEY_ARRAY_NAME),
        "kv_value_tensors": file_bytes(kv_dir / KVStore.VALUE_ARRAY_NAME),
        "graph_sqlite": file_bytes(graph_dir / GraphStore.DB_NAME),
        "entity_ids": file_bytes(graph_dir / GraphStore.ENTITY_IDS_NAME),
        "entity_vectors": file_bytes(graph_dir / GraphStore.ENTITY_VECTORS_NAME),
        "entity_hnsw": file_bytes(graph_dir / GraphStore.HNSW_INDEX_NAME),
    }
    with KVStore(kv_dir, create=False) as kv_store, GraphStore(graph_dir, create=False) as graph_store:
        counts = {"kv_records": len(kv_store), **graph_store.stats()}
    return {
        "total_bytes": sum(files.values()),
        "files": files,
        "counts": counts,
    }


def collect_v2_space(store_dir: Path) -> dict[str, Any]:
    graph_dir = store_dir / "graph_v2"
    kv_dir = store_dir / "kv_v2"
    files = {
        "kv_sqlite": file_bytes(kv_dir / KVStoreV2.DB_NAME),
        "kv_manifest": file_bytes(kv_dir / KVStoreV2.TENSOR_MANIFEST_NAME),
        "kv_segments_key": sum(file_bytes(path) for path in (kv_dir / "key").glob("*.npy")),
        "kv_segments_value": sum(file_bytes(path) for path in (kv_dir / "value").glob("*.npy")),
        "graph_manifest": file_bytes(graph_dir / GraphStoreV2.MANIFEST_NAME),
        "graph_arrays": sum(file_bytes(path) for path in graph_dir.glob("*.npy")),
        "graph_json": sum(file_bytes(path) for path in graph_dir.glob("*.json")) - file_bytes(graph_dir / GraphStoreV2.MANIFEST_NAME),
        "entity_hnsw": file_bytes(graph_dir / GraphStoreV2.HNSW_INDEX_NAME),
    }
    with KVStoreV2(kv_dir, create=False) as kv_store, GraphStoreV2(graph_dir, create=False) as graph_store:
        counts = {"kv_records": len(kv_store), **graph_store.stats()}
    return {
        "total_bytes": sum(files.values()),
        "files": files,
        "counts": counts,
    }


def load_queries(path: Path, limit: int) -> list[QuerySample]:
    samples = []
    for row in load_rows(path)[:limit]:
        answers = [str(item) for item in (row.get("answer") or []) if str(item).strip()]
        samples.append(QuerySample(question=str(row.get("question", "")), answers=answers))
    return samples


def encode_queries(model: Any, questions: list[str], batch_size: int) -> np.ndarray:
    vectors = np.asarray(
        model.encode(
            questions,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors


def evaluate_answer_recall(kv_pairs: set[tuple[str, str]], answers: list[str]) -> bool:
    for _, value in kv_pairs:
        if any(value_matches_answer(value, answer) for answer in answers):
            return True
    return False


def time_one_query_v1(retriever: DAGKVStoreRetriever, query: str, vector: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    total_started = time.perf_counter()
    started = time.perf_counter()
    vector_hits = [
        EntityHit(node_id, retriever.graph_store.get_node_name(node_id), score)
        for node_id, score in retriever.graph_store.search_entities(
            vector,
            top_k=retriever.entity_top_k,
            backend=retriever.search_backend,
        )
    ]
    ann_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    hits = retriever._merge_entity_hits(query, vector_hits)
    mention_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    node_ids: set[int] = set()
    triples: dict[int, Any] = {}
    for hit in hits:
        local_nodes, local_triples = retriever.graph_store.get_local_subgraph(
            [hit.node_id], hops=retriever.subgraph_hops, max_triples=retriever.max_triples_per_seed
        )
        node_ids.update(local_nodes)
        triples.update((triple.triple_id, triple) for triple in local_triples)
    ordered_triples = [triples[triple_id] for triple_id in sorted(triples)]
    graph_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    kv_pairs: set[tuple[str, str]] = set()
    offsets: list[int] = []
    seen_offsets: set[int] = set()
    for triple in ordered_triples:
        for record in retriever.kv_store.get_many(triple.kv_offsets):
            kv_pairs.add((record.key_text, record.value_text))
            if record.offset not in seen_offsets:
                seen_offsets.add(record.offset)
                offsets.append(record.offset)
    kv_text_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    if offsets:
        retriever.kv_store.get_tensors(offsets)
    kv_tensor_ms = (time.perf_counter() - started) * 1000.0
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return {
        "ann": ann_ms,
        "mention": mention_ms,
        "graph": graph_ms,
        "kv_text": kv_text_ms,
        "kv_tensor": kv_tensor_ms,
        "total": total_ms,
    }, {
        "triples": len(ordered_triples),
        "kvs": len(kv_pairs),
        "kv_pairs": kv_pairs,
    }


def time_one_query_v2(retriever: DAGKVStoreRetrieverV2, query: str, vector: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    total_started = time.perf_counter()
    started = time.perf_counter()
    vector_hits = retriever.graph_store.search_entities(
        vector,
        top_k=retriever.entity_candidate_top_k if retriever.seed_strategy == "hybrid" else retriever.entity_top_k,
        backend=retriever.search_backend,
    )
    ann_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    if retriever.seed_strategy == "hybrid":
        hits_raw = retriever.graph_store.shortlist_entity_hits(
            f" {canonical_entity_key(query)} ",
            final_top_k=retriever.entity_top_k,
            candidate_top_k=retriever.entity_candidate_top_k,
            min_mention_chars=retriever.mention_min_chars,
            backend=retriever.search_backend,
            candidates=vector_hits,
        )
    else:
        hits_raw = [(node_id, score, "vector") for node_id, score in vector_hits[: retriever.entity_top_k]]
    hits = [EntityHit(node_id, retriever.graph_store.get_node_name(node_id), score, source) for node_id, score, source in hits_raw]
    mention_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    node_ids: set[int] = set()
    triples: dict[int, Any] = {}
    for hit in hits:
        local_nodes, local_triples = retriever.graph_store.get_local_subgraph(
            [hit.node_id], hops=retriever.subgraph_hops, max_triples=retriever.max_triples_per_seed
        )
        node_ids.update(local_nodes)
        triples.update((triple.triple_id, triple) for triple in local_triples)
    ordered_triples = [triples[triple_id] for triple_id in sorted(triples)]
    graph_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    kv_pairs: set[tuple[str, str]] = set()
    offsets: list[int] = []
    seen_offsets: set[int] = set()
    for triple in ordered_triples:
        for record in retriever.kv_store.get_many(triple.kv_offsets):
            kv_pairs.add((record.key_text, record.value_text))
            if record.offset not in seen_offsets:
                seen_offsets.add(record.offset)
                offsets.append(record.offset)
    kv_text_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    if offsets:
        retriever.kv_store.get_tensors(offsets)
    kv_tensor_ms = (time.perf_counter() - started) * 1000.0
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return {
        "ann": ann_ms,
        "mention": mention_ms,
        "graph": graph_ms,
        "kv_text": kv_text_ms,
        "kv_tensor": kv_tensor_ms,
        "total": total_ms,
    }, {
        "triples": len(ordered_triples),
        "kvs": len(kv_pairs),
        "kv_pairs": kv_pairs,
    }


def summarize(observations: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ["ann", "mention", "graph", "kv_text", "kv_tensor", "total"]
    return {
        key: percentile_summary([row[key] for row in observations])
        for key in keys
    }


def main() -> None:
    args = parse_args()
    model_path = args.st_model or entity_embedding_model_path(args.store_v1)
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required for benchmark_pathweaver_store_v2.py")
    embedder = SentenceTransformer(model_path)
    samples = load_queries(args.queries, args.limit)
    questions = [sample.question for sample in samples]
    vectors = encode_queries(embedder, questions, args.query_batch_size)
    observations_v1: list[dict[str, float]] = []
    observations_v2: list[dict[str, float]] = []
    answer_hits_v1 = 0
    answer_hits_v2 = 0
    triples_v1: list[int] = []
    triples_v2: list[int] = []
    kvs_v1: list[int] = []
    kvs_v2: list[int] = []
    with DAGKVStoreRetriever(
        args.store_v1,
        embedder,
        entity_top_k=args.entity_top_k,
        subgraph_hops=args.subgraph_hops,
        search_backend=args.search_backend,
        seed_strategy=args.seed_strategy,
        mention_min_chars=args.mention_min_chars,
    ) as retriever_v1, DAGKVStoreRetrieverV2(
        args.store_v2,
        embedder,
        entity_top_k=args.entity_top_k,
        entity_candidate_top_k=args.entity_candidate_top_k,
        subgraph_hops=args.subgraph_hops,
        search_backend=args.search_backend,
        seed_strategy=args.seed_strategy,
        mention_min_chars=args.mention_min_chars,
    ) as retriever_v2:
        total = len(samples)
        warmup = min(args.warmup_queries, total)
        for _ in range(args.repeats):
            for index, sample in enumerate(samples):
                obs_v1, meta_v1 = time_one_query_v1(retriever_v1, sample.question, vectors[index])
                obs_v2, meta_v2 = time_one_query_v2(retriever_v2, sample.question, vectors[index])
                if index >= warmup:
                    observations_v1.append(obs_v1)
                    observations_v2.append(obs_v2)
                    triples_v1.append(meta_v1["triples"])
                    triples_v2.append(meta_v2["triples"])
                    kvs_v1.append(meta_v1["kvs"])
                    kvs_v2.append(meta_v2["kvs"])
                    answer_hits_v1 += int(evaluate_answer_recall(meta_v1["kv_pairs"], sample.answers))
                    answer_hits_v2 += int(evaluate_answer_recall(meta_v2["kv_pairs"], sample.answers))

    effective = max(1, (len(samples) - min(args.warmup_queries, len(samples))) * args.repeats)
    payload = {
        "config": {
            "store_v1": str(args.store_v1),
            "store_v2": str(args.store_v2),
            "queries": str(args.queries),
            "limit": args.limit,
            "warmup_queries": args.warmup_queries,
            "repeats": args.repeats,
            "entity_top_k": args.entity_top_k,
            "entity_candidate_top_k": args.entity_candidate_top_k,
            "subgraph_hops": args.subgraph_hops,
            "seed_strategy": args.seed_strategy,
            "search_backend": args.search_backend,
        },
        "space": {
            "v1": collect_v1_space(args.store_v1),
            "v2": collect_v2_space(args.store_v2),
        },
        "retrieval": {
            "v1": {
                "timings_ms": summarize(observations_v1),
                "candidate_triples_mean": float(np.mean(triples_v1)) if triples_v1 else 0.0,
                "candidate_kvs_mean": float(np.mean(kvs_v1)) if kvs_v1 else 0.0,
                "answer_recall": answer_hits_v1 / effective,
            },
            "v2": {
                "timings_ms": summarize(observations_v2),
                "candidate_triples_mean": float(np.mean(triples_v2)) if triples_v2 else 0.0,
                "candidate_kvs_mean": float(np.mean(kvs_v2)) if kvs_v2 else 0.0,
                "answer_recall": answer_hits_v2 / effective,
            },
        },
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
