#!/usr/bin/env python3
"""Measure disk-space and warm retrieval breakdown across PathWeaver Stores."""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kblam.dag_store_retriever import DAGKVStoreRetriever, EntityHit, entity_embedding_model_path
from kblam.stores import GraphStore, KVStore
from kblam.stores.common import normalize_text
try:
    from tools.benchmark_pathweaver_retrieval import (
        load_rows,
        percentile_summary,
        value_matches_answer,
    )
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from benchmark_pathweaver_retrieval import (  # type: ignore[no-redef]
        load_rows,
        percentile_summary,
        value_matches_answer,
    )


@dataclass(frozen=True)
class StoreTarget:
    label: str
    path: Path


def parse_store(value: str) -> StoreTarget:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--store must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not label or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid Store target: {value}")
    return StoreTarget(label, path)


def parse_seed_strategies(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(item not in {"vector", "hybrid"} for item in values):
        raise argparse.ArgumentTypeError("seed strategies must be vector and/or hybrid")
    return values


def file_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def sqlite_breakdown(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT name, SUM(pgsize) AS bytes FROM dbstat GROUP BY name ORDER BY bytes DESC"
        ).fetchall()
        return {str(name): int(size) for name, size in rows}
    finally:
        connection.close()


def collect_space_breakdown(store_dir: Path) -> dict[str, Any]:
    graph_dir = store_dir / "graph"
    kv_dir = store_dir / "kv"
    files = {
        "kv_key_tensors": file_bytes(kv_dir / KVStore.KEY_ARRAY_NAME),
        "kv_value_tensors": file_bytes(kv_dir / KVStore.VALUE_ARRAY_NAME),
        "kv_sqlite": file_bytes(kv_dir / KVStore.DB_NAME),
        "entity_vectors": file_bytes(graph_dir / GraphStore.ENTITY_VECTORS_NAME),
        "entity_vector_ids": file_bytes(graph_dir / GraphStore.ENTITY_IDS_NAME),
        "entity_hnsw": file_bytes(graph_dir / GraphStore.HNSW_INDEX_NAME),
        "graph_sqlite": file_bytes(graph_dir / GraphStore.DB_NAME),
    }
    known_paths = {
        kv_dir / KVStore.KEY_ARRAY_NAME,
        kv_dir / KVStore.VALUE_ARRAY_NAME,
        kv_dir / KVStore.DB_NAME,
        graph_dir / GraphStore.ENTITY_VECTORS_NAME,
        graph_dir / GraphStore.ENTITY_IDS_NAME,
        graph_dir / GraphStore.HNSW_INDEX_NAME,
        graph_dir / GraphStore.DB_NAME,
    }
    metadata_and_other = sum(
        path.stat().st_size
        for component in (graph_dir, kv_dir)
        for path in component.iterdir()
        if path.is_file() and path not in known_paths
    )
    files["metadata_and_other"] = metadata_and_other

    with KVStore(kv_dir, create=False) as kv_store, GraphStore(
        graph_dir, create=False
    ) as graph_store:
        tensor_rows = kv_store.tensor_count
        key_shape: list[int] = []
        value_shape: list[int] = []
        key_dtype = value_dtype = ""
        if tensor_rows:
            keys, values = kv_store._load_tensor_pair()
            key_shape = list(keys.shape)
            value_shape = list(values.shape)
            key_dtype = str(keys.dtype)
            value_dtype = str(values.dtype)
        counts = {"kv_records": len(kv_store), "tensor_rows": tensor_rows, **graph_store.stats()}

    total = sum(files.values())
    return {
        "total_bytes": total,
        "files": files,
        "counts": counts,
        "tensors": {
            "key_shape": key_shape,
            "value_shape": value_shape,
            "key_dtype": key_dtype,
            "value_dtype": value_dtype,
        },
        "sqlite_objects": {
            "kv": sqlite_breakdown(kv_dir / KVStore.DB_NAME),
            "graph": sqlite_breakdown(graph_dir / GraphStore.DB_NAME),
        },
        "bytes_per_unique_kv": total / max(1, counts["kv_records"]),
        "append_peak_extra_bytes_estimate": max(
            files["kv_key_tensors"], files["kv_value_tensors"]
        ),
    }


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def encode_queries(embedder: Any, questions: list[str], batch_size: int) -> tuple[np.ndarray, dict[str, float]]:
    synchronize_cuda()
    started = time.perf_counter()
    vectors = np.asarray(
        embedder.encode(
            questions,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    synchronize_cuda()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return vectors, {
        "total_ms": elapsed_ms,
        "amortized_ms_per_query": elapsed_ms / max(1, len(questions)),
    }


def time_one_query(
    retriever: DAGKVStoreRetriever,
    query: str,
    vector: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
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
            [hit.node_id],
            hops=retriever.subgraph_hops,
            max_triples=retriever.max_triples_per_seed,
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
        records = retriever.kv_store.get_many(triple.kv_offsets)
        for record in records:
            kv_pairs.add((record.key_text, record.value_text))
            if record.offset not in seen_offsets:
                seen_offsets.add(record.offset)
                offsets.append(record.offset)
    kv_text_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    tensor_bytes = 0
    if offsets:
        keys, values = retriever.kv_store.get_tensors(offsets)
        tensor_bytes = int(keys.nbytes + values.nbytes)
    kv_tensor_ms = (time.perf_counter() - started) * 1000.0

    total_ms = (time.perf_counter() - total_started) * 1000.0
    return (
        {
            "ann": ann_ms,
            "mention": mention_ms,
            "graph": graph_ms,
            "kv_text": kv_text_ms,
            "kv_tensor": kv_tensor_ms,
            "total": total_ms,
        },
        {
            "nodes": len(node_ids),
            "triples": len(ordered_triples),
            "kvs": len(kv_pairs),
            "tensor_bytes": tensor_bytes,
            "kv_pairs": kv_pairs,
        },
    )


def benchmark_store(
    target: StoreTarget,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    embedder: Any,
    *,
    seed_strategy: str,
    entity_top_k: int,
    subgraph_hops: int,
    max_triples_per_seed: int | None,
    warmup_queries: int,
    repeats: int,
) -> dict[str, Any]:
    with DAGKVStoreRetriever(
        target.path,
        embedder,
        entity_top_k=entity_top_k,
        subgraph_hops=subgraph_hops,
        max_triples_per_seed=max_triples_per_seed,
        search_backend="hnsw",
        seed_strategy=seed_strategy,
    ) as retriever:
        cold_timings, _ = time_one_query(
            retriever, str(rows[0].get("question", "")), embeddings[0]
        )
        for row, vector in zip(rows[:warmup_queries], embeddings[:warmup_queries]):
            time_one_query(retriever, str(row.get("question", "")), vector)

        timings = {name: [] for name in ("ann", "mention", "graph", "kv_text", "kv_tensor", "total")}
        counts = {name: [] for name in ("nodes", "triples", "kvs", "tensor_bytes")}
        answer_hits = 0
        for _ in range(repeats):
            for row, vector in zip(rows, embeddings):
                query_timings, query_counts = time_one_query(
                    retriever, str(row.get("question", "")), vector
                )
                for name, value in query_timings.items():
                    timings[name].append(value)
                for name in counts:
                    counts[name].append(query_counts[name])
                answer = str(row.get("answer", ""))
                answer_hits += any(
                    value_matches_answer(value, answer)
                    for _, value in query_counts["kv_pairs"]
                )

    return {
        "label": target.label,
        "store_dir": str(target.path.resolve()),
        "seed_strategy": seed_strategy,
        "query_samples": len(rows),
        "repeats": repeats,
        "observations": len(rows) * repeats,
        "answer_hits": answer_hits,
        "answer_recall": answer_hits / max(1, len(rows) * repeats),
        "process_cold_first_query_ms": cold_timings,
        "warm_latency_ms": {name: percentile_summary(values) for name, values in timings.items()},
        "candidate_counts": {name: percentile_summary(values) for name, values in counts.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", action="append", type=parse_store, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--st-model", default="")
    parser.add_argument("--seed-strategies", type=parse_seed_strategies, default=parse_seed_strategies("vector,hybrid"))
    parser.add_argument("--entity-top-k", type=int, default=1)
    parser.add_argument("--subgraph-hops", type=int, default=2)
    parser.add_argument("--max-triples-per-seed", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.queries)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No query rows selected")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    model_path = args.st_model or entity_embedding_model_path(args.store[0].path)
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_path)
    embeddings, query_embedding = encode_queries(
        embedder,
        [str(row.get("question", "")) for row in rows],
        args.query_batch_size,
    )

    reports = []
    for target in args.store:
        space = collect_space_breakdown(target.path)
        retrieval = []
        for seed_strategy in args.seed_strategies:
            print(f"[BENCH] {target.label} seed={seed_strategy}", flush=True)
            retrieval.append(
                benchmark_store(
                    target,
                    rows,
                    embeddings,
                    embedder,
                    seed_strategy=seed_strategy,
                    entity_top_k=args.entity_top_k,
                    subgraph_hops=args.subgraph_hops,
                    max_triples_per_seed=args.max_triples_per_seed,
                    warmup_queries=min(args.warmup_queries, len(rows)),
                    repeats=args.repeats,
                )
            )
        reports.append({"label": target.label, "space": space, "retrieval": retrieval})

    output = {
        "python": platform.python_version(),
        "queries": str(args.queries.resolve()),
        "query_samples": len(rows),
        "embedding_model": str(Path(model_path).resolve()),
        "query_embedding": query_embedding,
        "configuration": {
            "entity_top_k": args.entity_top_k,
            "subgraph_hops": args.subgraph_hops,
            "max_triples_per_seed": args.max_triples_per_seed,
            "warmup_queries": min(args.warmup_queries, len(rows)),
            "repeats": args.repeats,
            "search_backend": "hnsw",
            "cache_semantics": "process-cold first query plus warmed in-process queries; OS page cache not dropped",
        },
        "stores": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] report={args.output}", flush=True)


if __name__ == "__main__":
    main()
