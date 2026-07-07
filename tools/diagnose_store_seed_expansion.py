#!/usr/bin/env python3
"""Diagnose candidate-graph expansion differences between two PathWeaver stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from kblam.dag_store_retriever_v2 import DAGKVStoreRetrieverV2

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

try:
    from tools.benchmark_pathweaver_retrieval import load_rows
except ModuleNotFoundError:
    from benchmark_pathweaver_retrieval import load_rows  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-a", type=Path, required=True)
    parser.add_argument("--store-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="store_a")
    parser.add_argument("--label-b", type=str, default="store_b")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--st-model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--entity-top-k", type=int, default=1)
    parser.add_argument("--entity-candidate-top-k", type=int, default=64)
    parser.add_argument("--subgraph-hops", type=int, default=2)
    parser.add_argument("--search-backend", choices=["auto", "exact", "hnsw"], default="hnsw")
    parser.add_argument("--seed-strategy", choices=["vector", "hybrid"], default="hybrid")
    parser.add_argument("--mention-min-chars", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def percentile_summary(values: list[int | float]) -> dict[str, int | float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "min": float(data.min()),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


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


def unique_kv_count(retriever: DAGKVStoreRetrieverV2, triples: list[Any]) -> int:
    offsets: set[int] = set()
    for triple in triples:
        offsets.update(int(offset) for offset in triple.kv_offsets)
    return len(offsets)


def per_query_stats(retriever: DAGKVStoreRetrieverV2, query: str, vector: np.ndarray) -> dict[str, Any]:
    candidate = retriever._retrieve_from_vector(np.asarray(vector, dtype=np.float32), query)
    top_hit = candidate.entity_hits[0] if candidate.entity_hits else None
    if top_hit is None:
        return {
            "seed_name": None,
            "seed_source": None,
            "seed_score": None,
            "hop1_triples": 0,
            "hop1_kvs": 0,
            "hop2_triples": 0,
            "hop2_kvs": 0,
            "candidate_triples": len(candidate.triples),
            "candidate_kvs": unique_kv_count(retriever, list(candidate.triples)),
        }

    _, hop1_triples = retriever.graph_store.get_local_subgraph([top_hit.node_id], hops=1, max_triples=None)
    _, hop2_triples = retriever.graph_store.get_local_subgraph([top_hit.node_id], hops=2, max_triples=None)
    return {
        "seed_name": top_hit.name,
        "seed_source": top_hit.source,
        "seed_score": float(top_hit.score),
        "hop1_triples": len(hop1_triples),
        "hop1_kvs": unique_kv_count(retriever, hop1_triples),
        "hop2_triples": len(hop2_triples),
        "hop2_kvs": unique_kv_count(retriever, hop2_triples),
        "candidate_triples": len(candidate.triples),
        "candidate_kvs": unique_kv_count(retriever, list(candidate.triples)),
    }


def summarize_store(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_sources: dict[str, int] = {}
    for row in rows:
        key = row["seed_source"] or "none"
        seed_sources[key] = seed_sources.get(key, 0) + 1
    return {
        "label": label,
        "queries": len(rows),
        "seed_source_breakdown": seed_sources,
        "hop1_triples": percentile_summary([int(row["hop1_triples"]) for row in rows]),
        "hop1_kvs": percentile_summary([int(row["hop1_kvs"]) for row in rows]),
        "hop2_triples": percentile_summary([int(row["hop2_triples"]) for row in rows]),
        "hop2_kvs": percentile_summary([int(row["hop2_kvs"]) for row in rows]),
        "candidate_triples": percentile_summary([int(row["candidate_triples"]) for row in rows]),
        "candidate_kvs": percentile_summary([int(row["candidate_kvs"]) for row in rows]),
    }


def comparison_rows(questions: list[str], rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for question, left, right in zip(questions, rows_a, rows_b):
        rows.append(
            {
                "question": question,
                "seed_name_a": left["seed_name"],
                "seed_name_b": right["seed_name"],
                "same_seed_name": left["seed_name"] == right["seed_name"],
                "candidate_triples_a": left["candidate_triples"],
                "candidate_triples_b": right["candidate_triples"],
                "candidate_triples_delta": int(right["candidate_triples"]) - int(left["candidate_triples"]),
                "hop2_triples_a": left["hop2_triples"],
                "hop2_triples_b": right["hop2_triples"],
                "hop2_triples_delta": int(right["hop2_triples"]) - int(left["hop2_triples"]),
                "hop2_kvs_a": left["hop2_kvs"],
                "hop2_kvs_b": right["hop2_kvs"],
                "hop2_kvs_delta": int(right["hop2_kvs"]) - int(left["hop2_kvs"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required for diagnose_store_seed_expansion.py")

    samples = load_rows(args.queries)[: args.limit]
    questions = [str(row.get("question", "")) for row in samples]
    embedder = SentenceTransformer(args.st_model)
    vectors = encode_queries(embedder, questions, args.query_batch_size)

    with DAGKVStoreRetrieverV2(
        args.store_a,
        embedder,
        entity_top_k=args.entity_top_k,
        entity_candidate_top_k=args.entity_candidate_top_k,
        subgraph_hops=args.subgraph_hops,
        search_backend=args.search_backend,
        seed_strategy=args.seed_strategy,
        mention_min_chars=args.mention_min_chars,
    ) as retriever_a, DAGKVStoreRetrieverV2(
        args.store_b,
        embedder,
        entity_top_k=args.entity_top_k,
        entity_candidate_top_k=args.entity_candidate_top_k,
        subgraph_hops=args.subgraph_hops,
        search_backend=args.search_backend,
        seed_strategy=args.seed_strategy,
        mention_min_chars=args.mention_min_chars,
    ) as retriever_b:
        rows_a = [per_query_stats(retriever_a, q, v) for q, v in zip(questions, vectors)]
        rows_b = [per_query_stats(retriever_b, q, v) for q, v in zip(questions, vectors)]

    paired = comparison_rows(questions, rows_a, rows_b)
    payload = {
        "config": {
            "store_a": str(args.store_a),
            "store_b": str(args.store_b),
            "label_a": args.label_a,
            "label_b": args.label_b,
            "queries": str(args.queries),
            "limit": args.limit,
            "entity_top_k": args.entity_top_k,
            "entity_candidate_top_k": args.entity_candidate_top_k,
            "subgraph_hops": args.subgraph_hops,
            "search_backend": args.search_backend,
            "seed_strategy": args.seed_strategy,
        },
        args.label_a: summarize_store(args.label_a, rows_a),
        args.label_b: summarize_store(args.label_b, rows_b),
        "comparison": {
            "same_seed_name_rate": float(np.mean([1.0 if row["same_seed_name"] else 0.0 for row in paired])) if paired else 0.0,
            "candidate_triples_delta": percentile_summary([int(row["candidate_triples_delta"]) for row in paired]),
            "hop2_triples_delta": percentile_summary([int(row["hop2_triples_delta"]) for row in paired]),
            "hop2_kvs_delta": percentile_summary([int(row["hop2_kvs_delta"]) for row in paired]),
            "top_candidate_triple_inflation": sorted(
                paired,
                key=lambda row: (-int(row["candidate_triples_delta"]), row["question"]),
            )[:10],
        },
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
