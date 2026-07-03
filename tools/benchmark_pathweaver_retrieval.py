#!/usr/bin/env python3
"""Sweep entity top-k and graph hops for store-backed candidate retrieval."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kblam.dag_store_retriever import DAGKVStoreRetriever, entity_embedding_model_path
from kblam.stores.common import normalize_text


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data")
    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list or an object containing a data list")
    return payload


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def value_matches_answer(value: str, answer: str) -> bool:
    value_key = normalize_text(value).casefold()
    answer_key = normalize_text(answer).casefold()
    return bool(value_key and answer_key) and (
        value_key == answer_key or answer_key in value_key or value_key in answer_key
    )


def evaluate_configuration(
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    retriever: DAGKVStoreRetriever,
    *,
    warmup_queries: int,
) -> dict[str, Any]:
    for vector in embeddings[:warmup_queries]:
        retriever.retrieve_embedding(vector)

    answer_hits = 0
    node_counts: list[int] = []
    triple_counts: list[int] = []
    kv_counts: list[int] = []
    latencies_ms: list[float] = []

    for row, vector in zip(rows, embeddings):
        started = time.perf_counter()
        candidate = retriever.retrieve_embedding(vector)
        prepared = retriever.build_candidate_sample(row, candidate)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        kv_pairs = {
            (str(kv.get("key_string", "")), str(kv.get("value_string", "")))
            for triple in prepared["triple_list"]
            for kv in triple.get("kv_lists") or []
        }
        answer = str(row.get("answer", ""))
        answer_hits += any(value_matches_answer(value, answer) for _, value in kv_pairs)
        node_counts.append(len(candidate.node_ids))
        triple_counts.append(len(candidate.triples))
        kv_counts.append(len(kv_pairs))

    return {
        "samples": len(rows),
        "answer_hits": answer_hits,
        "answer_recall": answer_hits / max(1, len(rows)),
        "candidate_nodes": percentile_summary(node_counts),
        "candidate_triples": percentile_summary(triple_counts),
        "candidate_kvs": percentile_summary(kv_counts),
        "retrieval_latency_ms": percentile_summary(latencies_ms),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--st-model", default="", help="Defaults to graph/entity_vectors.json:model_path")
    parser.add_argument("--query-prompt-name", default=None)
    parser.add_argument("--entity-top-k-values", type=parse_int_list, default=parse_int_list("1,2,4,8"))
    parser.add_argument("--subgraph-hop-values", type=parse_int_list, default=parse_int_list("1,2,3"))
    parser.add_argument("--search-backend", choices=["hnsw", "exact"], default="hnsw")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    rows = load_rows(args.input)
    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    from sentence_transformers import SentenceTransformer

    model_path = args.st_model or entity_embedding_model_path(args.store_dir)
    embedder = SentenceTransformer(model_path)
    encode_kwargs: dict[str, Any] = {
        "batch_size": args.query_batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": True,
    }
    if args.query_prompt_name:
        encode_kwargs["prompt_name"] = args.query_prompt_name

    embedding_started = time.perf_counter()
    embeddings = np.asarray(
        embedder.encode([str(row.get("question", "")) for row in rows], **encode_kwargs),
        dtype=np.float32,
    )
    embedding_elapsed_ms = (time.perf_counter() - embedding_started) * 1000.0

    results = []
    for entity_top_k in args.entity_top_k_values:
        for subgraph_hops in args.subgraph_hop_values:
            with DAGKVStoreRetriever(
                args.store_dir,
                embedder,
                entity_top_k=entity_top_k,
                subgraph_hops=subgraph_hops,
                search_backend=args.search_backend,
                query_prompt_name=args.query_prompt_name,
            ) as retriever:
                result = evaluate_configuration(
                    rows,
                    embeddings,
                    retriever,
                    warmup_queries=min(args.warmup_queries, len(rows)),
                )
            result.update({"entity_top_k": entity_top_k, "subgraph_hops": subgraph_hops})
            results.append(result)
            latency = result["retrieval_latency_ms"]
            print(
                f"top_k={entity_top_k} hops={subgraph_hops} "
                f"recall={result['answer_recall']:.4f} "
                f"triples_mean={result['candidate_triples']['mean']:.1f} "
                f"latency_ms={latency['mean']:.2f} p95={latency['p95']:.2f}",
                flush=True,
            )

    report = {
        "input": str(args.input.resolve()),
        "store_dir": str(args.store_dir.resolve()),
        "embedding_model": str(model_path),
        "search_backend": args.search_backend,
        "python": platform.python_version(),
        "samples": len(rows),
        "query_embedding": {
            "batch_size": args.query_batch_size,
            "total_ms": embedding_elapsed_ms,
            "amortized_ms_per_query": embedding_elapsed_ms / max(1, len(rows)),
            "excluded_from_retrieval_latency": True,
        },
        "warmup_queries_per_configuration": min(args.warmup_queries, len(rows)),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
