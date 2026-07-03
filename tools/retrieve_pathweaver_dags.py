#!/usr/bin/env python3
"""Retrieve local store subgraphs and run the trainable DAG extractor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

from kblam.dag_store_retriever import (
    DAGExtractionConfig,
    DAGKVStoreRetriever,
    TrainableDAGExtractor,
    entity_embedding_model_path,
)


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


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if path.suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, ensure_ascii=False, indent=2)
        return
    raise ValueError(f"Unsupported output suffix: {path.suffix}")


def retrieve_rows(
    rows: list[dict[str, Any]],
    retriever: DAGKVStoreRetriever,
    extractor: TrainableDAGExtractor,
    *,
    answerable_only: bool = False,
    keep_retrieval_meta: bool = True,
) -> list[dict[str, Any]]:
    candidates = retriever.retrieve_many([str(row.get("question", "")) for row in rows])
    prepared = []
    retrieval_meta: dict[int, dict[str, Any]] = {}
    for index, (row, candidate) in enumerate(zip(rows, candidates)):
        candidate_row = retriever.build_candidate_sample(row, candidate)
        candidate_row["__pathweaver_input_index"] = index
        prepared.append(candidate_row)
        retrieval_meta[index] = candidate.metadata()

    extracted = extractor.extract(prepared)
    output = []
    for candidate_row in extracted:
        index = int(candidate_row.pop("__pathweaver_input_index"))
        dag = copy.deepcopy(candidate_row.get("dag") or {})
        if keep_retrieval_meta:
            dag.setdefault("meta", {})["retrieval"] = retrieval_meta[index]
        if answerable_only and not (dag.get("kv_nodes") or []):
            continue
        final_row = dict(rows[index])
        final_row["dag"] = dag
        output.append(final_row)
    return output


def compare_outputs(
    generated: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_by_id = {_sample_id(row, i): row for i, row in enumerate(generated)}
    reference_by_id = {_sample_id(row, i): row for i, row in enumerate(reference)}
    shared_ids = sorted(generated_by_id.keys() & reference_by_id.keys())

    exact = 0
    nonempty_generated = 0
    nonempty_reference = 0
    generated_answer_covered = 0
    reference_answer_covered = 0
    generated_kv_count = 0
    reference_kv_count = 0
    precision_sum = recall_sum = f1_sum = 0.0
    for sample_id in shared_ids:
        generated_row = generated_by_id[sample_id]
        reference_row = reference_by_id[sample_id]
        generated_dag = generated_row.get("dag") or {}
        reference_dag = reference_row.get("dag") or {}
        generated_kvs = _kv_set(generated_dag)
        reference_kvs = _kv_set(reference_dag)
        nonempty_generated += bool(generated_kvs)
        nonempty_reference += bool(reference_kvs)
        generated_kv_count += len(generated_kvs)
        reference_kv_count += len(reference_kvs)
        answer = str(generated_row.get("answer", reference_row.get("answer", "")))
        generated_answer_covered += _answer_covered(generated_kvs, answer)
        reference_answer_covered += _answer_covered(reference_kvs, answer)

        overlap = len(generated_kvs & reference_kvs)
        precision = overlap / len(generated_kvs) if generated_kvs else float(not reference_kvs)
        recall = overlap / len(reference_kvs) if reference_kvs else float(not generated_kvs)
        precision_sum += precision
        recall_sum += recall
        f1_sum += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact += _dag_without_retrieval_meta(generated_dag) == _dag_without_retrieval_meta(reference_dag)

    denominator = max(1, len(shared_ids))
    return {
        "generated_samples": len(generated),
        "reference_samples": len(reference),
        "shared_samples": len(shared_ids),
        "exact_dag_ratio": exact / denominator,
        "generated_nonempty_ratio": nonempty_generated / denominator,
        "reference_nonempty_ratio": nonempty_reference / denominator,
        "generated_answer_coverage": generated_answer_covered / denominator,
        "reference_answer_coverage": reference_answer_covered / denominator,
        "mean_generated_kv_nodes": generated_kv_count / denominator,
        "mean_reference_kv_nodes": reference_kv_count / denominator,
        "macro_kv_precision": precision_sum / denominator,
        "macro_kv_recall": recall_sum / denominator,
        "macro_kv_f1": f1_sum / denominator,
    }


def _sample_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("_id", row.get("id", f"__row_{index}")))


def _kv_set(dag: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(node.get("key", "")), str(node.get("value", "")))
        for node in dag.get("kv_nodes") or []
    }


def _answer_covered(kvs: set[tuple[str, str]], answer: str) -> bool:
    answer = answer.strip().casefold()
    if not answer:
        return False
    return any(
        answer == value.casefold()
        or answer in value.casefold()
        or value.casefold() in answer
        for _, value in kvs
        if value
    )


def _dag_without_retrieval_meta(dag: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dag)
    meta = normalized.get("meta")
    if isinstance(meta, dict):
        meta.pop("retrieval", None)
    return normalized


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--model-ckpt", required=True, type=Path)
    parser.add_argument(
        "--dag-script",
        type=Path,
        default=root / "docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py",
    )
    parser.add_argument("--st-model", default="", help="Defaults to graph/entity_vectors.json:model_path")
    parser.add_argument("--query-prompt-name", default=None)
    parser.add_argument("--entity-top-k", type=int, default=1)
    parser.add_argument("--subgraph-hops", type=int, default=2)
    parser.add_argument("--max-triples-per-seed", type=int, default=None)
    parser.add_argument("--search-backend", choices=["auto", "hnsw", "exact"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--answerable-only", action="store_true")
    parser.add_argument("--no-retrieval-meta", action="store_true")
    parser.add_argument("--reference-output", type=Path)

    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--topic-top-k", type=int, default=8)
    parser.add_argument("--dde-hops", type=int, default=3)
    parser.add_argument("--mention-bonus", type=float, default=0.2)
    parser.add_argument("--seed-edge-topk", type=int, default=18)
    parser.add_argument("--expansion-hops", type=int, default=2)
    parser.add_argument("--per-src-cap", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=30)
    parser.add_argument("--max-edges", type=int, default=40)
    parser.add_argument("--max-sinks", type=int, default=3)
    parser.add_argument("--answer-aware", action="store_true")
    parser.add_argument("--keep-score", action="store_true")
    parser.add_argument("--reverse-sink-edge-topk", type=int, default=2)
    parser.add_argument("--reverse-sink-hops", type=int, default=4)
    parser.add_argument("--reverse-sink-beam-width", type=int, default=4)
    parser.add_argument("--end-alpha", type=float, default=0.60)
    parser.add_argument("--end-beta", type=float, default=0.35)
    parser.add_argument("--end-gamma", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for label, path in (
        ("input", args.input),
        ("model checkpoint", args.model_ckpt),
        ("DAG script", args.dag_script),
        ("reference output", args.reference_output),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    rows = load_rows(args.input)
    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    from sentence_transformers import SentenceTransformer

    model_path = args.st_model or entity_embedding_model_path(args.store_dir)
    embedder = SentenceTransformer(model_path)
    config = DAGExtractionConfig(
        infer_batch_size=args.infer_batch_size,
        topic_top_k=args.topic_top_k,
        dde_hops=args.dde_hops,
        mention_bonus=args.mention_bonus,
        seed_edge_topk=args.seed_edge_topk,
        expansion_hops=args.expansion_hops,
        per_src_cap=args.per_src_cap,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_sinks=args.max_sinks,
        answer_aware=args.answer_aware,
        keep_score=args.keep_score,
        reverse_sink_edge_topk=args.reverse_sink_edge_topk,
        reverse_sink_hops=args.reverse_sink_hops,
        reverse_sink_beam_width=args.reverse_sink_beam_width,
        end_alpha=args.end_alpha,
        end_beta=args.end_beta,
        end_gamma=args.end_gamma,
    )
    extractor = TrainableDAGExtractor(
        args.dag_script,
        args.model_ckpt,
        embedder,
        config=config,
        cpu=args.cpu,
    )
    with DAGKVStoreRetriever(
        args.store_dir,
        embedder,
        entity_top_k=args.entity_top_k,
        subgraph_hops=args.subgraph_hops,
        max_triples_per_seed=args.max_triples_per_seed,
        search_backend=args.search_backend,
        query_prompt_name=args.query_prompt_name,
    ) as retriever:
        output = retrieve_rows(
            rows,
            retriever,
            extractor,
            answerable_only=args.answerable_only,
            keep_retrieval_meta=not args.no_retrieval_meta,
        )

    write_rows(args.output, output)
    print(f"[DONE] input={len(rows)} output={len(output)} saved_to={args.output}")
    if args.reference_output:
        report = compare_outputs(output, load_rows(args.reference_output))
        print(json.dumps({"comparison": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
