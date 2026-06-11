#!/usr/bin/env python3
"""Build cumulative KV-only DAG datasets for multi-round editable evaluation.

For each target dataset file, this script keeps the target file's sample order,
but replaces each sample's dag with a cumulative version built from the same
sample id across round0..round_r main-round DAG files, where r is the target
sample's edit_meta.round.

The cumulative dag:
- concatenates kv_nodes from every available historical round up to r
- builds a block-diagonal adjacency so each round's local graph stays isolated

This is intended for KV-only evaluation without --path_attn, where we want to
inject all historical KB tokens of a sample together.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_id_of(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("_id", row.get("id", fallback)))


def zero_adj(n: int) -> list[list[float]]:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def block_diag(parts: list[list[list[float]]]) -> list[list[float]]:
    total = sum(len(part) for part in parts)
    out = zero_adj(total)
    offset = 0
    for part in parts:
        n = len(part)
        for i in range(n):
            row = part[i]
            for j in range(min(n, len(row))):
                out[offset + i][offset + j] = float(row[j])
        offset += n
    return out


def normalize_adj(adj: Any, n: int) -> list[list[float]]:
    if not isinstance(adj, list) or len(adj) != n:
        return zero_adj(n)
    out: list[list[float]] = []
    for row in adj:
        if not isinstance(row, list) or len(row) != n:
            return zero_adj(n)
        out.append([float(x) for x in row])
    return out


def build_history_index(main_round_files: list[Path]) -> dict[int, dict[str, dict[str, Any]]]:
    history: dict[int, dict[str, dict[str, Any]]] = {}
    for round_id, path in enumerate(main_round_files):
        rows = load_jsonl(path)
        history[round_id] = {sample_id_of(row, idx): row for idx, row in enumerate(rows)}
    return history


def cumulative_dag_for_sample(
    sample_id: str,
    round_id: int,
    history_index: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    kv_nodes: list[dict[str, Any]] = []
    adj_parts: list[list[list[float]]] = []

    for rid in range(round_id + 1):
        row = history_index.get(rid, {}).get(sample_id)
        if row is None:
            continue
        dag = row.get("dag") or {}
        current_nodes = dag.get("kv_nodes") or []
        copied_nodes = [copy.deepcopy(node) for node in current_nodes if isinstance(node, dict)]
        n = len(copied_nodes)
        kv_nodes.extend(copied_nodes)
        adj_parts.append(normalize_adj(dag.get("adj") or [], n))

    meta = {
        "kv_only_cumulative": True,
        "history_rounds_included": [rid for rid in range(round_id + 1) if sample_id in history_index.get(rid, {})],
    }
    return {
        "kv_nodes": kv_nodes,
        "adj": block_diag(adj_parts),
        "meta": meta,
    }


def infer_round_id(row: dict[str, Any], default_round: int) -> int:
    edit_meta = row.get("edit_meta") or {}
    value = edit_meta.get("round", default_round)
    try:
        return int(value)
    except Exception:
        return default_round


def build_target_file(
    target_path: Path,
    history_index: dict[int, dict[str, dict[str, Any]]],
    output_suffix: str,
) -> Path:
    rows = load_jsonl(target_path)
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = sample_id_of(row, idx)
        round_id = infer_round_id(row, 0)
        updated = copy.deepcopy(row)
        updated["dag"] = cumulative_dag_for_sample(sample_id, round_id, history_index)
        out_rows.append(updated)

    output_path = target_path.with_name(f"{target_path.stem}{output_suffix}{target_path.suffix}")
    write_jsonl(output_path, out_rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cumulative KV-only DAG datasets")
    parser.add_argument(
        "--main-round-files",
        nargs="+",
        required=True,
        help="Ordered round0..roundN main DAG files used as cumulative history.",
    )
    parser.add_argument(
        "--target-files",
        nargs="+",
        required=True,
        help="Target DAG files to convert into cumulative KV-only DAG files.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_kv_only_cumulative",
        help="Suffix inserted before .jsonl for generated target files.",
    )
    args = parser.parse_args()

    main_round_files = [Path(p) for p in args.main_round_files]
    target_files = [Path(p) for p in args.target_files]

    history_index = build_history_index(main_round_files)
    for target_path in target_files:
        output_path = build_target_file(target_path, history_index, args.output_suffix)
        print(f"built {output_path}")


if __name__ == "__main__":
    main()
