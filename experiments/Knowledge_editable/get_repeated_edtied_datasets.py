#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split the final multi-round dataset by per-sample total edit count.

Given a file prefix such as:
    /path/to/rounds_qwen35_27b/round

and --max-round 10, this script will load:
    /path/to/rounds_qwen35_27b/round0.jsonl
    ...
    /path/to/rounds_qwen35_27b/round10.jsonl

Assuming each round contains the same sample set in the same order, the script
counts for each sample how many times it was modified from round r-1 to round r.
The final round dataset is then split into separate jsonl files by total edit
count and written into a subdirectory under the input directory.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def round_file(prefix: str, round_id: int, suffix: str) -> Path:
    return Path(f"{prefix}{round_id}{suffix}")


def sample_identity(row: Dict[str, Any], index: int) -> str:
    value = row.get("_id")
    if isinstance(value, str) and value:
        return value
    meta = row.get("edit_meta", {})
    value = meta.get("source_sample_id")
    if isinstance(value, str) and value:
        return value
    return f"index:{index}"


def canonicalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    out.pop("edit_meta", None)
    return out


def row_was_edited(prev_row: Dict[str, Any], curr_row: Dict[str, Any]) -> bool:
    meta = curr_row.get("edit_meta", {})
    if isinstance(meta, dict) and "edited_in_round" in meta:
        return bool(meta.get("edited_in_round"))
    return canonicalize_row(prev_row) != canonicalize_row(curr_row)


def load_rounds(prefix: str, max_round: int, suffix: str) -> List[List[Dict[str, Any]]]:
    all_rounds: List[List[Dict[str, Any]]] = []
    for round_id in range(max_round + 1):
        path = round_file(prefix, round_id, suffix)
        if not path.exists():
            raise FileNotFoundError(f"Missing round file: {path}")
        all_rounds.append(read_jsonl(path))
    return all_rounds


def validate_rounds(all_rounds: Sequence[Sequence[Dict[str, Any]]]) -> None:
    if not all_rounds:
        raise ValueError("No round datasets loaded")
    expected_size = len(all_rounds[0])
    for round_id, rows in enumerate(all_rounds):
        if len(rows) != expected_size:
            raise ValueError(
                f"Round {round_id} has {len(rows)} rows, expected {expected_size}"
            )
    for idx in range(expected_size):
        base_id = sample_identity(all_rounds[0][idx], idx)
        for round_id in range(1, len(all_rounds)):
            current_id = sample_identity(all_rounds[round_id][idx], idx)
            if current_id != base_id:
                raise ValueError(
                    f"Sample identity mismatch at row {idx}: "
                    f"round0={base_id}, round{round_id}={current_id}"
                )


def build_split_datasets(
    all_rounds: Sequence[Sequence[Dict[str, Any]]],
    max_round: int,
) -> Dict[int, List[Dict[str, Any]]]:
    final_rows = all_rounds[max_round]
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for idx, final_row in enumerate(final_rows):
        edit_rounds: List[int] = []
        for round_id in range(1, max_round + 1):
            prev_row = all_rounds[round_id - 1][idx]
            curr_row = all_rounds[round_id][idx]
            if row_was_edited(prev_row, curr_row):
                edit_rounds.append(round_id)

        sample = copy.deepcopy(final_row)
        sample["edit_count_total"] = len(edit_rounds)
        sample["edit_rounds"] = edit_rounds
        buckets[len(edit_rounds)].append(sample)

    return dict(sorted(buckets.items()))


def merge_small_tail_buckets(
    buckets: Dict[int, List[Dict[str, Any]]],
    min_split: int,
) -> List[Tuple[List[int], List[Dict[str, Any]]]]:
    ordered = sorted(buckets.items())
    if min_split <= 0:
        return [([edit_count], rows) for edit_count, rows in ordered]

    merged: List[Tuple[List[int], List[Dict[str, Any]]]] = []
    for idx, (edit_count, rows) in enumerate(ordered):
        if len(rows) < min_split:
            tail_counts = [count for count, _ in ordered[idx:]]
            tail_rows: List[Dict[str, Any]] = []
            for _, tail_bucket_rows in ordered[idx:]:
                tail_rows.extend(tail_bucket_rows)
            merged.append((tail_counts, tail_rows))
            break
        merged.append(([edit_count], rows))
    return merged


def bucket_output_name(max_round: int, counts: Sequence[int]) -> str:
    if len(counts) == 1:
        return f"round{max_round}_edited_{counts[0]}_times.jsonl"
    return f"round{max_round}_edited_{counts[0]}_plus_times.jsonl"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Split the final round dataset by total per-sample edit count"
    )
    ap.add_argument(
        "--round-prefix",
        type=str,
        required=True,
        help="Common prefix before the round number, e.g. /path/to/rounds/round",
    )
    ap.add_argument(
        "--max-round",
        type=int,
        required=True,
        help="Maximum round id to analyze, e.g. 10 for round0..round10",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default=".jsonl",
        help="Round file suffix, default: .jsonl",
    )
    ap.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Optional output subdirectory name under the input directory",
    )
    ap.add_argument(
        "--min-split",
        type=int,
        default=0,
        help="If a bucket has fewer rows than this, merge it with all larger edit_count buckets",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.max_round < 1:
        raise ValueError("--max-round must be at least 1")

    prefix_path = Path(args.round_prefix)
    base_dir = prefix_path.parent
    output_subdir = (
        args.output_subdir
        if args.output_subdir
        else f"repeated_edit_splits_round{args.max_round}"
    )
    output_dir = base_dir / output_subdir

    all_rounds = load_rounds(args.round_prefix, args.max_round, args.suffix)
    validate_rounds(all_rounds)
    buckets = build_split_datasets(all_rounds, args.max_round)

    summary = {
        "round_prefix": args.round_prefix,
        "max_round": args.max_round,
        "suffix": args.suffix,
        "min_split": args.min_split,
        "num_samples": len(all_rounds[0]),
        "output_dir": str(output_dir),
        "bucket_sizes": {},
        "bucket_files": {},
        "bucket_ranges": {},
    }

    ensure_dir(output_dir)
    merged_buckets = merge_small_tail_buckets(buckets, args.min_split)
    for counts, rows in merged_buckets:
        out_path = output_dir / bucket_output_name(args.max_round, counts)
        key = str(counts[0]) if len(counts) == 1 else f"{counts[0]}+"
        write_jsonl(out_path, rows)
        summary["bucket_sizes"][key] = len(rows)
        summary["bucket_files"][key] = str(out_path)
        summary["bucket_ranges"][key] = list(counts)
        if len(counts) == 1:
            print(f"[saved] edit_count={counts[0]} rows={len(rows)} -> {out_path}")
        else:
            print(f"[saved] edit_count={counts[0]}+ rows={len(rows)} merged_from={list(counts)} -> {out_path}")

    summary_path = output_dir / f"round{args.max_round}_split_summary.json"
    write_json(summary_path, summary)
    print(f"[saved] summary -> {summary_path}")


if __name__ == "__main__":
    main()
