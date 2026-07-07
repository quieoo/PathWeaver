#!/usr/bin/env python3
"""Report the distribution of per-entity KV counts for a PathWeaver store."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True, help="Path to store root")
    parser.add_argument("--store-version", choices=["auto", "v1", "v2"], default="auto")
    parser.add_argument(
        "--scope",
        choices=["incident", "subject"],
        default="incident",
        help="incident counts subject + reverse entity-linked triples; subject counts only outgoing triples",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def detect_store_version(store_dir: Path) -> str:
    if (store_dir / "graph_v2").is_dir():
        return "v2"
    if (store_dir / "graph").is_dir():
        return "v1"
    raise FileNotFoundError(f"Cannot detect store version under {store_dir}")


def summarize(values: list[int]) -> dict[str, int | float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "min": int(data.min()),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": int(data.max()),
    }


def load_v1_counts(store_dir: Path, scope: str) -> list[int]:
    db_path = store_dir / "graph" / "graph_store.sqlite3"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        entity_ids = [
            int(row["node_id"])
            for row in connection.execute("SELECT node_id FROM nodes WHERE kind = 'entity' ORDER BY node_id")
        ]
        kv_sets = {node_id: set() for node_id in entity_ids}

        query = """
            SELECT
                t.subject_id,
                t.object_id,
                t.object_kind,
                tk.kv_offset
            FROM triples t
            LEFT JOIN triple_kvs tk ON tk.triple_id = t.triple_id
        """
        for row in connection.execute(query):
            if row["kv_offset"] is None:
                continue
            offset = int(row["kv_offset"])
            subject_id = int(row["subject_id"])
            if subject_id in kv_sets:
                kv_sets[subject_id].add(offset)
            if scope == "incident" and str(row["object_kind"]) == "entity":
                object_id = int(row["object_id"])
                if object_id != subject_id and object_id in kv_sets:
                    kv_sets[object_id].add(offset)

        return [len(kv_sets[node_id]) for node_id in entity_ids]
    finally:
        connection.close()


def load_v2_counts(store_dir: Path, scope: str) -> list[int]:
    root = store_dir / "graph_v2"
    entity_node_ids = np.load(root / "entity_node_ids.npy", mmap_mode="r")
    triple_subject_pos = np.load(root / "triple_subject_pos.npy", mmap_mode="r")
    triple_object_node_id = np.load(root / "triple_object_node_id.npy", mmap_mode="r")
    triple_object_kind = np.load(root / "triple_object_kind.npy", mmap_mode="r")
    triple_kv_index = np.load(root / "triple_kv_index.npy", mmap_mode="r")
    triple_kv_offsets = np.load(root / "triple_kv_offsets.npy", mmap_mode="r")

    entity_ids = [int(node_id) for node_id in entity_node_ids.tolist()]
    kv_sets = {node_id: set() for node_id in entity_ids}

    triple_count = int(triple_subject_pos.shape[0])
    for triple_idx in range(triple_count):
        subject_id = int(entity_node_ids[int(triple_subject_pos[triple_idx])])
        start = int(triple_kv_index[triple_idx])
        end = int(triple_kv_index[triple_idx + 1])
        offsets = [int(offset) for offset in triple_kv_offsets[start:end].tolist()]

        if subject_id in kv_sets:
            kv_sets[subject_id].update(offsets)
        if scope == "incident" and str(triple_object_kind[triple_idx]) == "entity":
            object_id = int(triple_object_node_id[triple_idx])
            if object_id != subject_id and object_id in kv_sets:
                kv_sets[object_id].update(offsets)

    return [len(kv_sets[node_id]) for node_id in entity_ids]


def main() -> None:
    args = parse_args()
    version = detect_store_version(args.store_dir) if args.store_version == "auto" else args.store_version
    counts = load_v1_counts(args.store_dir, args.scope) if version == "v1" else load_v2_counts(args.store_dir, args.scope)

    payload = {
        "config": {
            "store_dir": str(args.store_dir),
            "store_version": version,
            "scope": args.scope,
        },
        "kv_per_entity": summarize(counts),
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
