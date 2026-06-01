#!/usr/bin/env python3
"""Create noisy DAG-KV datasets by reversing a ratio of DAG edges.

The script reads a SubgraphRAG output JSONL file, reverses a target ratio of
edges in each sample's ``dag.adj`` matrix, and guarantees the perturbed graph
remains acyclic. Output samples preserve the original dataset schema so the
result can be used directly by the embedding and inference pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

AdjMatrix = List[List[int]]
Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Reverse a ratio of edges in dag.adj while preserving DAG-ness."
    )
    ap.add_argument("--input", required=True, help="Input JSONL produced by SubgraphRAG.")
    ap.add_argument("--output", required=True, help="Output JSONL with noisy DAGs.")
    ap.add_argument(
        "--flip_ratio",
        type=float,
        required=True,
        help="Target ratio of original DAG edges to reverse, in [0, 1].",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. The per-sample seed is derived from this value.",
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar while processing the dataset.",
    )
    return ap.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_adj(adj: Any, n: int, sample_id: str) -> AdjMatrix:
    if not isinstance(adj, list) or len(adj) != n:
        raise ValueError(
            f"Sample {sample_id}: dag.adj must be a square {n}x{n} matrix, got {type(adj).__name__}."
        )
    out: AdjMatrix = []
    for row_idx, row in enumerate(adj):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(
                f"Sample {sample_id}: dag.adj row {row_idx} has invalid length; expected {n}."
            )
        clean_row = []
        for col_idx, val in enumerate(row):
            intval = int(val)
            if intval not in (0, 1):
                raise ValueError(
                    f"Sample {sample_id}: dag.adj[{row_idx}][{col_idx}] must be 0/1, got {val!r}."
                )
            clean_row.append(intval)
        out.append(clean_row)
    return out


def list_edges(adj: AdjMatrix) -> List[Edge]:
    edges: List[Edge] = []
    for src, row in enumerate(adj):
        for dst, val in enumerate(row):
            if val == 1:
                edges.append((src, dst))
    return edges


def is_acyclic(adj: AdjMatrix) -> bool:
    n = len(adj)
    indeg = [0] * n
    out_neighbors: List[List[int]] = [[] for _ in range(n)]
    for src, row in enumerate(adj):
        for dst, val in enumerate(row):
            if val:
                indeg[dst] += 1
                out_neighbors[src].append(dst)

    queue = [idx for idx, deg in enumerate(indeg) if deg == 0]
    visited = 0
    q_ptr = 0
    while q_ptr < len(queue):
        node = queue[q_ptr]
        q_ptr += 1
        visited += 1
        for nxt in out_neighbors[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return visited == n


def try_reverse_edge(adj: AdjMatrix, src: int, dst: int) -> bool:
    if src == dst or adj[src][dst] != 1 or adj[dst][src] != 0:
        return False

    adj[src][dst] = 0
    adj[dst][src] = 1
    if is_acyclic(adj):
        return True

    adj[dst][src] = 0
    adj[src][dst] = 1
    return False


def perturb_sample(
    sample: Dict[str, Any], flip_ratio: float, base_seed: int, sample_idx: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out_sample = copy.deepcopy(sample)
    dag = out_sample.get("dag")
    sample_id = str(out_sample.get("_id", sample_idx))

    if not isinstance(dag, dict):
        raise ValueError(f"Sample {sample_id}: missing dag field.")

    kv_nodes = dag.get("kv_nodes", [])
    if not isinstance(kv_nodes, list):
        raise ValueError(f"Sample {sample_id}: dag.kv_nodes must be a list.")

    n = len(kv_nodes)
    adj = validate_adj(dag.get("adj", []), n, sample_id)
    if not is_acyclic(adj):
        raise ValueError(f"Sample {sample_id}: input dag.adj is not acyclic.")

    original_edges = list_edges(adj)
    original_num_edges = len(original_edges)
    requested_num_flips = int(math.floor(original_num_edges * flip_ratio + 1e-12))

    rng = random.Random(base_seed + sample_idx)
    rng.shuffle(original_edges)

    flipped_edges: List[Dict[str, int]] = []
    for src, dst in original_edges:
        if len(flipped_edges) >= requested_num_flips:
            break
        if try_reverse_edge(adj, src, dst):
            flipped_edges.append(
                {
                    "from": src,
                    "to": dst,
                    "reversed_from": dst,
                    "reversed_to": src,
                }
            )

    actual_num_flips = len(flipped_edges)
    actual_flip_ratio = (
        float(actual_num_flips) / float(original_num_edges) if original_num_edges > 0 else 0.0
    )

    dag["adj"] = adj
    meta = dag.get("meta", {})
    if not isinstance(meta, dict):
        meta = {"original_meta": meta}
    meta.update(
        {
            "noise_type": "acyclic_edge_reversal",
            "requested_flip_ratio": float(flip_ratio),
            "actual_flip_ratio": actual_flip_ratio,
            "original_num_edges": int(original_num_edges),
            "requested_num_flips": int(requested_num_flips),
            "actual_num_flips": int(actual_num_flips),
            "noise_seed": int(base_seed + sample_idx),
            "flip_feasible": actual_num_flips == requested_num_flips,
            "flipped_edges": flipped_edges,
        }
    )
    dag["meta"] = meta
    out_sample["dag"] = dag
    stats = {
        "sample_id": sample_id,
        "original_num_edges": int(original_num_edges),
        "requested_num_flips": int(requested_num_flips),
        "actual_num_flips": int(actual_num_flips),
        "actual_flip_ratio": float(actual_flip_ratio),
        "flip_feasible": bool(actual_num_flips == requested_num_flips),
    }
    return out_sample, stats


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.flip_ratio <= 1.0:
        raise ValueError(f"--flip_ratio must be in [0, 1], got {args.flip_ratio}.")

    rows = read_jsonl(args.input)
    iterator: Iterable[Tuple[int, Dict[str, Any]]] = enumerate(rows)
    if args.progress:
        iterator = tqdm(iterator, total=len(rows), desc="Create noisy DAGs")

    out_rows: List[Dict[str, Any]] = []
    per_sample_stats: List[Dict[str, Any]] = []
    for sample_idx, sample in iterator:
        out_row, stats = perturb_sample(
                sample=sample,
                flip_ratio=args.flip_ratio,
                base_seed=args.seed,
                sample_idx=sample_idx,
            )
        out_rows.append(out_row)
        per_sample_stats.append(stats)

    write_jsonl(args.output, out_rows)

    total_edges = sum(item["original_num_edges"] for item in per_sample_stats)
    total_requested_flips = sum(item["requested_num_flips"] for item in per_sample_stats)
    total_actual_flips = sum(item["actual_num_flips"] for item in per_sample_stats)
    feasible_samples = sum(1 for item in per_sample_stats if item["flip_feasible"])
    actual_dataset_flip_ratio = (
        float(total_actual_flips) / float(total_edges) if total_edges > 0 else 0.0
    )
    requested_dataset_flip_ratio = (
        float(total_requested_flips) / float(total_edges) if total_edges > 0 else 0.0
    )
    mean_actual_flip_ratio = (
        mean(item["actual_flip_ratio"] for item in per_sample_stats) if per_sample_stats else 0.0
    )
    max_actual_flip_ratio = (
        max(item["actual_flip_ratio"] for item in per_sample_stats) if per_sample_stats else 0.0
    )
    min_actual_flip_ratio = (
        min(item["actual_flip_ratio"] for item in per_sample_stats) if per_sample_stats else 0.0
    )

    print(f"Output written to: {args.output}")
    print(
        "Dataset flip ratio: "
        f"requested={requested_dataset_flip_ratio:.6f} "
        f"actual={actual_dataset_flip_ratio:.6f}"
    )
    print(
        "Per-sample actual flip ratio: "
        f"mean={mean_actual_flip_ratio:.6f} "
        f"min={min_actual_flip_ratio:.6f} "
        f"max={max_actual_flip_ratio:.6f}"
    )
    print(
        "Feasible samples: "
        f"{feasible_samples}/{len(per_sample_stats)} "
        f"({(feasible_samples / len(per_sample_stats)) if per_sample_stats else 0.0:.6f})"
    )
    print(
        "Edge counts: "
        f"total_edges={total_edges} "
        f"requested_flips={total_requested_flips} "
        f"actual_flips={total_actual_flips}"
    )


if __name__ == "__main__":
    main()
