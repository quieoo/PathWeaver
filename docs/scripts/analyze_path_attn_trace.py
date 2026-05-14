#!/usr/bin/env python3
"""
Analyze DAG-KV path-attention trace files dumped by dump_path_attn_trace(...).

The trace is expected to be a torch.save file with a structure like:
{
    "records": [
        {
            "layer_idx": int,
            "kb_len": int,
            "shape": (B, H, Q, K),
            "path_attn_mix_ratio": float,
            "kb_layer_frequency": int,
            "adj_fingerprint": int | None,
            "adj_nnz": int | None,
            "context": {...},
            "alpha_kb": Tensor[B,H,Q,K],          # optional, before propagation
            "beta_kb": Tensor[B,H,Q,K],           # optional, after propagation
            "alpha_kb_norm": Tensor[B,H,Q,K],     # optional, KB-normalized before
            "beta_kb_norm": Tensor[B,H,Q,K],      # optional, KB-normalized after
        },
        ...
    ]
}

This script does NOT require answer-KV labels. It summarizes all KV tokens by:
  1. reducing attention over heads and query positions;
  2. reporting top-k KV ids before and after DAG propagation;
  3. reporting which KV ids are promoted most by propagation;
  4. writing readable CSV/JSON files for downstream analysis.

Example:
    python analyze_path_attn_trace.py trace.pt --out-dir trace_report --topk 20
    python analyze_path_attn_trace.py trace.pt --query-reduce last --topk 10 --print-samples 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


def _to_cpu_float(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    return x.detach().to(device="cpu", dtype=torch.float32)


def _safe_scalar(x: Any) -> Any:
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_safe_scalar(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe_scalar(v) for k, v in x.items()}
    return str(x)


def _json_dumps(obj: Any) -> str:
    return json.dumps(_safe_scalar(obj), ensure_ascii=False, sort_keys=True)


def _context_value(context: Mapping[str, Any], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        if k in context and context[k] is not None:
            return str(context[k])
    return default


def _reduce_attention(
    attn: torch.Tensor,
    *,
    query_reduce: str = "last",
    head_reduce: str = "mean",
) -> torch.Tensor:
    """
    Convert attention tensor [B,H,Q,K] to score tensor [B,K].
    """
    if attn.ndim != 4:
        raise ValueError(f"expected attention tensor with shape [B,H,Q,K], got {tuple(attn.shape)}")

    # Head reduction: [B,H,Q,K] -> [B,Q,K]
    if head_reduce == "mean":
        x = attn.mean(dim=1)
    elif head_reduce == "max":
        x = attn.max(dim=1).values
    else:
        raise ValueError(f"unknown head_reduce={head_reduce!r}")

    # Query-position reduction: [B,Q,K] -> [B,K]
    if query_reduce == "last":
        return x[:, -1, :]
    if query_reduce == "mean":
        return x.mean(dim=1)
    if query_reduce == "max":
        return x.max(dim=1).values
    if query_reduce == "sum":
        return x.sum(dim=1)
    raise ValueError(f"unknown query_reduce={query_reduce!r}")


def _topk(scores: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
    if scores.numel() == 0:
        return [], []
    kk = min(k, scores.numel())
    vals, idx = torch.topk(scores, kk)
    return idx.tolist(), vals.tolist()


def _rank_map(scores: torch.Tensor) -> torch.Tensor:
    """
    Return ranks where rank 1 is the largest score.
    """
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=order.dtype)
    return ranks


def _record_base_info(record: Mapping[str, Any], record_id: int) -> Dict[str, Any]:
    context = record.get("context") or {}
    if not isinstance(context, Mapping):
        context = {"context": str(context)}

    shape = record.get("shape", "")
    if torch.is_tensor(shape):
        shape = tuple(shape.tolist())

    return {
        "record_id": record_id,
        "dataset": _context_value(context, ["dataset", "dataset_name"]),
        "sample_id": _context_value(context, ["sample_id", "id", "qid", "_id"]),
        "turn_id": _context_value(context, ["turn_id", "step", "decode_step"]),
        "layer_idx": record.get("layer_idx", ""),
        "kb_len": record.get("kb_len", ""),
        "shape": str(tuple(shape)) if isinstance(shape, (list, tuple)) else str(shape),
        "path_attn_mix_ratio": record.get("path_attn_mix_ratio", ""),
        "kb_layer_frequency": record.get("kb_layer_frequency", ""),
        "adj_fingerprint": record.get("adj_fingerprint", ""),
        "adj_nnz": record.get("adj_nnz", ""),
        "context_json": _json_dumps(context),
    }


def _choose_attention_pair(record: Mapping[str, Any], prefer_normalized: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    """
    Pick before/after attention tensors.
    Prefer KB-normalized tensors because they isolate propagation inside KB tokens.
    """
    if prefer_normalized:
        alpha = _to_cpu_float(record.get("alpha_kb_norm"))
        beta = _to_cpu_float(record.get("beta_kb_norm"))
        if alpha is not None and beta is not None:
            return alpha, beta, "kb_normalized"

    alpha = _to_cpu_float(record.get("alpha_kb"))
    beta = _to_cpu_float(record.get("beta_kb"))
    if alpha is not None and beta is not None:
        return alpha, beta, "raw"

    # Fallback in case only one form exists.
    alpha = _to_cpu_float(record.get("alpha_kb_norm"))
    beta = _to_cpu_float(record.get("beta_kb_norm"))
    if alpha is not None and beta is not None:
        return alpha, beta, "kb_normalized"

    return None, None, "missing"


def analyze_trace(
    trace_path: str,
    *,
    out_dir: str,
    topk: int,
    query_reduce: str,
    head_reduce: str,
    prefer_normalized: bool,
    print_samples: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    payload = torch.load(trace_path, map_location="cpu")
    if isinstance(payload, Mapping) and "records" in payload:
        records = payload["records"]
        meta = {k: _safe_scalar(v) for k, v in payload.items() if k != "records"}
    elif isinstance(payload, list):
        records = payload
        meta = {}
    else:
        raise ValueError("trace file must be a dict containing 'records' or a list of records")

    if not isinstance(records, list):
        raise ValueError(f"records must be a list, got {type(records)}")

    summary_rows: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []
    promotion_rows: List[Dict[str, Any]] = []

    dataset_counter: Counter[str] = Counter()
    layer_counter: Counter[str] = Counter()
    missing_records = 0
    total_batches = 0

    for record_id, record in enumerate(records):
        if not isinstance(record, Mapping):
            missing_records += 1
            continue

        base = _record_base_info(record, record_id)
        dataset_counter[str(base["dataset"])] += 1
        layer_counter[str(base["layer_idx"])] += 1

        alpha, beta, attn_kind = _choose_attention_pair(record, prefer_normalized=prefer_normalized)
        if alpha is None or beta is None:
            missing_records += 1
            row = dict(base)
            row.update({"attention_kind": attn_kind, "status": "missing_attention"})
            summary_rows.append(row)
            continue

        alpha_scores = _reduce_attention(alpha, query_reduce=query_reduce, head_reduce=head_reduce)  # [B,K]
        beta_scores = _reduce_attention(beta, query_reduce=query_reduce, head_reduce=head_reduce)    # [B,K]

        if alpha_scores.shape != beta_scores.shape:
            raise ValueError(
                f"record {record_id}: before/after reduced shapes differ: "
                f"{tuple(alpha_scores.shape)} vs {tuple(beta_scores.shape)}"
            )

        B, K = alpha_scores.shape
        total_batches += B

        for b in range(B):
            before = alpha_scores[b]
            after = beta_scores[b]
            delta = after - before
            log_gain = torch.log((after + 1e-12) / (before + 1e-12))

            before_top_idx, before_top_val = _topk(before, topk)
            after_top_idx, after_top_val = _topk(after, topk)
            delta_top_idx, delta_top_val = _topk(delta, topk)
            log_gain_top_idx, log_gain_top_val = _topk(log_gain, topk)

            before_ranks = _rank_map(before)
            after_ranks = _rank_map(after)
            rank_delta = before_ranks.to(torch.int64) - after_ranks.to(torch.int64)  # positive means promoted
            rank_delta_top_idx, rank_delta_top_val = _topk(rank_delta.to(torch.float32), topk)

            row = dict(base)
            row.update(
                {
                    "batch_idx": b,
                    "attention_kind": attn_kind,
                    "status": "ok",
                    "B": B,
                    "K": K,
                    "query_reduce": query_reduce,
                    "head_reduce": head_reduce,
                    "before_sum": float(before.sum().item()),
                    "after_sum": float(after.sum().item()),
                    "before_max": float(before.max().item()) if K else math.nan,
                    "after_max": float(after.max().item()) if K else math.nan,
                    "before_entropy": float((-(before.clamp_min(1e-12) * before.clamp_min(1e-12).log()).sum()).item()),
                    "after_entropy": float((-(after.clamp_min(1e-12) * after.clamp_min(1e-12).log()).sum()).item()),
                    "top1_before_kv": before_top_idx[0] if before_top_idx else "",
                    "top1_before_score": before_top_val[0] if before_top_val else "",
                    "top1_after_kv": after_top_idx[0] if after_top_idx else "",
                    "top1_after_score": after_top_val[0] if after_top_val else "",
                    "top1_delta_kv": delta_top_idx[0] if delta_top_idx else "",
                    "top1_delta_score": delta_top_val[0] if delta_top_val else "",
                }
            )
            summary_rows.append(row)

            def add_top_rows(kind: str, indices: List[int], values: List[float]) -> None:
                for rank, (kv_id, score) in enumerate(zip(indices, values), start=1):
                    r = dict(base)
                    r.update(
                        {
                            "batch_idx": b,
                            "attention_kind": attn_kind,
                            "score_type": kind,
                            "rank": rank,
                            "kv_id": kv_id,
                            "score": score,
                            "query_reduce": query_reduce,
                            "head_reduce": head_reduce,
                        }
                    )
                    topk_rows.append(r)

            add_top_rows("before", before_top_idx, before_top_val)
            add_top_rows("after", after_top_idx, after_top_val)
            add_top_rows("delta_after_minus_before", delta_top_idx, delta_top_val)
            add_top_rows("log_gain_after_over_before", log_gain_top_idx, log_gain_top_val)

            for rank, (kv_id, gain) in enumerate(zip(rank_delta_top_idx, rank_delta_top_val), start=1):
                r = dict(base)
                r.update(
                    {
                        "batch_idx": b,
                        "attention_kind": attn_kind,
                        "rank": rank,
                        "kv_id": kv_id,
                        "rank_before": int(before_ranks[kv_id].item()),
                        "rank_after": int(after_ranks[kv_id].item()),
                        "rank_improvement": int(gain),
                        "score_before": float(before[kv_id].item()),
                        "score_after": float(after[kv_id].item()),
                        "score_delta": float(delta[kv_id].item()),
                        "log_gain": float(log_gain[kv_id].item()),
                        "query_reduce": query_reduce,
                        "head_reduce": head_reduce,
                    }
                )
                promotion_rows.append(r)

    summary_csv = os.path.join(out_dir, "record_summary.csv")
    topk_csv = os.path.join(out_dir, "topk_kv_scores.csv")
    promotion_csv = os.path.join(out_dir, "promoted_kv_by_rank.csv")
    report_json = os.path.join(out_dir, "report.json")

    _write_csv(summary_csv, summary_rows)
    _write_csv(topk_csv, topk_rows)
    _write_csv(promotion_csv, promotion_rows)

    report = {
        "trace_path": os.path.abspath(trace_path),
        "out_dir": os.path.abspath(out_dir),
        "num_records": len(records),
        "num_ok_summary_rows": sum(1 for r in summary_rows if r.get("status") == "ok"),
        "num_missing_records": missing_records,
        "num_batch_items": total_batches,
        "datasets": dict(dataset_counter),
        "layers": dict(sorted(layer_counter.items(), key=lambda kv: str(kv[0]))),
        "topk": topk,
        "query_reduce": query_reduce,
        "head_reduce": head_reduce,
        "prefer_normalized": prefer_normalized,
        "payload_meta": meta,
        "outputs": {
            "record_summary_csv": os.path.abspath(summary_csv),
            "topk_kv_scores_csv": os.path.abspath(topk_csv),
            "promoted_kv_by_rank_csv": os.path.abspath(promotion_csv),
        },
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_human_report(report, summary_rows, topk_rows, promotion_rows, print_samples=print_samples)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_human_report(
    report: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    topk_rows: Sequence[Mapping[str, Any]],
    promotion_rows: Sequence[Mapping[str, Any]],
    *,
    print_samples: int,
) -> None:
    print("\n=== DAG-KV Path Attention Trace Report ===")
    print(f"trace              : {report['trace_path']}")
    print(f"records            : {report['num_records']}")
    print(f"ok batch rows       : {report['num_ok_summary_rows']}")
    print(f"missing records     : {report['num_missing_records']}")
    print(f"query/head reduce   : {report['query_reduce']} / {report['head_reduce']}")
    print(f"top-k              : {report['topk']}")
    print(f"datasets           : {report['datasets']}")
    print(f"layers             : {report['layers']}")
    print("\nOutputs:")
    for name, path in report["outputs"].items():
        print(f"  - {name}: {path}")

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    if not ok_rows:
        print("\nNo readable attention records found.")
        return

    print("\nQuick examples:")
    for row in ok_rows[: max(0, print_samples)]:
        rid = row.get("record_id")
        ds = row.get("dataset") or "?"
        sid = row.get("sample_id") or "?"
        layer = row.get("layer_idx")
        b = row.get("batch_idx")
        print(
            f"  record={rid} dataset={ds} sample={sid} layer={layer} batch={b} "
            f"top1_before=KV{row.get('top1_before_kv')}({float(row.get('top1_before_score') or 0):.4g}) "
            f"top1_after=KV{row.get('top1_after_kv')}({float(row.get('top1_after_score') or 0):.4g}) "
            f"largest_delta=KV{row.get('top1_delta_kv')}({float(row.get('top1_delta_score') or 0):.4g})"
        )

    # Aggregate rough top promoted KV ids by dataset/layer, only based on top promotion rows.
    print("\nMost frequently promoted KV ids in top promotion rows:")
    bucket: Dict[Tuple[str, str], Counter[int]] = defaultdict(Counter)
    for r in promotion_rows:
        ds = str(r.get("dataset") or "?")
        layer = str(r.get("layer_idx") or "?")
        kv_id = r.get("kv_id")
        if isinstance(kv_id, int):
            bucket[(ds, layer)][kv_id] += 1
    for (ds, layer), counter in list(bucket.items())[:10]:
        print(f"  dataset={ds} layer={layer}: {counter.most_common(5)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and summarize DAG-KV path-attention trace files.")
    parser.add_argument("trace_path", help="Path to torch.save trace file produced by dump_path_attn_trace(...).")
    parser.add_argument("--out-dir", default="path_attn_report", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--topk", type=int, default=20, help="Number of KV ids to keep for top-k reports.")
    parser.add_argument(
        "--query-reduce",
        choices=["last", "mean", "max", "sum"],
        default="last",
        help="How to reduce the query-position dimension Q. 'last' is closest to first-answer prediction.",
    )
    parser.add_argument(
        "--head-reduce",
        choices=["mean", "max"],
        default="mean",
        help="How to reduce attention heads.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw alpha_kb/beta_kb instead of KB-normalized tensors when both are available.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=5,
        help="Number of quick examples to print to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_trace(
        args.trace_path,
        out_dir=args.out_dir,
        topk=args.topk,
        query_reduce=args.query_reduce,
        head_reduce=args.head_reduce,
        prefer_normalized=not args.raw,
        print_samples=args.print_samples,
    )


if __name__ == "__main__":
    main()
