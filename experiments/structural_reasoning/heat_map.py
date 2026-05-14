#!/usr/bin/env python3
"""
Render per-sample heatmaps for path-attention traces.

Goal:
  - consume a saved trace file such as `2wiki_dag_kv.pt`
  - for each sample, draw two heatmaps:
      1. attention over KB tokens before propagation
      2. attention over KB tokens after propagation
  - x-axis: KB tokens
  - y-axis:
      - query tokens in `--y-axis-mode query`
      - layers in `--y-axis-mode layer`
  - heat value:
      - mean over heads at a selected layer for query mode
      - mean over heads and reduced over query positions for layer mode
  - also export every KV token's content for the sample

Default record selection:
  - group trace records by sample
  - pick the "middle" layer among available layer ids
  - within that layer, pick the record with the largest Q length
    because it is usually the prefill pass and keeps the full prompt tokens

Typical usage:
  python heat_map.py \
    --trace-path /mnt/n0/PathWeaver/experiments/path_attn/2wiki_dag_kv.pt \
    --dataset-path /path/to/dag_dataset.json \
    --tokenizer-path /path/to/qwen3_or_base_model \
    --model-format qwen3 \
    --out-dir /mnt/n0/PathWeaver/experiments/path_attn/2wiki_heatmaps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


QWEN3_SHORT_ANSWER_PROMPT = (
    "Answer with a short span from the context. "
    "Do not explain or output reasoning."
)


def _safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")
    return text or "sample"


def _safe_scalar(x: Any) -> Any:
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return {"shape": tuple(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_safe_scalar(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe_scalar(v) for k, v in x.items()}
    return str(x)


def _load_trace(trace_path: str) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    payload = torch.load(trace_path, map_location="cpu")
    if isinstance(payload, Mapping) and "records" in payload:
        records = payload["records"]
        meta = {str(k): _safe_scalar(v) for k, v in payload.items() if k != "records"}
        return records, meta
    if isinstance(payload, list):
        return payload, {}
    raise ValueError("trace file must be a dict with 'records' or a list of records")


def _context_value(context: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        if key in context and context[key] is not None:
            return context[key]
    return default


def _choose_attention_pair(record: Mapping[str, Any], prefer_normalized: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    if prefer_normalized:
        alpha = record.get("alpha_kb_norm")
        beta = record.get("beta_kb_norm")
        if torch.is_tensor(alpha) and torch.is_tensor(beta):
            return alpha.detach().cpu().float(), beta.detach().cpu().float(), "kb_normalized"

    alpha = record.get("alpha_kb")
    beta = record.get("beta_kb")
    if torch.is_tensor(alpha) and torch.is_tensor(beta):
        return alpha.detach().cpu().float(), beta.detach().cpu().float(), "raw"

    alpha = record.get("alpha_kb_norm")
    beta = record.get("beta_kb_norm")
    if torch.is_tensor(alpha) and torch.is_tensor(beta):
        return alpha.detach().cpu().float(), beta.detach().cpu().float(), "kb_normalized"

    return None, None, "missing"


def _mean_over_heads(attn: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
    if attn.ndim != 4:
        raise ValueError(f"expected [B,H,Q,K], got {tuple(attn.shape)}")
    if batch_idx >= attn.shape[0]:
        raise IndexError(f"batch_idx={batch_idx} out of range for B={attn.shape[0]}")
    return attn[batch_idx].mean(dim=0)


def _reduce_query_dim(x: torch.Tensor, reduce: str) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected [Q,K], got {tuple(x.shape)}")
    if reduce == "last":
        return x[-1]
    if reduce == "mean":
        return x.mean(dim=0)
    if reduce == "max":
        return x.max(dim=0).values
    if reduce == "sum":
        return x.sum(dim=0)
    raise ValueError(f"unknown query reduction: {reduce}")


def _pick_middle_layer(records: Sequence[Mapping[str, Any]]) -> int:
    layers = sorted(
        {
            int(r.get("layer_idx"))
            for r in records
            if r.get("layer_idx") is not None and str(r.get("layer_idx")) != ""
        }
    )
    if not layers:
        raise ValueError("no layer_idx found in sample records")
    return layers[len(layers) // 2]


def _pick_record_for_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    prefer_normalized: bool,
    layer_id: Optional[int],
) -> Tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, str]:
    usable: List[Tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, str]] = []
    for record in records:
        alpha, beta, kind = _choose_attention_pair(record, prefer_normalized=prefer_normalized)
        if alpha is None or beta is None:
            continue
        usable.append((record, alpha, beta, kind))

    if not usable:
        raise ValueError("no usable attention tensors found for sample")

    if layer_id is None:
        layer_id = _pick_middle_layer([u[0] for u in usable])

    same_layer = [u for u in usable if int(u[0].get("layer_idx")) == int(layer_id)]
    if not same_layer:
        raise ValueError(f"layer_id={layer_id} not found in sample records")

    # Prefer the longest Q to capture the prefill prompt rather than decode Q=1 records.
    same_layer.sort(key=lambda u: (u[1].shape[2], u[1].shape[0], u[1].shape[3]), reverse=True)
    return same_layer[0]


def _pick_best_record_per_layer(
    records: Sequence[Mapping[str, Any]],
    *,
    prefer_normalized: bool,
) -> List[Tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, str]]:
    by_layer: Dict[int, List[Tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, str]]] = defaultdict(list)
    for record in records:
        alpha, beta, kind = _choose_attention_pair(record, prefer_normalized=prefer_normalized)
        if alpha is None or beta is None:
            continue
        layer_idx = int(record.get("layer_idx"))
        by_layer[layer_idx].append((record, alpha, beta, kind))

    out = []
    for layer_idx in sorted(by_layer):
        candidates = by_layer[layer_idx]
        candidates.sort(key=lambda u: (u[1].shape[2], u[1].shape[0], u[1].shape[3]), reverse=True)
        out.append(candidates[0])
    return out


def _load_dataset(dataset_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not dataset_path:
        return None
    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("dataset json must be a list of rows")
        return data
    raise ValueError(f"unsupported dataset format: {dataset_path}")


def _find_dataset_row(
    dataset: Optional[Sequence[Mapping[str, Any]]],
    context: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    if dataset is None:
        return None

    source_index = _context_value(context, ["source_index"], default=None)
    if isinstance(source_index, int) and 0 <= source_index < len(dataset):
        return dataset[source_index]

    sample_id = str(_context_value(context, ["sample_id", "_id", "id"], default=""))
    if sample_id:
        for row in dataset:
            row_id = row.get("_id", row.get("id"))
            if row_id is not None and str(row_id) == sample_id:
                return row
    return None


def _extract_question(row: Optional[Mapping[str, Any]]) -> str:
    if row is None:
        return ""
    for key in ("Q", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_question_from_context(context: Mapping[str, Any]) -> str:
    value = context.get("question")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _format_prompt(question: str, model_format: str, tokenizer) -> str:
    if model_format == "qwen3":
        user_content = f"{question}\n\n{QWEN3_SHORT_ANSWER_PROMPT}"
        messages = [{"role": "user", "content": user_content}]
        if tokenizer is None:
            return user_content
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if model_format == "llama3":
        return (
            "<|start_header_id|>user<|end_header_id|> "
            + question
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>"
        )
    if model_format == "phi3":
        return "<|user|>\n" + question + "<|end|>\n" + "<|assistant|>\n"
    if model_format == "olmo3":
        if tokenizer is None:
            return question
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def _tokenize_query_tokens(
    question: str,
    *,
    tokenizer,
    model_format: str,
    expected_q: int,
) -> List[str]:
    if tokenizer is None or not question:
        return [f"q{i}" for i in range(expected_q)]

    prompt = _format_prompt(question, model_format, tokenizer)
    token_ids = tokenizer(prompt, add_special_tokens=False, return_tensors=None)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    labels = [_pretty_token(tok) for tok in tokens]

    if len(labels) == expected_q:
        return labels

    # Best-effort alignment: keep the tail if prompt is longer than Q.
    if len(labels) > expected_q:
        return labels[-expected_q:]

    labels = labels + [f"q{i}" for i in range(len(labels), expected_q)]
    return labels[:expected_q]


def _query_labels_from_context(context: Mapping[str, Any], expected_q: int) -> Optional[List[str]]:
    prompt_tokens = context.get("prompt_tokens")
    if not isinstance(prompt_tokens, list) or not prompt_tokens:
        return None
    labels = [_pretty_token(tok) for tok in prompt_tokens]
    if len(labels) == expected_q:
        return labels
    if len(labels) > expected_q:
        return labels[-expected_q:]
    labels = labels + [f"q{i}" for i in range(len(labels), expected_q)]
    return labels[:expected_q]


def _pretty_token(token: str) -> str:
    token = str(token)
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n")
    if token == "":
        return "<empty>"
    return token


def _kv_items_from_row(row: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if row is None:
        return []

    dag = row.get("dag")
    if isinstance(dag, Mapping):
        kv_nodes = dag.get("kv_nodes") or []
        items = []
        for idx, kv in enumerate(kv_nodes):
            if not isinstance(kv, Mapping):
                items.append({"kv_id": idx, "key": str(kv), "value": ""})
                continue
            items.append(
                {
                    "kv_id": idx,
                    "key": str(kv.get("key", "")),
                    "value": str(kv.get("value", "")),
                    "edge_score": kv.get("edge_score"),
                    "score": kv.get("score"),
                }
            )
        return items

    triples = row.get("triple_lists")
    if isinstance(triples, list):
        items = []
        for idx, tri in enumerate(triples):
            if not isinstance(tri, Mapping):
                items.append({"kv_id": idx, "key": str(tri), "value": ""})
                continue
            key = tri.get("key_string")
            if key is None:
                key = _join_nonempty(
                    tri.get("entity"),
                    tri.get("relation"),
                    tri.get("property"),
                    sep=" | ",
                )
            value = tri.get("value_string")
            if value is None:
                value = _join_nonempty(
                    tri.get("description"),
                    tri.get("tail"),
                    tri.get("value"),
                    sep=" | ",
                )
            items.append(
                {
                    "kv_id": idx,
                    "key": str(key or ""),
                    "value": str(value or ""),
                }
            )
        return items

    return []


def _kv_items_from_context(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    kv_items = context.get("kv_items")
    if not isinstance(kv_items, list):
        return []
    out = []
    for idx, item in enumerate(kv_items):
        if isinstance(item, Mapping):
            out.append(
                {
                    "kv_id": int(item.get("kv_id", idx)),
                    "key": str(item.get("key", "")),
                    "value": str(item.get("value", "")),
                    "edge_score": item.get("edge_score"),
                    "score": item.get("score"),
                }
            )
        else:
            out.append({"kv_id": idx, "key": str(item), "value": "", "edge_score": None, "score": None})
    return out


def _merge_kv_items(
    context_items: Sequence[Mapping[str, Any]],
    row_items: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not context_items:
        return [dict(item) for item in row_items]
    if not row_items:
        return [dict(item) for item in context_items]

    row_by_kv_id: Dict[int, Mapping[str, Any]] = {}
    for idx, item in enumerate(row_items):
        if not isinstance(item, Mapping):
            continue
        try:
            kv_id = int(item.get("kv_id", idx))
        except Exception:
            kv_id = idx
        row_by_kv_id[kv_id] = item

    merged: List[Dict[str, Any]] = []
    for idx, item in enumerate(context_items):
        merged_item = dict(item)
        try:
            kv_id = int(merged_item.get("kv_id", idx))
        except Exception:
            kv_id = idx
        row_item = row_by_kv_id.get(kv_id)
        if row_item is not None:
            if merged_item.get("edge_score") is None:
                merged_item["edge_score"] = row_item.get("edge_score")
            if merged_item.get("score") is None:
                merged_item["score"] = row_item.get("score")
            if not merged_item.get("key"):
                merged_item["key"] = row_item.get("key", "")
            if not merged_item.get("value"):
                merged_item["value"] = row_item.get("value", "")
        merged.append(merged_item)
    return merged


def _adjacency_from_row(
    row: Optional[Mapping[str, Any]],
    *,
    max_kv_len: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if row is None:
        return []

    dag = row.get("dag")
    if not isinstance(dag, Mapping):
        return []

    adj = dag.get("adj")
    if not isinstance(adj, list):
        return []

    limit = max_kv_len if max_kv_len is not None else len(adj)
    edges: List[Tuple[int, int]] = []
    for i, row_vals in enumerate(adj[:limit]):
        if not isinstance(row_vals, list):
            continue
        for j, val in enumerate(row_vals[:limit]):
            try:
                edge_val = float(val)
            except Exception:
                continue
            if edge_val > 0:
                edges.append((int(i), int(j)))
    return edges


def _join_nonempty(*parts: Any, sep: str = " ") -> str:
    return sep.join(str(p) for p in parts if p not in (None, ""))


def _write_kv_contents(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            line = {
                "kv_id": item.get("kv_id"),
                "key": item.get("key", ""),
                "value": item.get("value", ""),
                "edge_score": item.get("edge_score"),
                "score": item.get("score"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _save_matrix_csv(
    path: Path,
    matrix: torch.Tensor,
    *,
    kv_labels: Sequence[str],
    row_labels: Sequence[str],
    row_label_name: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{row_label_name}," + ",".join(_csv_escape(x) for x in kv_labels) + "\n")
        for q_idx in range(matrix.shape[0]):
            row = [row_labels[q_idx]]
            row.extend(f"{float(v):.8f}" for v in matrix[q_idx].tolist())
            f.write(",".join(_csv_escape(x) for x in row) + "\n")


def _csv_escape(x: Any) -> str:
    s = str(x)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _draw_heatmap_pair(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    row_labels: Sequence[str],
    kv_labels: Sequence[str],
    title_prefix: str,
    save_path: Path,
    cmap: str,
    y_axis_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(kv_labels) * 0.45), max(6, len(row_labels) * 0.28)), constrained_layout=True)

    vmax = float(max(before.max().item(), after.max().item(), 1e-12))
    titles = [f"{title_prefix} - Before", f"{title_prefix} - After"]
    matrices = [before, after]

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix.numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("KB tokens")
        ax.set_ylabel(y_axis_label)
        ax.set_xticks(range(len(kv_labels)))
        ax.set_xticklabels(kv_labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _draw_dag_graph(
    *,
    kv_items: Sequence[Mapping[str, Any]],
    edges: Sequence[Tuple[int, int]],
    save_path: Path,
    title: str,
) -> None:
    n = len(kv_items)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_title(title)
        ax.text(0.5, 0.5, "No KV nodes", ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return

    roots = sorted(set(range(n)) - {dst for _, dst in edges})
    if not roots:
        roots = [0]

    level = {node: 0 for node in roots}
    queue = list(roots)
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for src, dst in edges:
            if src != u:
                continue
            cand = level[u] + 1
            if dst not in level or cand > level[dst]:
                level[dst] = cand
                queue.append(dst)

    nodes_by_level: Dict[int, List[int]] = defaultdict(list)
    for node in range(n):
        nodes_by_level[level.get(node, 0)].append(node)

    positions: Dict[int, Tuple[float, float]] = {}
    max_width = max(len(nodes) for nodes in nodes_by_level.values())
    for lv in sorted(nodes_by_level):
        nodes = nodes_by_level[lv]
        width = len(nodes)
        offset = (max_width - width) / 2.0
        for idx, node in enumerate(nodes):
            positions[node] = (offset + idx, -lv)

    fig_w = max(8, max_width * 2.2)
    fig_h = max(4, (max(level.values(), default=0) + 1) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.set_title(title)

    for src, dst in edges:
        x1, y1 = positions.get(src, (0.0, 0.0))
        x2, y2 = positions.get(dst, (0.0, 0.0))
        ax.annotate(
            "",
            xy=(x2, y2 + 0.08),
            xytext=(x1, y1 - 0.08),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#666"},
        )

    for node in range(n):
        x, y = positions.get(node, (float(node), 0.0))
        item = kv_items[node] if node < len(kv_items) else {"key": "", "value": ""}
        key = re.sub(r"\s+", " ", str(item.get("key", "")).strip())
        value = re.sub(r"\s+", " ", str(item.get("value", "")).strip())
        edge_score = item.get("edge_score")
        key = key[:36] + ("…" if len(key) > 36 else "")
        value = value[:36] + ("…" if len(value) > 36 else "")
        if edge_score is None:
            score_text = "edge_score=None"
        else:
            try:
                score_text = f"edge_score={float(edge_score):.4g}"
            except Exception:
                score_text = f"edge_score={edge_score}"
        # label = f"KV{node}\n{key}\n{value}\n{score_text}"
        label = f"KV{node}\n{key}\n{value}"
        ax.scatter([x], [y], s=900, facecolors="#e8f1fb", edgecolors="#2b6cb0", linewidths=1.5, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=12, wrap=True, zorder=4)

    ax.axis("off")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _sample_title(sample_id: Any, layer_idx: int, attn_kind: str) -> str:
    return f"sample={sample_id} layer={layer_idx} ({attn_kind})"


def _group_records_by_sample(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        context = record.get("context") or {}
        if not isinstance(context, Mapping):
            context = {"context": str(context)}
        sample_id = str(_context_value(context, ["sample_id", "_id", "id", "qid"], default=""))
        if not sample_id:
            sample_id = f"record_{len(grouped)}"
        grouped[sample_id].append(record)
    return grouped


def render_heatmaps(
    *,
    trace_path: str,
    out_dir: str,
    dataset_path: Optional[str],
    tokenizer_path: Optional[str],
    model_format: str,
    prefer_normalized: bool,
    layer_id: Optional[int],
    y_axis_mode: str,
    query_reduce: str,
    max_samples: Optional[int],
    sample_ids: Optional[Sequence[str]],
    cmap: str,
) -> None:
    records, payload_meta = _load_trace(trace_path)
    dataset = _load_dataset(dataset_path)
    tokenizer = None
    if tokenizer_path:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not available but tokenizer_path was provided")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    grouped = _group_records_by_sample(records)
    selected_sample_ids = list(grouped.keys())
    if sample_ids:
        allowed = {str(x) for x in sample_ids}
        selected_sample_ids = [sid for sid in selected_sample_ids if sid in allowed]
    if max_samples is not None:
        selected_sample_ids = selected_sample_ids[:max_samples]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []

    for sample_id in selected_sample_ids:
        sample_records = grouped[sample_id]
        batch_idx = 0

        if y_axis_mode == "query":
            record, alpha, beta, attn_kind = _pick_record_for_sample(
                sample_records,
                prefer_normalized=prefer_normalized,
                layer_id=layer_id,
            )
            context = record.get("context") or {}
            if not isinstance(context, Mapping):
                context = {"context": str(context)}

            before = _mean_over_heads(alpha, batch_idx=batch_idx)
            after = _mean_over_heads(beta, batch_idx=batch_idx)
            q_len, k_len = before.shape

            row = _find_dataset_row(dataset, context)
            question = _extract_question_from_context(context) or _extract_question(row)
            row_labels = _query_labels_from_context(context, q_len)
            if row_labels is None:
                row_labels = _tokenize_query_tokens(
                    question,
                    tokenizer=tokenizer,
                    model_format=model_format,
                    expected_q=q_len,
                )
            row_labels = [f"Q{i}" for i in range(q_len)]
            y_axis_label = "Query tokens"
            selected_layer_idx = int(record.get("layer_idx"))
        else:
            layer_records = _pick_best_record_per_layer(
                sample_records,
                prefer_normalized=prefer_normalized,
            )
            if not layer_records:
                raise ValueError(f"no usable records found for sample={sample_id}")

            if layer_id is not None:
                layer_records = [x for x in layer_records if int(x[0].get("layer_idx")) == int(layer_id)]
                if not layer_records:
                    raise ValueError(f"layer_id={layer_id} not found for sample={sample_id}")

            context = layer_records[0][0].get("context") or {}
            if not isinstance(context, Mapping):
                context = {"context": str(context)}

            before_rows = []
            after_rows = []
            row_labels = []
            k_len = None
            for record, alpha, beta, attn_kind in layer_records:
                before_qk = _mean_over_heads(alpha, batch_idx=batch_idx)
                after_qk = _mean_over_heads(beta, batch_idx=batch_idx)
                before_k = _reduce_query_dim(before_qk, query_reduce)
                after_k = _reduce_query_dim(after_qk, query_reduce)
                if k_len is None:
                    k_len = before_k.shape[0]
                if before_k.shape[0] != k_len:
                    raise ValueError(
                        f"inconsistent kb_len across layers for sample={sample_id}: "
                        f"{k_len} vs {before_k.shape[0]}"
                    )
                before_rows.append(before_k)
                after_rows.append(after_k)
                row_labels.append(f"L{int(record.get('layer_idx'))}")

            before = torch.stack(before_rows, dim=0)
            after = torch.stack(after_rows, dim=0)
            q_len = before.shape[0]
            row = _find_dataset_row(dataset, context)
            question = _extract_question_from_context(context) or _extract_question(row)
            y_axis_label = f"Layers (query={query_reduce})"
            selected_layer_idx = -1

        kv_items_all = _merge_kv_items(
            _kv_items_from_context(context),
            _kv_items_from_row(row),
        )
        kv_items = kv_items_all[:k_len] if kv_items_all else []
        dag_edges = _adjacency_from_row(row, max_kv_len=k_len)
        kv_labels = [f"KV{i}" for i in range(k_len)]

        sample_dir = out_root / _safe_name(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        _draw_heatmap_pair(
            before,
            after,
            row_labels=row_labels,
            kv_labels=kv_labels,
            title_prefix=_sample_title(sample_id, selected_layer_idx, attn_kind),
            save_path=sample_dir / f"attention_before_after_{y_axis_mode}.png",
            cmap=cmap,
            y_axis_label=y_axis_label,
        )
        _save_matrix_csv(
            sample_dir / f"attention_before_{y_axis_mode}.csv",
            before,
            kv_labels=kv_labels,
            row_labels=row_labels,
            row_label_name="query_token" if y_axis_mode == "query" else "layer",
        )
        _save_matrix_csv(
            sample_dir / f"attention_after_{y_axis_mode}.csv",
            after,
            kv_labels=kv_labels,
            row_labels=row_labels,
            row_label_name="query_token" if y_axis_mode == "query" else "layer",
        )
        _write_kv_contents(sample_dir / "kv_contents.jsonl", kv_items)
        _draw_dag_graph(
            kv_items=kv_items,
            edges=dag_edges,
            save_path=sample_dir / "dag_graph.png",
            title=f"sample={sample_id} DAG",
        )

        metadata = {
            "sample_id": sample_id,
            "selected_layer_idx": selected_layer_idx,
            "attention_kind": attn_kind,
            "record_shape": tuple(alpha.shape) if y_axis_mode == "query" else [tuple(x[1].shape) for x in layer_records],
            "query_len": q_len,
            "kb_len": k_len,
            "y_axis_mode": y_axis_mode,
            "query_reduce": query_reduce if y_axis_mode == "layer" else None,
            "row_labels": row_labels,
            "kv_items": kv_items,
            "model_output_raw": context.get("model_output_raw", ""),
            "model_output_final": context.get("model_output_final", ""),
            "answer_final": context.get("answer_final", ""),
            "context": _safe_scalar(context),
            "payload_meta": payload_meta,
            "question": question,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        index_rows.append(
            {
                "sample_id": sample_id,
                "layer_idx": selected_layer_idx,
                "attention_kind": attn_kind,
                "query_len": q_len,
                "kb_len": k_len,
                "y_axis_mode": y_axis_mode,
                "query_reduce": query_reduce if y_axis_mode == "layer" else None,
                "sample_dir": str(sample_dir),
                "question": question,
            }
        )

    with open(out_root / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(index_rows)} sample heatmap folders to {out_root}")


def _compact_kv_label(idx: int, item: Mapping[str, Any], max_chars: int = 36) -> str:
    key = str(item.get("key", "")).strip()
    value = str(item.get("value", "")).strip()
    text = key if key else value
    if value and key:
        text = f"{key} -> {value}"
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return f"KV{idx}: {text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render before/after KB attention heatmaps from a path-attn trace file.")
    parser.add_argument("--trace-path", required=True, help="Path to trace .pt file, e.g. 2wiki_dag_kv.pt")
    parser.add_argument("--out-dir", required=True, help="Directory to save per-sample heatmaps and KV contents")
    parser.add_argument("--dataset-path", default=None, help="Optional dataset .json/.jsonl for recovering question text and KV contents")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer/model path for reconstructing query tokens")
    parser.add_argument(
        "--model-format",
        choices=["qwen3", "llama3", "phi3", "olmo3", "plain"],
        default="qwen3",
        help="Prompt format used during evaluation when reconstructing query tokens",
    )
    parser.add_argument(
        "--y-axis-mode",
        choices=["query", "layer"],
        default="query",
        help="Use query tokens or layers on the y-axis.",
    )
    parser.add_argument(
        "--query-reduce",
        choices=["last", "mean", "max", "sum"],
        default="mean",
        help="How to reduce query tokens when --y-axis-mode=layer.",
    )
    parser.add_argument("--layer-id", type=int, default=None, help="Specific layer to visualize. Default picks the middle available layer per sample.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples to render")
    parser.add_argument("--sample-id", action="append", default=None, help="Restrict to one or more sample ids")
    parser.add_argument("--raw", action="store_true", help="Use raw alpha_kb/beta_kb instead of KB-normalized tensors")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap for heatmaps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_heatmaps(
        trace_path=args.trace_path,
        out_dir=args.out_dir,
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        model_format=args.model_format,
        prefer_normalized=not args.raw,
        layer_id=args.layer_id,
        y_axis_mode=args.y_axis_mode,
        query_reduce=args.query_reduce,
        max_samples=args.max_samples,
        sample_ids=args.sample_id,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
