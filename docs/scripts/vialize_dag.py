#!/usr/bin/env python3
"""Visualize DAG samples stored in dataset files.

The script reads a dataset file (JSON or JSONL), extracts the ``dag`` field
from one or more samples, and renders the KV-node DAG into SVG files.
It uses only the Python standard library so it can run in minimal
environments.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable, List, Sequence
from xml.sax.saxutils import escape


NODE_WIDTH = 320
NODE_HEIGHT = 120
H_GAP = 120
V_GAP = 48
MARGIN_X = 48
MARGIN_Y = 48
FONT_FAMILY = "Arial, Helvetica, sans-serif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read dataset file and visualize dag.kv_nodes + dag.adj to SVG."
    )
    parser.add_argument("dataset", help="Path to dataset file (.json or .jsonl).")
    parser.add_argument(
        "--mode",
        choices=["image", "text"],
        default="image",
        help="Rendering mode: save SVG locally or print text directly.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to render. Ignored when --all is provided.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every sample that contains a non-empty dag.",
    )
    parser.add_argument(
        "--isfilter",
        action="store_true",
        help="Enable filtering of samples.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG file or directory. Defaults to docs/scripts/dag_svgs/<dataset_stem>/",
    )
    parser.add_argument(
        "--title-field",
        default="question",
        help="Optional sample field used as graph title. Empty string disables it.",
    )
    parser.add_argument(
        "--max-label-width",
        type=int,
        default=36,
        help="Approximate text wrap width for node labels.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        samples: List[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object.")
                samples.append(obj)
        return samples

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            if not all(isinstance(item, dict) for item in obj):
                raise ValueError("JSON list must contain only objects.")
            return obj
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                if not all(isinstance(item, dict) for item in obj["data"]):
                    raise ValueError("JSON field 'data' must contain only objects.")
                return obj["data"]
            return [obj]
        raise ValueError("JSON file must be an object or a list of objects.")

    raise ValueError(f"Unsupported file type: {path.suffix}")


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._") or "dag"


def chunk_text(text: str, max_width: int) -> List[str]:
    text = " ".join(str(text or "").split())
    if not text:
        return []

    words = text.split(" ")
    lines: List[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word

    lines.append(current)
    return lines


def ensure_square_adj(adj: Sequence[Sequence[Any]], n: int) -> List[List[int]]:
    if len(adj) != n:
        raise ValueError(f"Adjacency row count {len(adj)} does not match kv_nodes count {n}.")

    normalized: List[List[int]] = []
    for row_idx, row in enumerate(adj):
        if len(row) != n:
            raise ValueError(f"Adjacency row {row_idx} length {len(row)} does not match {n}.")
        normalized.append([1 if int(value) else 0 for value in row])
    return normalized


def topo_layers(adj: Sequence[Sequence[int]]) -> List[int]:
    n = len(adj)
    indegree = [0] * n
    outgoing: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                outgoing[i].append(j)
                indegree[j] += 1

    q = deque(i for i, deg in enumerate(indegree) if deg == 0)
    visited = 0
    layer = [0] * n

    while q:
        node = q.popleft()
        visited += 1
        for nxt in outgoing[node]:
            layer[nxt] = max(layer[nxt], layer[node] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if visited != n:
        raise ValueError("Adjacency is not a DAG; cycle detected.")

    return layer


def build_node_label(node: dict[str, Any], max_width: int) -> List[str]:
    lines: List[str] = []

    key = node.get("key", "")
    if key:
        lines.extend(chunk_text(f"key: {key}", max_width))

    value = node.get("value", "")
    if value:
        lines.extend(chunk_text(f"value: {value}", max_width))

    score = node.get("score", None)
    if score is not None:
        lines.append(f"score: {score:.4f}" if isinstance(score, float) else f"score: {score}")

    return lines[:10]


def compute_positions(layers: Sequence[int]) -> tuple[dict[int, tuple[int, int]], int, int]:
    nodes_by_layer: dict[int, List[int]] = {}
    for idx, layer in enumerate(layers):
        nodes_by_layer.setdefault(layer, []).append(idx)

    max_layer = max(layers, default=0)
    max_nodes_in_layer = max((len(nodes) for nodes in nodes_by_layer.values()), default=1)
    width = MARGIN_X * 2 + (max_layer + 1) * NODE_WIDTH + max_layer * H_GAP
    height = MARGIN_Y * 2 + max_nodes_in_layer * NODE_HEIGHT + max(max_nodes_in_layer - 1, 0) * V_GAP

    positions: dict[int, tuple[int, int]] = {}
    for layer, node_indices in nodes_by_layer.items():
        layer_height = len(node_indices) * NODE_HEIGHT + max(len(node_indices) - 1, 0) * V_GAP
        offset_y = MARGIN_Y + max(0, (height - 2 * MARGIN_Y - layer_height) // 2)
        x = MARGIN_X + layer * (NODE_WIDTH + H_GAP)
        for rank, idx in enumerate(node_indices):
            y = offset_y + rank * (NODE_HEIGHT + V_GAP)
            positions[idx] = (x, y)

    return positions, width, height


def edge_svg(x1: int, y1: int, x2: int, y2: int) -> str:
    mid_x = (x1 + x2) / 2
    return (
        f'<path d="M {x1} {y1} C {mid_x:.1f} {y1}, {mid_x:.1f} {y2}, {x2} {y2}" '
        'stroke="#4b5563" stroke-width="2.2" fill="none" marker-end="url(#arrow)" />'
    )


def node_svg(node: dict[str, Any], idx: int, x: int, y: int, max_width: int) -> str:
    label_lines = build_node_label(node, max_width)
    text_parts: List[str] = [
        f'<rect x="{x}" y="{y}" width="{NODE_WIDTH}" height="{NODE_HEIGHT}" '
        'rx="16" ry="16" fill="#f8fafc" stroke="#0f172a" stroke-width="1.6" />',
        f'<text x="{x + 16}" y="{y + 24}" font-family="{FONT_FAMILY}" '
        'font-size="13" font-weight="700" fill="#0f172a">'
        f'{escape(f"#{idx}")}</text>',
    ]

    base_y = y + 46
    for line_idx, line in enumerate(label_lines):
        text_parts.append(
            f'<text x="{x + 16}" y="{base_y + line_idx * 15}" font-family="{FONT_FAMILY}" '
            f'font-size="12" fill="#1f2937">{escape(line)}</text>'
        )

    return "\n".join(text_parts)


def graph_title(sample: dict[str, Any], sample_idx: int, title_field: str) -> str:
    if title_field:
        value = sample.get(title_field)
        if isinstance(value, str) and value.strip():
            return f"Sample {sample_idx}: {value.strip()}"
    return f"Sample {sample_idx}"


def render_text(sample: dict[str, Any], sample_idx: int, title_field: str) -> str:
    dag = sample.get("dag")
    if not isinstance(dag, dict):
        raise ValueError(f"Sample {sample_idx} has no object-valued 'dag' field.")

    kv_nodes = dag.get("kv_nodes")
    adj = dag.get("adj")
    if not isinstance(kv_nodes, list) or not isinstance(adj, list):
        raise ValueError(f"Sample {sample_idx} dag must contain list fields 'kv_nodes' and 'adj'.")
    if not kv_nodes:
        raise ValueError(f"Sample {sample_idx} dag.kv_nodes is empty.")
    if not all(isinstance(node, dict) for node in kv_nodes):
        raise ValueError(f"Sample {sample_idx} dag.kv_nodes must contain objects.")
    if not all(isinstance(row, list) for row in adj):
        raise ValueError(f"Sample {sample_idx} dag.adj must be a 2D list.")

    adj_matrix = ensure_square_adj(adj, len(kv_nodes))
    title = graph_title(sample, sample_idx, title_field)
    lines: List[str] = [title, "=" * len(title), "Nodes:"]

    answer = sample.get("answer", sample.get("A"))
    if answer is not None:
        lines.insert(2, f"Answer: {' '.join(str(answer).split())}")

    for idx, node in enumerate(kv_nodes):
        key = " ".join(str(node.get("key", "")).split())
        value = " ".join(str(node.get("value", "")).split())
        score = node.get("score", None)
        lines.append(f"[{idx}] key={key}")
        lines.append(f"    value={value}")
        if score is not None:
            score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
            lines.append(f"    score={score_text}")

    lines.append("Edges:")
    edge_count = 0
    for i, row in enumerate(adj_matrix):
        for j, edge in enumerate(row):
            if edge:
                edge_count += 1
                lines.append(f"{i} -> {j}")

    if edge_count == 0:
        lines.append("(no edges)")

    return "\n".join(lines)


def render_svg(sample: dict[str, Any], sample_idx: int, max_label_width: int, title_field: str) -> str:
    dag = sample.get("dag")
    if not isinstance(dag, dict):
        raise ValueError(f"Sample {sample_idx} has no object-valued 'dag' field.")

    kv_nodes = dag.get("kv_nodes")
    adj = dag.get("adj")
    if not isinstance(kv_nodes, list) or not isinstance(adj, list):
        raise ValueError(f"Sample {sample_idx} dag must contain list fields 'kv_nodes' and 'adj'.")
    if not kv_nodes:
        raise ValueError(f"Sample {sample_idx} dag.kv_nodes is empty.")
    if not all(isinstance(node, dict) for node in kv_nodes):
        raise ValueError(f"Sample {sample_idx} dag.kv_nodes must contain objects.")
    if not all(isinstance(row, list) for row in adj):
        raise ValueError(f"Sample {sample_idx} dag.adj must be a 2D list.")

    adj_matrix = ensure_square_adj(adj, len(kv_nodes))
    layers = topo_layers(adj_matrix)
    positions, width, height = compute_positions(layers)

    title = graph_title(sample, sample_idx, title_field)
    title_lines = chunk_text(title, 80)
    title_block_height = 26 + max(0, len(title_lines) - 1) * 18
    canvas_height = height + title_block_height + 16

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{canvas_height}" '
        f'viewBox="0 0 {width} {canvas_height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5563" />',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{canvas_height}" fill="#ffffff" />',
    ]

    for line_idx, line in enumerate(title_lines):
        svg_parts.append(
            f'<text x="{MARGIN_X}" y="{32 + line_idx * 18}" font-family="{FONT_FAMILY}" '
            f'font-size="20" font-weight="700" fill="#111827">{escape(line)}</text>'
        )

    y_shift = title_block_height
    for i, row in enumerate(adj_matrix):
        x1, y1 = positions[i]
        sx = x1 + NODE_WIDTH
        sy = y1 + NODE_HEIGHT // 2 + y_shift
        for j, edge in enumerate(row):
            if not edge:
                continue
            x2, y2 = positions[j]
            tx = x2
            ty = y2 + NODE_HEIGHT // 2 + y_shift
            svg_parts.append(edge_svg(sx, sy, tx, ty))

    for idx, node in enumerate(kv_nodes):
        x, y = positions[idx]
        svg_parts.append(node_svg(node, idx, x, y + y_shift, max_label_width))

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def iter_targets(samples: Sequence[dict[str, Any]], render_all: bool, index: int) -> Iterable[tuple[int, dict[str, Any]]]:
    if render_all:
        for sample_idx, sample in enumerate(samples):
            dag = sample.get("dag")
            if isinstance(dag, dict) and dag.get("kv_nodes"):
                yield sample_idx, sample
        return

    if index < 0 or index >= len(samples):
        raise IndexError(f"Sample index {index} out of range 0..{len(samples) - 1}.")
    yield index, samples[index]


def resolve_output(args: argparse.Namespace, dataset_path: Path, render_all: bool) -> Path:
    if args.output:
        return Path(args.output)
    base_dir = Path("docs/scripts/dag_svgs") / sanitize_filename(dataset_path.stem)
    if render_all:
        return base_dir
    return base_dir / f"sample_{args.index}.svg"


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    filtered_id=[
        '5a85fb085542994775f606de'
    ]

    samples = load_dataset(dataset_path)
    targets = list(iter_targets(samples, args.isfilter or args.all, args.index))
    if not targets:
        raise ValueError("No samples with non-empty dag found.")

    if args.isfilter:
        targets = [(idx, sample) for idx, sample in targets if str(sample.get("_id")) in filtered_id]
        if not targets:
            raise ValueError("No samples matched the filter criteria.")
    
    

    if args.mode == "text":
        print(f"Loaded {len(samples)} samples from: {dataset_path}")
        for render_idx, (sample_idx, sample) in enumerate(targets):
            if render_idx > 0:
                print()
            print(render_text(sample=sample, sample_idx=sample_idx, title_field=args.title_field))
        print()
        print(f"Printed {len(targets)} graph(s).")
        return

    output_path = resolve_output(args, dataset_path, args.all)
    if args.all:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    written_files: List[Path] = []
    for sample_idx, sample in targets:
        svg = render_svg(
            sample=sample,
            sample_idx=sample_idx,
            max_label_width=args.max_label_width,
            title_field=args.title_field,
        )

        if args.all:
            out_file = output_path / f"sample_{sample_idx:05d}.svg"
        else:
            out_file = output_path

        out_file.write_text(svg, encoding="utf-8")
        written_files.append(out_file)

    print(f"Loaded {len(samples)} samples from: {dataset_path}")
    print(f"Rendered {len(written_files)} graph(s).")
    for path in written_files[:10]:
        print(path)
    if len(written_files) > 10:
        print(f"... and {len(written_files) - 10} more")


if __name__ == "__main__":
    main()
