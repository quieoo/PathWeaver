#!/usr/bin/env python3
"""Inspect a PathWeaver KVStore/GraphStore and emit a JSON quality report."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


NodeMap = dict[int, tuple[str, str]]
TripleRow = dict[str, Any]


@dataclass
class GraphSnapshot:
    """In-memory graph indexes reused by all report sections."""

    nodes: NodeMap
    triples: list[TripleRow]
    entity_neighbors: dict[int, set[int]]
    entity_incident_triples: dict[int, set[int]]
    triple_offsets: dict[int, set[int]]

    @property
    def entity_ids(self) -> set[int]:
        return {node_id for node_id, (_, kind) in self.nodes.items() if kind == "entity"}


def _readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _percentile_summary(values: Iterable[int | float]) -> dict[str, int | float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "min": int(data.min()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": int(data.max()),
    }


class _DisjointSet:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in self.parent}

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def component_sizes(self) -> list[int]:
        counts = Counter(self.find(item) for item in self.parent)
        return sorted(counts.values(), reverse=True)


def _load_graph(connection: sqlite3.Connection) -> GraphSnapshot:
    nodes = {
        int(row["node_id"]): (str(row["canonical_name"]), str(row["kind"]))
        for row in connection.execute("SELECT node_id, canonical_name, kind FROM nodes")
    }
    triples: list[dict[str, Any]] = []
    entity_neighbors: dict[int, set[int]] = defaultdict(set)
    entity_incident_triples: dict[int, set[int]] = defaultdict(set)
    triple_offsets: dict[int, set[int]] = defaultdict(set)

    for row in connection.execute(
        "SELECT triple_id, triple_type, subject_id, predicate, object_id, object_kind FROM triples"
    ):
        triple = {
            "triple_id": int(row["triple_id"]),
            "triple_type": str(row["triple_type"]),
            "subject_id": int(row["subject_id"]),
            "predicate": str(row["predicate"]),
            "object_id": int(row["object_id"]),
            "object_kind": str(row["object_kind"]),
        }
        triples.append(triple)
        triple_id = triple["triple_id"]
        subject_id = triple["subject_id"]
        object_id = triple["object_id"]
        entity_incident_triples[subject_id].add(triple_id)
        if triple["triple_type"] == "RELATION" and triple["object_kind"] == "entity":
            entity_incident_triples[object_id].add(triple_id)
            entity_neighbors[subject_id].add(object_id)
            entity_neighbors[object_id].add(subject_id)

    for row in connection.execute("SELECT triple_id, kv_offset FROM triple_kvs"):
        triple_offsets[int(row["triple_id"])].add(int(row["kv_offset"]))
    return GraphSnapshot(nodes, triples, entity_neighbors, entity_incident_triples, triple_offsets)


def _top_predicates(triples: Sequence[TripleRow], top_k: int, triple_type: str | None = None) -> list[dict[str, Any]]:
    counts = Counter(
        triple["predicate"]
        for triple in triples
        if triple_type is None or triple["triple_type"] == triple_type
    )
    return [{"predicate": predicate, "count": count} for predicate, count in counts.most_common(top_k)]


def _basic_graph_stats(
    connection: sqlite3.Connection,
    snapshot: GraphSnapshot,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    nodes = snapshot.nodes
    triples = snapshot.triples
    entity_ids = snapshot.entity_ids
    type_counts = Counter(triple["triple_type"] for triple in triples)
    predicate_counts = Counter(triple["predicate"] for triple in triples)
    in_degree = Counter({node_id: 0 for node_id in entity_ids})
    out_degree = Counter({node_id: 0 for node_id in entity_ids})
    relation_pairs: set[tuple[int, int]] = set()
    self_loops = 0
    disjoint_set = _DisjointSet(entity_ids)

    for triple in triples:
        if triple["triple_type"] != "RELATION" or triple["object_kind"] != "entity":
            continue
        subject_id = triple["subject_id"]
        object_id = triple["object_id"]
        out_degree[subject_id] += 1
        in_degree[object_id] += 1
        relation_pairs.add((subject_id, object_id))
        if subject_id == object_id:
            self_loops += 1
        if subject_id in entity_ids and object_id in entity_ids:
            disjoint_set.union(subject_id, object_id)

    total_degree = {node_id: in_degree[node_id] + out_degree[node_id] for node_id in entity_ids}
    top_entities = sorted(total_degree, key=lambda node_id: (-total_degree[node_id], nodes[node_id][0]))[:top_k]
    component_sizes = disjoint_set.component_sizes()
    largest_component = component_sizes[0] if component_sizes else 0
    triple_occurrences = _count(connection, "triple_sources")

    graph_summary = {
        "datasets": _count(connection, "datasets"),
        "nodes": len(nodes),
        "entities": len(entity_ids),
        "literals": sum(kind == "literal" for _, kind in nodes.values()),
        "unique_triples": len(triples),
        "triple_occurrences": triple_occurrences,
        "triple_kv_links": _count(connection, "triple_kvs"),
        "triple_types": dict(sorted(type_counts.items())),
        "unique_predicates": len(predicate_counts),
        "triple_dedup_ratio": (
            1.0 - len(triples) / triple_occurrences
            if triple_occurrences
            else 0.0
        ),
    }
    topology = {
        "relation_edges": type_counts.get("RELATION", 0),
        "unique_directed_entity_pairs": len(relation_pairs),
        "self_loops": self_loops,
        "entities_without_relations": sum(total_degree[node_id] == 0 for node_id in entity_ids),
        "in_degree": _percentile_summary(in_degree.values()),
        "out_degree": _percentile_summary(out_degree.values()),
        "total_degree": _percentile_summary(total_degree.values()),
        "weakly_connected_components": len(component_sizes),
        "single_entity_components": sum(size == 1 for size in component_sizes),
        "largest_component_entities": largest_component,
        "largest_component_ratio": largest_component / len(entity_ids) if entity_ids else 0.0,
        "component_size": _percentile_summary(component_sizes),
        "top_degree_entities": [
            {
                "node_id": node_id,
                "name": nodes[node_id][0],
                "in_degree": in_degree[node_id],
                "out_degree": out_degree[node_id],
                "total_degree": total_degree[node_id],
            }
            for node_id in top_entities
        ],
    }
    predicates = {
        "top": _top_predicates(triples, top_k),
        "relation_top": _top_predicates(triples, top_k, "RELATION"),
        "attribute_top": _top_predicates(triples, top_k, "ATTRIBUTE"),
    }
    return graph_summary, topology, predicates


def _entity_resolution_stats(
    connection: sqlite3.Connection,
    nodes: NodeMap,
    top_k: int,
) -> dict[str, Any]:
    alias_counts = {
        int(row["node_id"]): int(row["n"])
        for row in connection.execute("SELECT node_id, COUNT(*) AS n FROM aliases GROUP BY node_id")
    }
    ambiguous_aliases = [
        {"alias_key": str(row["alias_key"]), "node_count": int(row["n"])}
        for row in connection.execute(
            """
            SELECT alias_key, COUNT(DISTINCT node_id) AS n
            FROM aliases GROUP BY alias_key HAVING COUNT(DISTINCT node_id) > 1
            ORDER BY n DESC, alias_key LIMIT ?
            """,
            (top_k,),
        )
    ]
    entity_rows = [
        (node_id, name)
        for node_id, (name, kind) in nodes.items()
        if kind == "entity"
    ]
    suspicious = []
    for node_id, name in entity_rows:
        reasons = _suspicious_name_reasons(name)
        if reasons:
            suspicious.append({"node_id": node_id, "name": name, "reasons": reasons})

    top_alias_nodes = sorted(alias_counts, key=lambda node_id: (-alias_counts[node_id], nodes[node_id][0]))[:top_k]
    return {
        "aliases": _count(connection, "aliases"),
        "entities_with_aliases": sum(node_id in alias_counts for node_id, _ in entity_rows),
        "aliases_per_entity": _percentile_summary(alias_counts.get(node_id, 0) for node_id, _ in entity_rows),
        "ambiguous_alias_keys": len(
            connection.execute(
                """
                SELECT alias_key FROM aliases GROUP BY alias_key
                HAVING COUNT(DISTINCT node_id) > 1
                """
            ).fetchall()
        ),
        "ambiguous_alias_examples": ambiguous_aliases,
        "suspicious_entities": len(suspicious),
        "suspicious_entity_examples": suspicious[:top_k],
        "top_alias_entities": [
            {"node_id": node_id, "name": nodes[node_id][0], "alias_count": alias_counts[node_id]}
            for node_id in top_alias_nodes
        ],
    }


def _suspicious_name_reasons(name: str) -> list[str]:
    compact = "".join(character for character in name if not character.isspace())
    reasons = []
    if len(compact) <= 1:
        reasons.append("very_short")
    if compact.isdigit():
        reasons.append("numeric_only")
    if compact and not any(character.isalnum() for character in compact):
        reasons.append("symbol_only")
    if len(name) > 120:
        reasons.append("very_long")
    return reasons


def _kv_integrity_stats(
    store_dir: Path,
    kv_connection: sqlite3.Connection,
    graph_connection: sqlite3.Connection,
    triple_offsets: dict[int, set[int]],
) -> dict[str, Any]:
    kv_offsets = {
        int(row["offset"])
        for row in kv_connection.execute("SELECT offset FROM kv_records")
    }
    referenced_offsets = {offset for offsets in triple_offsets.values() for offset in offsets}
    invalid_offsets = sorted(referenced_offsets - kv_offsets)
    orphan_offsets = sorted(kv_offsets - referenced_offsets)
    kv_records = len(kv_offsets)
    kv_occurrences = _count(kv_connection, "kv_sources")
    triple_ids = [
        int(row["triple_id"])
        for row in graph_connection.execute("SELECT triple_id FROM triples")
    ]
    per_triple_counts = [len(triple_offsets.get(triple_id, set())) for triple_id in triple_ids]
    max_offset = max(kv_offsets) if kv_offsets else -1
    contiguous = not kv_offsets or (min(kv_offsets) == 0 and max_offset + 1 == kv_records)

    key_array = store_dir / "kv" / "key_tensors.npy"
    value_array = store_dir / "kv" / "value_tensors.npy"
    tensor_status: dict[str, Any] = {
        "key_array_exists": key_array.is_file(),
        "value_array_exists": value_array.is_file(),
        "complete": False,
        "rows": 0,
        "coverage_ratio": 0.0,
    }
    if key_array.is_file() and value_array.is_file():
        keys = np.load(key_array, mmap_mode="r")
        values = np.load(value_array, mmap_mode="r")
        rows = min(int(keys.shape[0]), int(values.shape[0]))
        tensor_status.update(
            {
                "complete": keys.shape[0] == values.shape[0] == kv_records,
                "rows": rows,
                "coverage_ratio": rows / kv_records if kv_records else 0.0,
                "key_shape": list(keys.shape),
                "value_shape": list(values.shape),
                "key_dtype": str(keys.dtype),
                "value_dtype": str(values.dtype),
            }
        )
    tensor_metadata = _read_json(store_dir / "kv" / "tensor_metadata.json")
    if tensor_metadata is not None:
        tensor_status["metadata"] = tensor_metadata

    return {
        "records": kv_records,
        "occurrences": kv_occurrences,
        "dedup_ratio": 1.0 - kv_records / kv_occurrences if kv_occurrences else 0.0,
        "offsets_contiguous": contiguous,
        "max_offset": max_offset,
        "referenced_offsets": len(referenced_offsets),
        "invalid_references": len(invalid_offsets),
        "invalid_reference_examples": invalid_offsets[:20],
        "orphan_records": len(orphan_offsets),
        "orphan_offset_examples": orphan_offsets[:20],
        "triples_without_kv": sum(count == 0 for count in per_triple_counts),
        "kv_offsets_per_triple": _percentile_summary(per_triple_counts),
        "tensor_arrays": tensor_status,
    }


def _dataset_stats(
    graph_connection: sqlite3.Connection,
    kv_connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    graph_rows = {
        str(row["dataset_id"]): {
            "source_path": str(row["source_path"]),
            "samples": 0,
            "triple_occurrences": 0,
            "unique_triples": 0,
        }
        for row in graph_connection.execute("SELECT dataset_id, source_path FROM datasets")
    }
    for row in graph_connection.execute(
        """
        SELECT dataset_id, COUNT(DISTINCT sample_id) AS samples,
               COUNT(*) AS occurrences, COUNT(DISTINCT triple_id) AS unique_triples
        FROM triple_sources GROUP BY dataset_id
        """
    ):
        item = graph_rows.setdefault(str(row["dataset_id"]), {"source_path": ""})
        item.update(
            {
                "samples": int(row["samples"]),
                "triple_occurrences": int(row["occurrences"]),
                "unique_triples": int(row["unique_triples"]),
            }
        )
    kv_rows = {
        str(row["dataset_id"]): (int(row["occurrences"]), int(row["unique_kv"]))
        for row in kv_connection.execute(
            """
            SELECT dataset_id, COUNT(*) AS occurrences, COUNT(DISTINCT offset) AS unique_kv
            FROM kv_sources GROUP BY dataset_id
            """
        )
    }
    return [
        {
            "dataset_id": dataset_id,
            **values,
            "kv_occurrences": kv_rows.get(dataset_id, (0, 0))[0],
            "unique_kv": kv_rows.get(dataset_id, (0, 0))[1],
        }
        for dataset_id, values in sorted(graph_rows.items())
    ]


def _local_expansion_stats(
    snapshot: GraphSnapshot,
    *,
    max_hops: int,
    max_seeds: int,
    random_seed: int,
) -> dict[str, Any]:
    entity_ids = sorted(snapshot.entity_ids)
    sampled = entity_ids
    if max_seeds > 0 and len(entity_ids) > max_seeds:
        sampled = sorted(random.Random(random_seed).sample(entity_ids, max_seeds))

    metrics = {
        hop: {"entities": [], "triples": [], "kv_offsets": []}
        for hop in range(1, max_hops + 1)
    }
    for seed_id in sampled:
        visited_entities = {seed_id}
        frontier = {seed_id}
        selected_triples: set[int] = set()
        for hop in range(1, max_hops + 1):
            next_frontier: set[int] = set()
            for node_id in frontier:
                selected_triples.update(snapshot.entity_incident_triples.get(node_id, set()))
                next_frontier.update(snapshot.entity_neighbors.get(node_id, set()))
            next_frontier -= visited_entities
            visited_entities.update(next_frontier)
            frontier = next_frontier
            selected_offsets = {
                offset
                for triple_id in selected_triples
                for offset in snapshot.triple_offsets.get(triple_id, set())
            }
            metrics[hop]["entities"].append(len(visited_entities))
            metrics[hop]["triples"].append(len(selected_triples))
            metrics[hop]["kv_offsets"].append(len(selected_offsets))

    return {
        "total_entities": len(entity_ids),
        "sampled_entities": len(sampled),
        "random_seed": random_seed,
        "by_hop": {
            str(hop): {
                name: _percentile_summary(values)
                for name, values in hop_metrics.items()
            }
            for hop, hop_metrics in metrics.items()
        },
    }


def _entity_index_stats(store_dir: Path, entity_count: int) -> dict[str, Any]:
    graph_dir = store_dir / "graph"
    ids_path = graph_dir / "entity_vector_ids.npy"
    vectors_path = graph_dir / "entity_vectors.npy"
    hnsw_path = graph_dir / "entity_hnsw.bin"
    hnsw_meta_path = graph_dir / "entity_hnsw.json"
    vector_meta_path = graph_dir / "entity_vectors.json"
    report: dict[str, Any] = {
        "entity_embeddings_exist": ids_path.is_file() and vectors_path.is_file(),
        "hnsw_exists": hnsw_path.is_file() and hnsw_meta_path.is_file(),
        "entity_count": entity_count,
        "embedding_rows": 0,
        "coverage_ratio": 0.0,
    }
    if ids_path.is_file() and vectors_path.is_file():
        ids = np.load(ids_path, mmap_mode="r")
        vectors = np.load(vectors_path, mmap_mode="r")
        rows = min(int(ids.shape[0]), int(vectors.shape[0]))
        norms = np.linalg.norm(np.asarray(vectors), axis=1) if rows else np.asarray([])
        report.update(
            {
                "embedding_rows": rows,
                "coverage_ratio": rows / entity_count if entity_count else 0.0,
                "embedding_dimension": int(vectors.shape[1]) if vectors.ndim == 2 else None,
                "id_vector_rows_match": ids.shape[0] == vectors.shape[0],
                "zero_norm_vectors": int(np.sum(norms <= 1e-12)),
                "non_finite_values": int(np.size(vectors) - np.count_nonzero(np.isfinite(vectors))),
            }
        )
    vector_metadata = _read_json(vector_meta_path)
    hnsw_metadata = _read_json(hnsw_meta_path)
    if vector_metadata is not None:
        report["embedding_metadata"] = vector_metadata
    if hnsw_metadata is not None:
        report["hnsw_metadata"] = hnsw_metadata
    return report


def inspect_store(
    store_dir: str | Path,
    *,
    top_k: int = 20,
    max_hops: int = 2,
    max_seeds: int = 1000,
    random_seed: int = 1,
) -> dict[str, Any]:
    root = Path(store_dir)
    graph_db = root / "graph" / "graph_store.sqlite3"
    kv_db = root / "kv" / "kv_store.sqlite3"
    graph_connection = _readonly_connection(graph_db)
    kv_connection = _readonly_connection(kv_db)
    try:
        snapshot = _load_graph(graph_connection)
        graph_summary, topology, predicates = _basic_graph_stats(graph_connection, snapshot, top_k)
        kv_integrity = _kv_integrity_stats(
            root,
            kv_connection,
            graph_connection,
            snapshot.triple_offsets,
        )
        entity_count = len(snapshot.entity_ids)
        return {
            "store_dir": str(root.resolve()),
            "graph": graph_summary,
            "topology": topology,
            "predicates": predicates,
            "entity_resolution": _entity_resolution_stats(graph_connection, snapshot.nodes, top_k),
            "kv_integrity": kv_integrity,
            "datasets": _dataset_stats(graph_connection, kv_connection),
            "local_expansion": _local_expansion_stats(
                snapshot,
                max_hops=max_hops,
                max_seeds=max_seeds,
                random_seed=random_seed,
            ),
            "entity_index": _entity_index_stats(root, entity_count),
        }
    finally:
        graph_connection.close()
        kv_connection.close()


def print_summary(report: dict[str, Any]) -> None:
    graph = report["graph"]
    topology = report["topology"]
    kv = report["kv_integrity"]
    print(f"PathWeaver store: {report['store_dir']}")
    print(
        "Graph: "
        f"datasets={graph['datasets']} entities={graph['entities']} literals={graph['literals']} "
        f"triples={graph['unique_triples']} occurrences={graph['triple_occurrences']}"
    )
    print(f"Triple types: {graph['triple_types']}")
    degree = topology["total_degree"]
    print(
        "Entity degree (relation-only): "
        f"mean={degree['mean']:.2f} p50={degree['p50']:.1f} p95={degree['p95']:.1f} "
        f"p99={degree['p99']:.1f} max={degree['max']}"
    )
    print(
        "Connectivity: "
        f"components={topology['weakly_connected_components']} "
        f"largest={topology['largest_component_entities']} "
        f"({topology['largest_component_ratio']:.2%})"
    )
    print(
        "KV integrity: "
        f"records={kv['records']} referenced={kv['referenced_offsets']} "
        f"invalid={kv['invalid_references']} orphan={kv['orphan_records']} "
        f"triples_without_kv={kv['triples_without_kv']}"
    )
    entity_model = report["entity_index"].get("embedding_metadata", {}).get("model_path")
    kv_model = kv.get("tensor_arrays", {}).get("metadata", {}).get("model_path")
    if entity_model or kv_model:
        print(f"Embedding models: HNSW={entity_model or 'not built'}; KV={kv_model or 'not built'}")
    for hop, values in report["local_expansion"]["by_hop"].items():
        print(
            f"{hop}-hop candidate size: "
            f"triples p50={values['triples']['p50']:.1f} p95={values['triples']['p95']:.1f} "
            f"p99={values['triples']['p99']:.1f}; "
            f"KV p50={values['kv_offsets']['p50']:.1f} p95={values['kv_offsets']['p95']:.1f}"
        )
    print("Top predicates: " + ", ".join(
        f"{item['predicate']}={item['count']}" for item in report["predicates"]["top"][:10]
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=1000,
        help="Maximum entities sampled for local expansion statistics; 0 means all entities.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.max_hops < 1:
        raise ValueError("--max-hops must be positive")
    if args.max_seeds < 0:
        raise ValueError("--max-seeds cannot be negative")
    report = inspect_store(
        args.store_dir,
        top_k=args.top_k,
        max_hops=args.max_hops,
        max_seeds=args.max_seeds,
        random_seed=args.seed,
    )
    print_summary(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
