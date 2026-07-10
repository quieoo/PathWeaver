"""Graph serving snapshot for Store V2."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kblam.stores.graph_store import GraphStore, GraphTriple
from kblam.stores.common import canonical_entity_key
from kblam.stores.io import write_json_atomic


@dataclass(frozen=True)
class SnapshotTriple:
    triple_id: int
    triple_type: str
    subject_pos: int
    predicate: str
    object_node_id: int
    object_name: str
    object_kind: str
    kv_offsets: tuple[int, ...]
    title: str


class GraphStoreV2:
    """Entity-centric in-memory graph snapshot with optional copied ANN assets."""

    MANIFEST_NAME = "manifest.json"
    ENTITY_IDS_NAME = "entity_vector_ids.npy"
    ENTITY_VECTORS_NAME = "entity_vectors.npy"
    HNSW_INDEX_NAME = "entity_hnsw.bin"
    HNSW_META_NAME = "entity_hnsw.json"

    def __init__(self, root: str | Path, *, create: bool = False) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            raise FileNotFoundError(self.root)
        self.manifest_path = self.root / self.MANIFEST_NAME
        self.entity_ids_path = self.root / self.ENTITY_IDS_NAME
        self.entity_vectors_path = self.root / self.ENTITY_VECTORS_NAME
        self.hnsw_index_path = self.root / self.HNSW_INDEX_NAME
        self.hnsw_meta_path = self.root / self.HNSW_META_NAME
        self._manifest: dict[str, Any] | None = None
        self._hnsw_index: Any | None = None
        self._hnsw_meta: dict[str, Any] | None = None
        self._entity_node_ids: np.ndarray | None = None
        self._entity_names: list[str] | None = None
        self._entity_name_keys: list[str] | None = None
        self._entity_alias_keys: list[list[str]] | None = None
        self._forward_index: np.ndarray | None = None
        self._forward_triples: np.ndarray | None = None
        self._reverse_index: np.ndarray | None = None
        self._reverse_triples: np.ndarray | None = None
        self._triple_subject_pos: np.ndarray | None = None
        self._triple_object_node_id: np.ndarray | None = None
        self._triple_object_name: list[str] | None = None
        self._triple_object_kind: np.ndarray | None = None
        self._triple_predicate: list[str] | None = None
        self._triple_type: np.ndarray | None = None
        self._triple_title: list[str] | None = None
        self._triple_kv_index: np.ndarray | None = None
        self._triple_kv_offsets: np.ndarray | None = None
        self._triple_ids: np.ndarray | None = None
        self._node_id_to_entity_pos: dict[int, int] | None = None

    def __enter__(self) -> "GraphStoreV2":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._hnsw_index = None
        self._hnsw_meta = None

    def stats(self) -> dict[str, int]:
        manifest = self._load_manifest()
        return {
            "entities": int(manifest["entities"]),
            "triples": int(manifest["triples"]),
            "triple_kvs": int(manifest["triple_kvs"]),
        }

    def entity_nodes(self) -> list[tuple[int, str]]:
        ids = self._load_entity_node_ids()
        names = self._load_entity_names()
        return [(int(node_id), names[index]) for index, node_id in enumerate(ids.tolist())]

    def get_node_name(self, node_id: int) -> str:
        pos = self._node_id_to_pos().get(int(node_id))
        if pos is None:
            raise KeyError(node_id)
        return self._load_entity_names()[pos]

    def search_entities(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 10,
        backend: str = "auto",
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        if backend not in {"auto", "exact", "hnsw"}:
            raise ValueError(f"Unsupported vector search backend: {backend}")
        if backend in {"auto", "hnsw"} and self.hnsw_index_path.exists():
            try:
                return self._search_hnsw(query_embedding, top_k)
            except ImportError:
                if backend == "hnsw":
                    raise
        return self._search_exact(query_embedding, top_k)

    def shortlist_entity_hits(
        self,
        query_key: str | None,
        *,
        final_top_k: int,
        candidate_top_k: int,
        min_mention_chars: int,
        backend: str = "auto",
        query_embedding: np.ndarray | None = None,
        candidates: Sequence[tuple[int, float]] | None = None,
    ) -> list[tuple[int, float, str]]:
        if candidates is None:
            if query_embedding is None:
                raise ValueError("shortlist_entity_hits requires candidates or query_embedding")
            candidates = self.search_entities(query_embedding, top_k=max(final_top_k, candidate_top_k), backend=backend)
        if not query_key:
            return [(node_id, score, "vector") for node_id, score in candidates[:final_top_k]]
        ranked = []
        aliases = self._load_entity_alias_keys()
        names = self._load_entity_names()
        pos_map = self._node_id_to_pos()
        for node_id, score in candidates:
            pos = pos_map[int(node_id)]
            alias_bonus = 0.0
            matched = False
            for alias_key in aliases[pos]:
                if len(alias_key) < min_mention_chars:
                    continue
                if f" {alias_key} " in query_key:
                    matched = True
                    alias_bonus = max(alias_bonus, 2.0 + min(0.5, len(alias_key) / 64.0))
            ranked.append((alias_bonus + float(score), int(node_id), float(score), "hybrid" if matched else "vector"))
        ranked.sort(key=lambda item: (-item[0], -item[2], item[1]))
        return [(node_id, vector_score, source) for _, node_id, vector_score, source in ranked[:final_top_k]]

    def get_local_subgraph(
        self,
        seed_node_ids: Sequence[int],
        *,
        hops: int = 1,
        max_triples: int | None = None,
        max_incident_triples_per_node: int | None = None,
    ) -> tuple[list[int], list[GraphTriple]]:
        if hops < 0:
            raise ValueError("hops must be non-negative")
        if max_incident_triples_per_node is not None and max_incident_triples_per_node <= 0:
            raise ValueError("max_incident_triples_per_node must be positive when provided")
        pos_map = self._node_id_to_pos()
        visited: set[int] = {int(node_id) for node_id in seed_node_ids}
        queue = deque((pos_map[int(node_id)], 0) for node_id in seed_node_ids if int(node_id) in pos_map)
        triples: dict[int, GraphTriple] = {}
        while queue:
            entity_pos, depth = queue.popleft()
            if depth >= hops:
                continue
            triple_indexes = self._incident_triple_indexes(entity_pos)
            if max_incident_triples_per_node is not None:
                triple_indexes = triple_indexes[:max_incident_triples_per_node]
            for triple_idx in triple_indexes:
                triple = self._get_triple_by_index(triple_idx)
                triples.setdefault(triple.triple_id, triple)
                if max_triples is not None and len(triples) >= max_triples:
                    return sorted(visited), list(triples.values())
                for neighbor_id in (triple.subject_id, triple.object_id):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    neighbor_pos = pos_map.get(int(neighbor_id))
                    if neighbor_pos is not None:
                        queue.append((neighbor_pos, depth + 1))
        return sorted(visited), list(triples.values())

    @classmethod
    def export_from_v1(
        cls,
        root: str | Path,
        source_store: GraphStore,
        *,
        subject_only_entities: bool = False,
    ) -> "GraphStoreV2":
        dst = cls(root, create=True)
        conn = sqlite3.connect(f"file:{source_store.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            if subject_only_entities:
                entities = conn.execute(
                    """
                    SELECT DISTINCT n.node_id, n.canonical_name
                    FROM triples t
                    JOIN nodes n ON n.node_id = t.subject_id
                    WHERE n.kind = 'entity'
                    ORDER BY n.node_id
                    """
                ).fetchall()
            else:
                entities = conn.execute(
                    "SELECT node_id, canonical_name FROM nodes WHERE kind = 'entity' ORDER BY node_id"
                ).fetchall()
            entity_node_ids = np.asarray([int(row["node_id"]) for row in entities], dtype=np.int64)
            entity_names = [str(row["canonical_name"]) for row in entities]
            node_id_to_entity_pos = {int(node_id): idx for idx, node_id in enumerate(entity_node_ids.tolist())}
            entity_node_id_set = set(entity_node_ids.tolist())

            alias_rows = conn.execute(
                """
                SELECT node_id, alias_name, alias_key
                FROM aliases
                WHERE node_id IN (SELECT node_id FROM nodes WHERE kind = 'entity')
                ORDER BY node_id, alias_name
                """
            ).fetchall()
            alias_map: dict[int, list[dict[str, str]]] = {node_id: [] for node_id in entity_node_ids.tolist()}
            for row in alias_rows:
                node_id = int(row["node_id"])
                if node_id not in alias_map:
                    continue
                alias_map[node_id].append(
                    {"alias_name": str(row["alias_name"]), "alias_key": str(row["alias_key"])}
                )

            triples = conn.execute(
                """
                SELECT
                    t.triple_id,
                    t.triple_type,
                    t.subject_id,
                    t.predicate,
                    t.object_id,
                    t.object_kind,
                    o.canonical_name AS object_name,
                    COALESCE(
                        (
                            SELECT ts.title FROM triple_sources ts
                            WHERE ts.triple_id = t.triple_id AND ts.title != ''
                            ORDER BY ts.dataset_id, ts.source_index, ts.triple_index
                            LIMIT 1
                        ),
                        ''
                    ) AS source_title
                FROM triples t
                JOIN nodes o ON o.node_id = t.object_id
                ORDER BY t.triple_id
                """
            ).fetchall()
            if subject_only_entities:
                triples = [row for row in triples if int(row["subject_id"]) in entity_node_id_set]

            triple_ids = np.asarray([int(row["triple_id"]) for row in triples], dtype=np.int64)
            triple_subject_pos = np.asarray(
                [node_id_to_entity_pos[int(row["subject_id"])] for row in triples], dtype=np.int64
            )
            triple_object_node_id = np.asarray([int(row["object_id"]) for row in triples], dtype=np.int64)
            triple_object_kind = np.asarray([str(row["object_kind"]) for row in triples], dtype="U8")
            triple_type = np.asarray([str(row["triple_type"]) for row in triples], dtype="U16")
            triple_predicate = [str(row["predicate"]) for row in triples]
            triple_object_name = [str(row["object_name"]) for row in triples]
            triple_title = [str(row["source_title"]) for row in triples]

            kv_rows = conn.execute(
                "SELECT triple_id, kv_offset FROM triple_kvs ORDER BY triple_id, ordinal, kv_offset"
            ).fetchall()
            kv_map: dict[int, list[int]] = {int(triple_id): [] for triple_id in triple_ids.tolist()}
            for row in kv_rows:
                kv_map[int(row["triple_id"])].append(int(row["kv_offset"]))
            triple_kv_offsets: list[int] = []
            triple_kv_index = [0]
            for triple_id in triple_ids.tolist():
                offsets = kv_map.get(int(triple_id), [])
                triple_kv_offsets.extend(offsets)
                triple_kv_index.append(len(triple_kv_offsets))

            forward_adj: list[list[int]] = [[] for _ in range(len(entity_node_ids))]
            reverse_adj: list[list[int]] = [[] for _ in range(len(entity_node_ids))]
            triple_id_to_index = {int(triple_id): idx for idx, triple_id in enumerate(triple_ids.tolist())}
            for triple_index, row in enumerate(triples):
                subject_pos = node_id_to_entity_pos[int(row["subject_id"])]
                forward_adj[subject_pos].append(triple_index)
                object_pos = node_id_to_entity_pos.get(int(row["object_id"]))
                if object_pos is not None:
                    reverse_adj[object_pos].append(triple_index)

            def flatten_index(items: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
                index = [0]
                flat = []
                for group in items:
                    flat.extend(group)
                    index.append(len(flat))
                return np.asarray(index, dtype=np.int64), np.asarray(flat, dtype=np.int64)

            forward_index, forward_triples = flatten_index(forward_adj)
            reverse_index, reverse_triples = flatten_index(reverse_adj)

            np.save(dst.root / "entity_node_ids.npy", entity_node_ids)
            write_json_atomic(dst.root / "entity_names.json", entity_names)
            write_json_atomic(
                dst.root / "entity_aliases.json",
                [alias_map[int(node_id)] for node_id in entity_node_ids.tolist()],
            )
            np.save(dst.root / "forward_index.npy", forward_index)
            np.save(dst.root / "forward_triples.npy", forward_triples)
            np.save(dst.root / "reverse_index.npy", reverse_index)
            np.save(dst.root / "reverse_triples.npy", reverse_triples)
            np.save(dst.root / "triple_ids.npy", triple_ids)
            np.save(dst.root / "triple_subject_pos.npy", triple_subject_pos)
            np.save(dst.root / "triple_object_node_id.npy", triple_object_node_id)
            np.save(dst.root / "triple_object_kind.npy", triple_object_kind)
            np.save(dst.root / "triple_type.npy", triple_type)
            write_json_atomic(dst.root / "triple_predicates.json", triple_predicate)
            write_json_atomic(dst.root / "triple_object_names.json", triple_object_name)
            write_json_atomic(dst.root / "triple_titles.json", triple_title)
            np.save(dst.root / "triple_kv_index.npy", np.asarray(triple_kv_index, dtype=np.int64))
            np.save(dst.root / "triple_kv_offsets.npy", np.asarray(triple_kv_offsets, dtype=np.int64))

            if source_store.entity_ids_path.exists() and source_store.entity_vectors_path.exists():
                src_ids, src_vectors = source_store._load_entity_vectors()
                src_ids_list = [int(node_id) for node_id in src_ids.tolist()]
                src_pos = {node_id: idx for idx, node_id in enumerate(src_ids_list)}
                kept_ids = [node_id for node_id in entity_node_ids.tolist() if int(node_id) in src_pos]
                kept_vectors = np.asarray([src_vectors[src_pos[int(node_id)]] for node_id in kept_ids], dtype=np.float32)
                np.save(dst.entity_ids_path, np.asarray(kept_ids, dtype=np.int64))
                np.save(dst.entity_vectors_path, kept_vectors)
                if source_store.hnsw_index_path.exists() and source_store.hnsw_meta_path.exists():
                    try:
                        hnswlib = cls._require_hnswlib()
                        index = hnswlib.Index(space="cosine", dim=int(kept_vectors.shape[1]))
                        index.init_index(max_elements=int(len(kept_ids)), ef_construction=200, M=16)
                        index.add_items(kept_vectors, np.asarray(kept_ids, dtype=np.int64))
                        index.set_ef(max(50, min(200, int(len(kept_ids)))))
                        index.save_index(str(dst.hnsw_index_path))
                        write_json_atomic(
                            dst.hnsw_meta_path,
                            {"space": "cosine", "dim": int(kept_vectors.shape[1]), "count": int(len(kept_ids))},
                        )
                    except ImportError:
                        pass

            write_json_atomic(
                dst.manifest_path,
                {
                    "entities": int(entity_node_ids.shape[0]),
                    "triples": int(triple_ids.shape[0]),
                    "triple_kvs": int(len(triple_kv_offsets)),
                    "exported_from_v1": str(source_store.root),
                    "subject_only_entities": bool(subject_only_entities),
                },
            )
            return dst
        finally:
            conn.close()

    def _incident_triple_indexes(self, entity_pos: int) -> list[int]:
        forward_index = self._load_forward_index()
        forward_triples = self._load_forward_triples()
        reverse_index = self._load_reverse_index()
        reverse_triples = self._load_reverse_triples()
        seen: set[int] = set()
        result: list[int] = []
        for array_index, array_values in (
            (forward_index, forward_triples),
            (reverse_index, reverse_triples),
        ):
            start = int(array_index[entity_pos])
            end = int(array_index[entity_pos + 1])
            for triple_idx in array_values[start:end].tolist():
                triple_idx = int(triple_idx)
                if triple_idx in seen:
                    continue
                seen.add(triple_idx)
                result.append(triple_idx)
        return result

    def _get_triple_by_index(self, triple_index: int) -> GraphTriple:
        triple_ids = self._load_triple_ids()
        subject_pos = self._load_triple_subject_pos()
        object_node_ids = self._load_triple_object_node_id()
        object_kind = self._load_triple_object_kind()
        triple_types = self._load_triple_type()
        predicates = self._load_triple_predicate()
        object_names = self._load_triple_object_name()
        titles = self._load_triple_title()
        kv_index = self._load_triple_kv_index()
        kv_offsets = self._load_triple_kv_offsets()
        entity_node_ids = self._load_entity_node_ids()
        entity_names = self._load_entity_names()
        start = int(kv_index[triple_index])
        end = int(kv_index[triple_index + 1])
        subject_entity_pos = int(subject_pos[triple_index])
        return GraphTriple(
            triple_id=int(triple_ids[triple_index]),
            triple_type=str(triple_types[triple_index]),
            subject_id=int(entity_node_ids[subject_entity_pos]),
            subject=entity_names[subject_entity_pos],
            predicate=predicates[triple_index],
            object_id=int(object_node_ids[triple_index]),
            object=object_names[triple_index],
            object_kind=str(object_kind[triple_index]),
            kv_offsets=tuple(int(offset) for offset in kv_offsets[start:end].tolist()),
            title=titles[triple_index],
        )

    def _search_exact(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        ids = np.load(self.entity_ids_path, mmap_mode="r")
        vectors = np.load(self.entity_vectors_path, mmap_mode="r")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != vectors.shape[1]:
            raise ValueError(f"Expected query dimension {vectors.shape[1]}, got {query.shape[0]}")
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = np.asarray(vectors @ query)
        k = min(top_k, scores.shape[0])
        if k == 0:
            return []
        selected = np.argpartition(-scores, k - 1)[:k]
        selected = selected[np.argsort(-scores[selected])]
        return [(int(ids[index]), float(scores[index])) for index in selected]

    def _search_hnsw(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        hnswlib = self._require_hnswlib()
        if self._hnsw_index is None or self._hnsw_meta is None:
            meta = json.loads(self.hnsw_meta_path.read_text(encoding="utf-8"))
            index = hnswlib.Index(space=meta["space"], dim=int(meta["dim"]))
            index.load_index(str(self.hnsw_index_path), max_elements=int(meta["count"]))
            index.set_ef(max(50, min(200, int(meta["count"]))))
            self._hnsw_index = index
            self._hnsw_meta = meta
        meta = self._hnsw_meta
        assert meta is not None
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != int(meta["dim"]):
            raise ValueError(f"Expected query dimension {meta['dim']}, got {query.shape[1]}")
        k = min(top_k, int(meta["count"]))
        labels, distances = self._hnsw_index.knn_query(query, k=k)
        return [(int(node_id), float(1.0 - distance)) for node_id, distance in zip(labels[0], distances[0])]

    @staticmethod
    def _require_hnswlib():
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError("hnswlib is required for HNSW entity indexing and search") from exc
        return hnswlib

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    def _load_entity_node_ids(self) -> np.ndarray:
        if self._entity_node_ids is None:
            self._entity_node_ids = np.load(self.root / "entity_node_ids.npy", mmap_mode="r")
        return self._entity_node_ids

    def _load_entity_names(self) -> list[str]:
        if self._entity_names is None:
            self._entity_names = json.loads((self.root / "entity_names.json").read_text(encoding="utf-8"))
        return self._entity_names

    def _load_entity_alias_keys(self) -> list[list[str]]:
        if self._entity_alias_keys is None:
            payload = json.loads((self.root / "entity_aliases.json").read_text(encoding="utf-8"))
            self._entity_alias_keys = []
            self._entity_name_keys = []
            for index, aliases in enumerate(payload):
                current = sorted({str(alias["alias_key"]) for alias in aliases if str(alias.get("alias_key", ""))})
                name = self._load_entity_names()[index]
                if not current:
                    current = [canonical_entity_key(name)]
                self._entity_alias_keys.append(current)
                self._entity_name_keys.append(name.casefold())
        return self._entity_alias_keys

    def _node_id_to_pos(self) -> dict[int, int]:
        if self._node_id_to_entity_pos is None:
            ids = self._load_entity_node_ids()
            self._node_id_to_entity_pos = {int(node_id): idx for idx, node_id in enumerate(ids.tolist())}
        return self._node_id_to_entity_pos

    def _load_forward_index(self) -> np.ndarray:
        if self._forward_index is None:
            self._forward_index = np.load(self.root / "forward_index.npy", mmap_mode="r")
        return self._forward_index

    def _load_forward_triples(self) -> np.ndarray:
        if self._forward_triples is None:
            self._forward_triples = np.load(self.root / "forward_triples.npy", mmap_mode="r")
        return self._forward_triples

    def _load_reverse_index(self) -> np.ndarray:
        if self._reverse_index is None:
            self._reverse_index = np.load(self.root / "reverse_index.npy", mmap_mode="r")
        return self._reverse_index

    def _load_reverse_triples(self) -> np.ndarray:
        if self._reverse_triples is None:
            self._reverse_triples = np.load(self.root / "reverse_triples.npy", mmap_mode="r")
        return self._reverse_triples

    def _load_triple_ids(self) -> np.ndarray:
        if self._triple_ids is None:
            self._triple_ids = np.load(self.root / "triple_ids.npy", mmap_mode="r")
        return self._triple_ids

    def _load_triple_subject_pos(self) -> np.ndarray:
        if self._triple_subject_pos is None:
            self._triple_subject_pos = np.load(self.root / "triple_subject_pos.npy", mmap_mode="r")
        return self._triple_subject_pos

    def _load_triple_object_node_id(self) -> np.ndarray:
        if self._triple_object_node_id is None:
            self._triple_object_node_id = np.load(self.root / "triple_object_node_id.npy", mmap_mode="r")
        return self._triple_object_node_id

    def _load_triple_object_kind(self) -> np.ndarray:
        if self._triple_object_kind is None:
            self._triple_object_kind = np.load(self.root / "triple_object_kind.npy", mmap_mode="r")
        return self._triple_object_kind

    def _load_triple_type(self) -> np.ndarray:
        if self._triple_type is None:
            self._triple_type = np.load(self.root / "triple_type.npy", mmap_mode="r")
        return self._triple_type

    def _load_triple_predicate(self) -> list[str]:
        if self._triple_predicate is None:
            self._triple_predicate = json.loads((self.root / "triple_predicates.json").read_text(encoding="utf-8"))
        return self._triple_predicate

    def _load_triple_object_name(self) -> list[str]:
        if self._triple_object_name is None:
            self._triple_object_name = json.loads((self.root / "triple_object_names.json").read_text(encoding="utf-8"))
        return self._triple_object_name

    def _load_triple_title(self) -> list[str]:
        if self._triple_title is None:
            self._triple_title = json.loads((self.root / "triple_titles.json").read_text(encoding="utf-8"))
        return self._triple_title

    def _load_triple_kv_index(self) -> np.ndarray:
        if self._triple_kv_index is None:
            self._triple_kv_index = np.load(self.root / "triple_kv_index.npy", mmap_mode="r")
        return self._triple_kv_index

    def _load_triple_kv_offsets(self) -> np.ndarray:
        if self._triple_kv_offsets is None:
            self._triple_kv_offsets = np.load(self.root / "triple_kv_offsets.npy", mmap_mode="r")
        return self._triple_kv_offsets
