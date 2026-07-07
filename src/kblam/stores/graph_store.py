"""Entity-centered graph store for retrieving local triple subgraphs."""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from kblam.stores.common import canonical_entity_key, canonical_relation, content_hash, normalize_text
from kblam.stores.io import open_sqlite, unlink_existing, write_json_atomic


ENTITY = "entity"
LITERAL = "literal"


@dataclass(frozen=True)
class ResolvedEntity:
    canonical_name: str
    canonical_key: str


class EntityResolver:
    """Deterministic resolver with an optional externally supplied alias map."""

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        self._aliases = {
            canonical_entity_key(alias): normalize_text(canonical)
            for alias, canonical in (aliases or {}).items()
            if canonical_entity_key(alias) and normalize_text(canonical)
        }

    def resolve(self, name: str) -> ResolvedEntity:
        display_name = normalize_text(name)
        alias_key = canonical_entity_key(display_name)
        canonical_name = self._aliases.get(alias_key, display_name)
        return ResolvedEntity(canonical_name, canonical_entity_key(canonical_name))


@dataclass(frozen=True)
class GraphTriple:
    triple_id: int
    triple_type: str
    subject_id: int
    subject: str
    predicate: str
    object_id: int
    object: str
    object_kind: str
    kv_offsets: tuple[int, ...]
    title: str = ""


class GraphStore:
    """Persistent entity graph with triple-to-KV-offset mappings."""

    DB_NAME = "graph_store.sqlite3"
    ENTITY_IDS_NAME = "entity_vector_ids.npy"
    ENTITY_VECTORS_NAME = "entity_vectors.npy"
    ENTITY_VECTOR_META_NAME = "entity_vectors.json"
    HNSW_INDEX_NAME = "entity_hnsw.bin"
    HNSW_META_NAME = "entity_hnsw.json"

    def __init__(
        self,
        root: str | Path,
        *,
        resolver: EntityResolver | None = None,
        create: bool = True,
    ) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            raise FileNotFoundError(self.root)
        self.resolver = resolver or EntityResolver()
        self.db_path = self.root / self.DB_NAME
        self.entity_ids_path = self.root / self.ENTITY_IDS_NAME
        self.entity_vectors_path = self.root / self.ENTITY_VECTORS_NAME
        self.entity_vector_meta_path = self.root / self.ENTITY_VECTOR_META_NAME
        self.hnsw_index_path = self.root / self.HNSW_INDEX_NAME
        self.hnsw_meta_path = self.root / self.HNSW_META_NAME
        self._hnsw_index: Any | None = None
        self._hnsw_meta: dict[str, Any] | None = None
        self._conn = open_sqlite(self.db_path)
        self._create_schema()
        self._registered_dataset_ids = {
            str(row["dataset_id"])
            for row in self._conn.execute("SELECT dataset_id FROM datasets")
        }

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_key TEXT NOT NULL UNIQUE,
                canonical_name TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('entity', 'literal'))
            );

            CREATE TABLE IF NOT EXISTS aliases (
                alias_key TEXT NOT NULL,
                alias_name TEXT NOT NULL,
                node_id INTEGER NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                PRIMARY KEY (alias_key, node_id)
            );
            CREATE INDEX IF NOT EXISTS idx_alias_key ON aliases(alias_key);

            CREATE TABLE IF NOT EXISTS triples (
                triple_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_key TEXT NOT NULL UNIQUE,
                triple_type TEXT NOT NULL,
                subject_id INTEGER NOT NULL REFERENCES nodes(node_id),
                predicate TEXT NOT NULL,
                object_id INTEGER NOT NULL REFERENCES nodes(node_id),
                object_kind TEXT NOT NULL CHECK(object_kind IN ('entity', 'literal'))
            );
            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject_id);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object_id);

            CREATE TABLE IF NOT EXISTS triple_kvs (
                triple_id INTEGER NOT NULL REFERENCES triples(triple_id) ON DELETE CASCADE,
                kv_offset INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (triple_id, kv_offset)
            );

            CREATE TABLE IF NOT EXISTS triple_sources (
                triple_id INTEGER NOT NULL REFERENCES triples(triple_id) ON DELETE CASCADE,
                dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
                sample_id TEXT NOT NULL,
                source_index INTEGER NOT NULL,
                triple_index INTEGER NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (triple_id, dataset_id, sample_id, triple_index)
            );
            """
        )
        self._conn.commit()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._hnsw_index = None
        self._hnsw_meta = None
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()
        self._registered_dataset_ids = {
            str(row["dataset_id"])
            for row in self._conn.execute("SELECT dataset_id FROM datasets")
        }

    def register_dataset(
        self,
        dataset_id: str,
        source_path: str = "",
        *,
        commit: bool = True,
    ) -> None:
        dataset_id = str(dataset_id)
        if dataset_id in self._registered_dataset_ids and not source_path:
            return
        self._conn.execute(
            """
            INSERT INTO datasets(dataset_id, source_path) VALUES (?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                source_path = CASE WHEN excluded.source_path != '' THEN excluded.source_path ELSE source_path END
            """,
            (dataset_id, str(source_path)),
        )
        self._registered_dataset_ids.add(dataset_id)
        if commit:
            self._conn.commit()

    def _get_or_add_node(self, name: str, kind: str) -> int:
        if kind not in {ENTITY, LITERAL}:
            raise ValueError(f"Unsupported node kind: {kind}")
        resolved = self.resolver.resolve(name)
        if not resolved.canonical_key:
            raise ValueError("Node name must be non-empty")

        row = self._find_node(resolved.canonical_key, canonical_entity_key(name))

        if row is None:
            cursor = self._conn.execute(
                "INSERT INTO nodes(canonical_key, canonical_name, kind) VALUES (?, ?, ?)",
                (resolved.canonical_key, resolved.canonical_name, kind),
            )
            node_id = int(cursor.lastrowid)
            if kind == ENTITY:
                self._invalidate_entity_vectors()
        else:
            node_id = int(row["node_id"])
            if row["kind"] == LITERAL and kind == ENTITY:
                # A node only becomes searchable once it appears as a subject.
                # If it was previously created as a literal/object-only node,
                # promote it and repair historical relation edges that point to it.
                self._promote_node_to_entity(node_id)

        alias_name = normalize_text(name)
        alias_key = canonical_entity_key(alias_name)
        if alias_key:
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases(alias_key, alias_name, node_id) VALUES (?, ?, ?)",
                (alias_key, alias_name, node_id),
            )
        return node_id

    def _promote_node_to_entity(self, node_id: int) -> None:
        row = self._conn.execute("SELECT kind FROM nodes WHERE node_id = ?", (int(node_id),)).fetchone()
        if row is None:
            raise KeyError(node_id)
        if str(row["kind"]) == ENTITY:
            return
        self._conn.execute("UPDATE nodes SET kind = ? WHERE node_id = ?", (ENTITY, int(node_id)))
        self._conn.execute(
            """
            UPDATE triples
            SET object_kind = ?
            WHERE object_id = ? AND triple_type = 'RELATION'
            """,
            (ENTITY, int(node_id)),
        )
        self._invalidate_entity_vectors()

    def _find_node(self, canonical_key: str, alias_key: str) -> sqlite3.Row | None:
        row = self._conn.execute(
            "SELECT node_id, kind FROM nodes WHERE canonical_key = ?", (canonical_key,)
        ).fetchone()
        if row is not None:
            return row
        return self._conn.execute(
            """
            SELECT n.node_id, n.kind FROM aliases a
            JOIN nodes n ON n.node_id = a.node_id
            WHERE a.alias_key = ? ORDER BY n.node_id LIMIT 1
            """,
            (alias_key,),
        ).fetchone()

    def _node_field(self, node_id: int, field: str) -> str:
        if field not in {"canonical_key", "canonical_name", "kind"}:
            raise ValueError(f"Unsupported node field: {field}")
        row = self._conn.execute(f"SELECT {field} FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        return str(row[field])

    def get_node_name(self, node_id: int) -> str:
        return self._node_field(node_id, "canonical_name")

    def resolve_node_id(self, name: str) -> int | None:
        key = self.resolver.resolve(name).canonical_key
        row = self._conn.execute(
            "SELECT node_id FROM nodes WHERE canonical_key = ?", (key,)
        ).fetchone()
        if row is None:
            row = self._conn.execute(
                "SELECT node_id FROM aliases WHERE alias_key = ? ORDER BY node_id LIMIT 1", (key,)
            ).fetchone()
        return None if row is None else int(row["node_id"])

    def add_triple(
        self,
        *,
        triple_type: str,
        subject: str,
        predicate: str,
        object_value: str,
        kv_offsets: Sequence[int],
        dataset_id: str,
        sample_id: str,
        source_index: int,
        triple_index: int,
        title: str = "",
        commit: bool = True,
    ) -> int:
        triple_type = normalize_text(triple_type).upper()
        subject = normalize_text(subject)
        predicate = normalize_text(predicate)
        object_value = normalize_text(object_value)
        if triple_type not in {"RELATION", "ATTRIBUTE"}:
            raise ValueError(f"Unsupported triple type: {triple_type}")
        if not subject or not predicate or not object_value:
            raise ValueError("subject, predicate, and object_value must be non-empty")

        self.register_dataset(dataset_id, commit=False)
        subject_id = self._get_or_add_node(subject, ENTITY)
        object_kind = LITERAL
        object_id = self._get_or_add_node(object_value, object_kind)
        if triple_type == "RELATION":
            object_row = self._conn.execute(
                "SELECT kind FROM nodes WHERE node_id = ?",
                (int(object_id),),
            ).fetchone()
            if object_row is None:
                raise KeyError(object_id)
            object_kind = str(object_row["kind"])
        triple_key = content_hash(
            triple_type,
            self._node_field(subject_id, "canonical_key"),
            canonical_relation(predicate),
            self._node_field(object_id, "canonical_key"),
        )
        triple_id = self._get_or_add_triple(
            triple_key,
            triple_type,
            subject_id,
            predicate,
            object_id,
            object_kind,
        )

        for ordinal, offset in enumerate(kv_offsets):
            self._conn.execute(
                "INSERT OR IGNORE INTO triple_kvs(triple_id, kv_offset, ordinal) VALUES (?, ?, ?)",
                (triple_id, int(offset), ordinal),
            )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO triple_sources(
                triple_id, dataset_id, sample_id, source_index, triple_index, title
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (triple_id, str(dataset_id), str(sample_id), int(source_index), int(triple_index), normalize_text(title)),
        )
        if commit:
            self._conn.commit()
        return triple_id

    def _get_or_add_triple(
        self,
        content_key: str,
        triple_type: str,
        subject_id: int,
        predicate: str,
        object_id: int,
        object_kind: str,
    ) -> int:
        row = self._conn.execute(
            "SELECT triple_id FROM triples WHERE content_key = ?", (content_key,)
        ).fetchone()
        if row is not None:
            return int(row["triple_id"])
        cursor = self._conn.execute(
            """
            INSERT INTO triples(content_key, triple_type, subject_id, predicate, object_id, object_kind)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (content_key, triple_type, subject_id, predicate, object_id, object_kind),
        )
        return int(cursor.lastrowid)

    def get_triple(self, triple_id: int) -> GraphTriple:
        row = self._conn.execute(
            """
            SELECT t.*, s.canonical_name AS subject_name, o.canonical_name AS object_name,
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
            JOIN nodes s ON s.node_id = t.subject_id
            JOIN nodes o ON o.node_id = t.object_id
            WHERE t.triple_id = ?
            """,
            (int(triple_id),),
        ).fetchone()
        if row is None:
            raise KeyError(triple_id)
        offsets = self._conn.execute(
            "SELECT kv_offset FROM triple_kvs WHERE triple_id = ? ORDER BY ordinal, kv_offset",
            (int(triple_id),),
        ).fetchall()
        return self._row_to_triple(row, offsets)

    @staticmethod
    def _row_to_triple(row: sqlite3.Row, offsets: Sequence[sqlite3.Row]) -> GraphTriple:
        return GraphTriple(
            triple_id=int(row["triple_id"]),
            triple_type=str(row["triple_type"]),
            subject_id=int(row["subject_id"]),
            subject=str(row["subject_name"]),
            predicate=str(row["predicate"]),
            object_id=int(row["object_id"]),
            object=str(row["object_name"]),
            object_kind=str(row["object_kind"]),
            kv_offsets=tuple(int(item["kv_offset"]) for item in offsets),
            title=str(row["source_title"]),
        )

    def incident_triples(self, node_id: int) -> list[GraphTriple]:
        rows = self._conn.execute(
            "SELECT triple_id FROM triples WHERE subject_id = ? OR object_id = ? ORDER BY triple_id",
            (int(node_id), int(node_id)),
        ).fetchall()
        return [self.get_triple(int(row["triple_id"])) for row in rows]

    def get_local_subgraph(
        self,
        seed_node_ids: Sequence[int],
        *,
        hops: int = 1,
        max_triples: int | None = None,
    ) -> tuple[list[int], list[GraphTriple]]:
        if hops < 0:
            raise ValueError("hops must be non-negative")
        visited = {int(node_id) for node_id in seed_node_ids}
        queue = deque((int(node_id), 0) for node_id in seed_node_ids)
        triples: dict[int, GraphTriple] = {}

        while queue:
            node_id, depth = queue.popleft()
            if depth >= hops:
                continue
            for triple in self.incident_triples(node_id):
                triples.setdefault(triple.triple_id, triple)
                if max_triples is not None and len(triples) >= max_triples:
                    return sorted(visited), list(triples.values())
                for neighbor_id in (triple.subject_id, triple.object_id):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    # Literal attribute values are returned in the subgraph but
                    # never expanded as entity-centric graph frontiers.
                    if self._node_field(neighbor_id, "kind") == ENTITY:
                        queue.append((neighbor_id, depth + 1))
        return sorted(visited), list(triples.values())

    def stats(self) -> dict[str, int]:
        def count(table: str) -> int:
            return int(self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])

        return {
            "datasets": count("datasets"),
            "nodes": count("nodes"),
            "entities": int(
                self._conn.execute("SELECT COUNT(*) AS n FROM nodes WHERE kind = ?", (ENTITY,)).fetchone()["n"]
            ),
            "literals": int(
                self._conn.execute("SELECT COUNT(*) AS n FROM nodes WHERE kind = ?", (LITERAL,)).fetchone()["n"]
            ),
            "triples": count("triples"),
            "triple_sources": count("triple_sources"),
            "triple_kvs": count("triple_kvs"),
        }

    def entity_nodes(self) -> list[tuple[int, str]]:
        rows = self._conn.execute(
            "SELECT node_id, canonical_name FROM nodes WHERE kind = ? ORDER BY node_id", (ENTITY,)
        ).fetchall()
        return [(int(row["node_id"]), str(row["canonical_name"])) for row in rows]

    def write_entity_embeddings(
        self,
        node_ids: Sequence[int],
        embeddings: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != ids.shape[0]:
            raise ValueError("embeddings must be a 2-D array aligned with node_ids")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("node_ids must be unique")
        valid_ids = {node_id for node_id, _ in self.entity_nodes()}
        unknown = set(ids.tolist()) - valid_ids
        if unknown:
            raise ValueError(f"Embeddings include non-entity or unknown node ids: {sorted(unknown)[:5]}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        np.save(self.entity_ids_path, ids)
        np.save(self.entity_vectors_path, vectors)
        write_json_atomic(
            self.entity_vector_meta_path,
            {
                **(metadata or {}),
                "rows": int(vectors.shape[0]),
                "dimension": int(vectors.shape[1]),
                "dtype": str(vectors.dtype),
            },
        )
        self._invalidate_hnsw()

    def _invalidate_entity_vectors(self) -> None:
        unlink_existing(self.entity_ids_path, self.entity_vectors_path, self.entity_vector_meta_path)
        self._invalidate_hnsw()

    def _invalidate_hnsw(self) -> None:
        self._hnsw_index = None
        self._hnsw_meta = None
        unlink_existing(self.hnsw_index_path, self.hnsw_meta_path)

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

    def _load_entity_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.entity_ids_path.exists() or not self.entity_vectors_path.exists():
            raise RuntimeError("Entity embeddings have not been written")
        ids = np.load(self.entity_ids_path, mmap_mode="r")
        vectors = np.load(self.entity_vectors_path, mmap_mode="r")
        if ids.shape[0] != vectors.shape[0]:
            raise RuntimeError("Entity id/vector row mismatch")
        return ids, vectors

    def _search_exact(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        ids, vectors = self._load_entity_vectors()
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

    def build_hnsw_index(self, *, space: str = "cosine", ef_construction: int = 200, m: int = 16) -> None:
        hnswlib = self._require_hnswlib()
        ids, vectors = self._load_entity_vectors()
        index = hnswlib.Index(space=space, dim=int(vectors.shape[1]))
        index.init_index(max_elements=int(ids.shape[0]), ef_construction=ef_construction, M=m)
        index.add_items(np.asarray(vectors), np.asarray(ids))
        index.set_ef(max(50, min(200, int(ids.shape[0]))))
        index.save_index(str(self.hnsw_index_path))
        metadata = {"space": space, "dim": int(vectors.shape[1]), "count": int(ids.shape[0])}
        write_json_atomic(
            self.hnsw_meta_path,
            metadata,
        )
        self._hnsw_index = index
        self._hnsw_meta = metadata

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
