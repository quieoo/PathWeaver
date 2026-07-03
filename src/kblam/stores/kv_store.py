"""Offset-addressed key/value store with optional dense tensor arrays."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from kblam.stores.common import content_hash, normalize_text
from kblam.stores.io import open_sqlite, write_json_atomic


@dataclass(frozen=True)
class KVRecord:
    offset: int
    key_text: str
    value_text: str
    metadata: dict[str, Any]


class KVStore:
    """Store KV text records under stable, zero-based offsets.

    Text and provenance are kept in SQLite. Dense base embeddings or final KV
    tensors can be attached later as two NumPy arrays whose first dimension is
    exactly aligned with the record offsets.
    """

    DB_NAME = "kv_store.sqlite3"
    KEY_ARRAY_NAME = "key_tensors.npy"
    VALUE_ARRAY_NAME = "value_tensors.npy"
    TENSOR_META_NAME = "tensor_metadata.json"

    def __init__(self, root: str | Path, create: bool = True) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            raise FileNotFoundError(self.root)

        self.db_path = self.root / self.DB_NAME
        self.key_array_path = self.root / self.KEY_ARRAY_NAME
        self.value_array_path = self.root / self.VALUE_ARRAY_NAME
        self.tensor_meta_path = self.root / self.TENSOR_META_NAME
        self._conn = open_sqlite(self.db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv_records (
                offset INTEGER PRIMARY KEY,
                content_key TEXT NOT NULL UNIQUE,
                key_text TEXT NOT NULL,
                value_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS kv_sources (
                offset INTEGER NOT NULL REFERENCES kv_records(offset) ON DELETE CASCADE,
                dataset_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                triple_index INTEGER NOT NULL,
                kv_index INTEGER NOT NULL,
                PRIMARY KEY (offset, dataset_id, sample_id, triple_index, kv_index)
            );
            """
        )
        self._conn.commit()

    def __enter__(self) -> "KVStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM kv_records").fetchone()
        return int(row["n"])

    def add(
        self,
        key_text: str,
        value_text: str,
        *,
        metadata: dict[str, Any] | None = None,
        dataset_id: str | None = None,
        sample_id: str | None = None,
        triple_index: int | None = None,
        kv_index: int | None = None,
        dedupe_key: str | None = None,
    ) -> int:
        key_text = normalize_text(key_text)
        value_text = normalize_text(value_text)
        if not key_text or not value_text:
            raise ValueError("key_text and value_text must be non-empty")

        record_key = dedupe_key or content_hash(key_text.casefold(), value_text.casefold())
        row = self._conn.execute(
            "SELECT offset FROM kv_records WHERE content_key = ?", (record_key,)
        ).fetchone()
        if row is None:
            next_offset = len(self)
            self._conn.execute(
                """
                INSERT INTO kv_records(offset, content_key, key_text, value_text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    next_offset,
                    record_key,
                    key_text,
                    value_text,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
            offset = next_offset
        else:
            offset = int(row["offset"])

        source = self._normalize_source(dataset_id, sample_id, triple_index, kv_index)
        if source is not None:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO kv_sources(offset, dataset_id, sample_id, triple_index, kv_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                (offset, *source),
            )
        self._conn.commit()
        return offset

    def get(self, offset: int) -> KVRecord:
        row = self._conn.execute(
            "SELECT offset, key_text, value_text, metadata_json FROM kv_records WHERE offset = ?",
            (int(offset),),
        ).fetchone()
        if row is None:
            raise KeyError(offset)
        return self._record_from_row(row)

    def get_many(self, offsets: Iterable[int]) -> list[KVRecord]:
        return [self.get(offset) for offset in offsets]

    def iter_records(self) -> Iterable[KVRecord]:
        rows = self._conn.execute(
            "SELECT offset, key_text, value_text, metadata_json FROM kv_records ORDER BY offset"
        )
        yield from map(self._record_from_row, rows)

    @property
    def tensor_count(self) -> int:
        if not self.key_array_path.exists() and not self.value_array_path.exists():
            return 0
        keys, values = self._load_tensor_pair()
        return int(keys.shape[0])

    def write_tensors(
        self,
        key_tensors: np.ndarray,
        value_tensors: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        keys = np.asarray(key_tensors)
        values = np.asarray(value_tensors)
        self._validate_tensor_pair(keys, values)
        if keys.shape[0] != len(self):
            raise ValueError(f"Expected {len(self)} tensor rows, got {keys.shape[0]}")
        self._save_tensor_pair(keys, values)
        self._write_tensor_metadata(keys, values, metadata)

    def append_tensors(
        self,
        key_tensors: np.ndarray,
        value_tensors: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        new_keys = np.asarray(key_tensors)
        new_values = np.asarray(value_tensors)
        self._validate_tensor_pair(new_keys, new_values)
        start = self.tensor_count
        end = start + int(new_keys.shape[0])
        if end > len(self):
            raise ValueError(f"Tensor rows would end at {end}, but the store has only {len(self)} records")
        if start == 0:
            self._save_tensor_pair(new_keys, new_values)
            self._write_tensor_metadata(new_keys, new_values, metadata)
            return start, end

        old_keys, old_values = self._load_tensor_pair()
        if old_keys.shape[1:] != new_keys.shape[1:] or old_values.shape[1:] != new_values.shape[1:]:
            raise ValueError("Appended tensor shapes must match the existing arrays")
        if old_keys.dtype != new_keys.dtype or old_values.dtype != new_values.dtype:
            raise ValueError("Appended tensor dtypes must match the existing arrays")

        self._atomic_concat(self.key_array_path, old_keys, new_keys)
        self._atomic_concat(self.value_array_path, old_values, new_values)
        merged_keys, merged_values = self._load_tensor_pair()
        self._write_tensor_metadata(merged_keys, merged_values, metadata)
        return start, end

    def get_tensors(self, offsets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        if self.tensor_count == 0:
            raise RuntimeError("No tensor arrays have been attached to this KVStore")
        index = np.asarray(offsets, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= self.tensor_count):
            raise IndexError("KV tensor offset out of range")
        keys, values = self._load_tensor_pair()
        return np.asarray(keys[index]), np.asarray(values[index])

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> KVRecord:
        return KVRecord(
            offset=int(row["offset"]),
            key_text=str(row["key_text"]),
            value_text=str(row["value_text"]),
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _normalize_source(
        dataset_id: str | None,
        sample_id: str | None,
        triple_index: int | None,
        kv_index: int | None,
    ) -> tuple[str, str, int, int] | None:
        values = (dataset_id, sample_id, triple_index, kv_index)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("dataset_id, sample_id, triple_index, and kv_index must be provided together")
        return str(dataset_id), str(sample_id), int(triple_index), int(kv_index)

    def _load_tensor_pair(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.key_array_path.exists() or not self.value_array_path.exists():
            raise RuntimeError("KV tensor store is incomplete: only one tensor array exists")
        keys = np.load(self.key_array_path, mmap_mode="r")
        values = np.load(self.value_array_path, mmap_mode="r")
        if keys.shape[0] != values.shape[0]:
            raise RuntimeError(f"KV tensor row mismatch: key={keys.shape[0]}, value={values.shape[0]}")
        return keys, values

    def _save_tensor_pair(self, keys: np.ndarray, values: np.ndarray) -> None:
        self._atomic_save(self.key_array_path, keys)
        self._atomic_save(self.value_array_path, values)

    def _write_tensor_metadata(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        metadata: dict[str, Any] | None,
    ) -> None:
        existing: dict[str, Any] = {}
        if self.tensor_meta_path.exists():
            existing = json.loads(self.tensor_meta_path.read_text(encoding="utf-8"))
        payload = {
            **existing,
            **(metadata or {}),
            "rows": int(keys.shape[0]),
            "key_shape": list(keys.shape),
            "value_shape": list(values.shape),
            "key_dtype": str(keys.dtype),
            "value_dtype": str(values.dtype),
        }
        write_json_atomic(self.tensor_meta_path, payload)

    @staticmethod
    def _validate_tensor_pair(keys: np.ndarray, values: np.ndarray) -> None:
        if keys.ndim < 2 or values.ndim < 2:
            raise ValueError("KV tensor arrays must have a row dimension and at least one feature dimension")
        if keys.shape[0] != values.shape[0]:
            raise ValueError(f"KV tensor row mismatch: key={keys.shape[0]}, value={values.shape[0]}")

    @staticmethod
    def _atomic_save(path: Path, array: np.ndarray) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("wb") as handle:
                np.save(handle, array)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _atomic_concat(path: Path, old: np.ndarray, new: np.ndarray) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            output = np.lib.format.open_memmap(
                tmp,
                mode="w+",
                dtype=old.dtype,
                shape=(old.shape[0] + new.shape[0], *old.shape[1:]),
            )
            output[: old.shape[0]] = old
            output[old.shape[0] :] = new
            output.flush()
            del output
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
