"""Segmented offset-addressed KV tensor store for Store V2."""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from kblam.stores.kv_store import KVRecord
from kblam.stores.io import open_sqlite, write_json_atomic


@dataclass(frozen=True)
class TensorSegment:
    segment_id: int
    offset_begin: int
    rows: int
    key_path: str
    value_path: str

    @property
    def offset_end(self) -> int:
        return self.offset_begin + self.rows


class KVStoreV2:
    """Store KV text in SQLite and base embeddings in append-only segments."""

    DB_NAME = "kv_store.sqlite3"
    TENSOR_MANIFEST_NAME = "tensor_manifest.json"

    def __init__(self, root: str | Path, create: bool = True) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            raise FileNotFoundError(self.root)
        self.db_path = self.root / self.DB_NAME
        self.key_root = self.root / "key"
        self.value_root = self.root / "value"
        self.manifest_path = self.root / self.TENSOR_MANIFEST_NAME
        if create:
            self.key_root.mkdir(parents=True, exist_ok=True)
            self.value_root.mkdir(parents=True, exist_ok=True)
        self._conn = open_sqlite(self.db_path)
        self._segments: list[TensorSegment] | None = None
        self._manifest: dict[str, Any] | None = None
        self._segment_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def __enter__(self) -> "KVStoreV2":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()
        self._segments = None
        self._manifest = None
        self._segment_cache.clear()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM kv_records").fetchone()
        return int(row["n"])

    @property
    def tensor_count(self) -> int:
        return int(self._load_manifest().get("rows", 0))

    def get(self, offset: int) -> KVRecord:
        row = self._conn.execute(
            "SELECT offset, key_text, value_text, metadata_json FROM kv_records WHERE offset = ?",
            (int(offset),),
        ).fetchone()
        if row is None:
            raise KeyError(offset)
        return KVRecord(
            offset=int(row["offset"]),
            key_text=str(row["key_text"]),
            value_text=str(row["value_text"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def get_many(self, offsets: Iterable[int]) -> list[KVRecord]:
        return [self.get(offset) for offset in offsets]

    def iter_records(self, *, start_offset: int = 0) -> Iterable[KVRecord]:
        rows = self._conn.execute(
            """
            SELECT offset, key_text, value_text, metadata_json
            FROM kv_records WHERE offset >= ? ORDER BY offset
            """,
            (int(start_offset),),
        )
        for row in rows:
            yield KVRecord(
                offset=int(row["offset"]),
                key_text=str(row["key_text"]),
                value_text=str(row["value_text"]),
                metadata=json.loads(row["metadata_json"]),
            )

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
            raise ValueError(f"Tensor rows would end at {end}, but store has only {len(self)} records")

        manifest = self._load_manifest()
        if manifest["rows"] and tuple(manifest["key_shape"][1:]) != tuple(new_keys.shape[1:]):
            raise ValueError("Appended key tensor shape must match existing segments")
        if manifest["rows"] and tuple(manifest["value_shape"][1:]) != tuple(new_values.shape[1:]):
            raise ValueError("Appended value tensor shape must match existing segments")
        if manifest["rows"] and manifest["key_dtype"] != str(new_keys.dtype):
            raise ValueError("Appended key tensor dtype must match existing segments")
        if manifest["rows"] and manifest["value_dtype"] != str(new_values.dtype):
            raise ValueError("Appended value tensor dtype must match existing segments")

        segment_id = len(self._segments_or_empty())
        key_path = self.key_root / f"seg_{segment_id:06d}.npy"
        value_path = self.value_root / f"seg_{segment_id:06d}.npy"
        np.save(key_path, new_keys)
        np.save(value_path, new_values)

        segments = self._segments_or_empty()
        segments.append(
            TensorSegment(
                segment_id=segment_id,
                offset_begin=start,
                rows=int(new_keys.shape[0]),
                key_path=str(key_path.relative_to(self.root)),
                value_path=str(value_path.relative_to(self.root)),
            )
        )
        payload = {
            **manifest,
            **(metadata or {}),
            "rows": end,
            "key_shape": [end, *new_keys.shape[1:]],
            "value_shape": [end, *new_values.shape[1:]],
            "key_dtype": str(new_keys.dtype),
            "value_dtype": str(new_values.dtype),
            "segments": [segment.__dict__ for segment in segments],
        }
        write_json_atomic(self.manifest_path, payload)
        self._segments = segments
        self._manifest = payload
        return start, end

    def get_tensors(self, offsets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        if self.tensor_count == 0:
            raise RuntimeError("No tensor segments have been attached to this KVStoreV2")
        index = np.asarray(offsets, dtype=np.int64)
        if index.size == 0:
            manifest = self._load_manifest()
            key_shape = tuple(manifest["key_shape"][1:])
            value_shape = tuple(manifest["value_shape"][1:])
            key_dtype = np.dtype(manifest["key_dtype"])
            value_dtype = np.dtype(manifest["value_dtype"])
            return (
                np.zeros((0, *key_shape), dtype=key_dtype),
                np.zeros((0, *value_shape), dtype=value_dtype),
            )
        if index.min() < 0 or index.max() >= self.tensor_count:
            raise IndexError("KV tensor offset out of range")

        manifest = self._load_manifest()
        keys_out = np.empty((index.shape[0], *manifest["key_shape"][1:]), dtype=np.dtype(manifest["key_dtype"]))
        values_out = np.empty(
            (index.shape[0], *manifest["value_shape"][1:]),
            dtype=np.dtype(manifest["value_dtype"]),
        )
        buckets: dict[int, list[tuple[int, int]]] = {}
        segments = self._segments_or_empty()
        for out_pos, offset in enumerate(index.tolist()):
            segment_idx = self._find_segment_index(int(offset), segments)
            buckets.setdefault(segment_idx, []).append((out_pos, int(offset)))

        for segment_idx, items in buckets.items():
            segment = segments[segment_idx]
            keys, values = self._load_segment(segment)
            local_index = np.asarray([offset - segment.offset_begin for _, offset in items], dtype=np.int64)
            out_index = np.asarray([out_pos for out_pos, _ in items], dtype=np.int64)
            keys_out[out_index] = np.asarray(keys[local_index])
            values_out[out_index] = np.asarray(values[local_index])
        return keys_out, values_out

    @classmethod
    def export_from_v1(
        cls,
        root: str | Path,
        source_store,
        *,
        segment_rows: int = 131072,
    ) -> "KVStoreV2":
        dst = cls(root, create=True)
        dst._conn.close()
        shutil.copy2(source_store.db_path, dst.db_path)
        dst._conn = open_sqlite(dst.db_path)
        if source_store.tensor_count:
            keys, values = source_store._load_tensor_pair()
            rows = int(keys.shape[0])
            for start in range(0, rows, segment_rows):
                end = min(rows, start + segment_rows)
                dst.append_tensors(
                    np.asarray(keys[start:end]),
                    np.asarray(values[start:end]),
                    metadata={"exported_from_v1": str(source_store.root)},
                )
        return dst

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            if not self.manifest_path.exists():
                self._manifest = {
                    "rows": 0,
                    "key_shape": [0],
                    "value_shape": [0],
                    "key_dtype": "",
                    "value_dtype": "",
                    "segments": [],
                }
            else:
                self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    def _segments_or_empty(self) -> list[TensorSegment]:
        if self._segments is None:
            self._segments = [TensorSegment(**segment) for segment in self._load_manifest().get("segments", [])]
        return list(self._segments)

    def _load_segment(self, segment: TensorSegment) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (segment.key_path, segment.value_path)
        if cache_key not in self._segment_cache:
            self._segment_cache[cache_key] = (
                np.load(self.root / segment.key_path, mmap_mode="r"),
                np.load(self.root / segment.value_path, mmap_mode="r"),
            )
        return self._segment_cache[cache_key]

    @staticmethod
    def _find_segment_index(offset: int, segments: Sequence[TensorSegment]) -> int:
        left, right = 0, len(segments) - 1
        while left <= right:
            mid = (left + right) // 2
            segment = segments[mid]
            if offset < segment.offset_begin:
                right = mid - 1
            elif offset >= segment.offset_end:
                left = mid + 1
            else:
                return mid
        raise IndexError(f"No KV tensor segment covers offset {offset}")

    @staticmethod
    def _validate_tensor_pair(keys: np.ndarray, values: np.ndarray) -> None:
        if keys.ndim < 2 or values.ndim < 2:
            raise ValueError("KV tensor arrays must have a row dimension and at least one feature dimension")
        if keys.shape[0] != values.shape[0]:
            raise ValueError(f"KV tensor row mismatch: key={keys.shape[0]}, value={values.shape[0]}")
