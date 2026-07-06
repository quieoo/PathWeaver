#!/usr/bin/env python3
"""Build reproducible cumulative PathWeaver Store tiers by copy-and-append.

The base Store is copied into the first output tier.  Every subsequent tier is
copied from its predecessor and extended with one dataset.  Existing KV and
entity embeddings are reused; only newly allocated KV offsets and newly added
entities are encoded.  HNSW is rebuilt after each append because hnswlib's
persisted index has no spare capacity in the current GraphStore format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kblam.stores import GraphStore, KVStore
try:
    from tools.build_pathweaver_stores import (
        _encode_texts,
        _load_kv_model,
        _model_metadata,
        _safe_encode_texts,
        ingest_dataset,
    )
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from build_pathweaver_stores import (  # type: ignore[no-redef]
        _encode_texts,
        _load_kv_model,
        _model_metadata,
        _safe_encode_texts,
        ingest_dataset,
    )


@dataclass(frozen=True)
class AppendTier:
    label: str
    dataset_id: str
    dataset_path: Path
    sample_start: int = 0
    sample_limit: int | None = None


def parse_append_tier(value: str) -> AppendTier:
    parts = value.split("::")
    if len(parts) not in {3, 5} or not all(parts[:3]):
        raise argparse.ArgumentTypeError(
            "--append-tier must be LABEL::DATASET_ID::DATASET_PATH or "
            "LABEL::DATASET_ID::DATASET_PATH::SAMPLE_START::SAMPLE_LIMIT"
        )
    try:
        sample_start = int(parts[3]) if len(parts) == 5 else 0
        sample_limit = int(parts[4]) if len(parts) == 5 else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample start/limit must be integers") from exc
    if sample_start < 0 or (sample_limit is not None and sample_limit <= 0):
        raise argparse.ArgumentTypeError("sample start must be >= 0 and limit must be > 0")
    tier = AppendTier(
        parts[0],
        parts[1],
        Path(parts[2]).expanduser(),
        sample_start,
        sample_limit,
    )
    if not tier.dataset_path.is_file():
        raise argparse.ArgumentTypeError(f"dataset does not exist: {tier.dataset_path}")
    return tier


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_store(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing tier {destination}; choose a new output root"
        )
    destination.mkdir(parents=True)
    for component in ("graph", "kv"):
        source_component = source / component
        if not source_component.is_dir():
            raise FileNotFoundError(source_component)
        shutil.copytree(source_component, destination / component)


def store_counts(store_dir: Path) -> dict[str, Any]:
    with KVStore(store_dir / "kv", create=False) as kv_store, GraphStore(
        store_dir / "graph", create=False
    ) as graph_store:
        return {"kv_records": len(kv_store), "graph": graph_store.stats()}


def extend_store(
    store_dir: Path,
    tier: AppendTier,
    *,
    entity_model: Any,
    entity_model_path: str,
    entity_batch_size: int,
    entity_prompt_name: str | None,
    kv_model: Any,
    kv_model_path: str,
    kv_batch_size: int,
    kv_prompt_name: str | None,
    kv_encoding_profile: str,
    kv_device: str,
    hnsw_ef_construction: int,
    hnsw_m: int,
    ingest_commit_interval: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    graph_dir = store_dir / "graph"
    kv_dir = store_dir / "kv"

    with KVStore(kv_dir, create=False) as kv_store, GraphStore(
        graph_dir, create=False
    ) as graph_store:
        old_kv_count = len(kv_store)
        if kv_store.tensor_count != old_kv_count:
            raise RuntimeError(
                f"Base tier has {old_kv_count} KV rows but {kv_store.tensor_count} tensor rows"
            )

        old_entity_ids, old_entity_vectors = graph_store._load_entity_vectors()
        old_entity_ids = np.asarray(old_entity_ids, dtype=np.int64).copy()
        old_entity_vectors = np.asarray(old_entity_vectors, dtype=np.float32).copy()
        old_entity_vectors_by_id = {
            int(node_id): vector
            for node_id, vector in zip(old_entity_ids.tolist(), old_entity_vectors)
        }

        ingest_started = time.perf_counter()
        ingest = ingest_dataset(
            tier.dataset_path,
            tier.dataset_id,
            kv_store,
            graph_store,
            sample_start=tier.sample_start,
            sample_limit=tier.sample_limit,
            commit_interval=ingest_commit_interval,
        )
        ingest_seconds = time.perf_counter() - ingest_started

        new_kv_count = len(kv_store)
        new_kv_rows = new_kv_count - old_kv_count
        new_records = list(kv_store.iter_records(start_offset=old_kv_count))
        if len(new_records) != new_kv_rows:
            raise RuntimeError("New KV offsets are not a contiguous suffix")

        kv_encode_started = time.perf_counter()
        if new_records:
            key_texts = [record.key_text for record in new_records]
            value_texts = [record.value_text for record in new_records]
            vectors = _safe_encode_texts(
                kv_model,
                key_texts + value_texts,
                kv_batch_size,
                kv_prompt_name,
            )
            kv_store.append_tensors(
                vectors[:new_kv_rows],
                vectors[new_kv_rows:],
                metadata={
                    **_model_metadata("kv_base_embedding", kv_model_path, kv_prompt_name),
                    "encoding_profile": kv_encoding_profile,
                    "padding_side": (
                        "left" if kv_encoding_profile == "qwen3-embedding-v2" else "model_default"
                    ),
                    "model_dtype": (
                        "float16_first_module"
                        if kv_encoding_profile == "qwen3-embedding-v2"
                        else "model_default"
                    ),
                    "device": kv_device,
                },
            )
        kv_encode_seconds = time.perf_counter() - kv_encode_started

        entities = graph_store.entity_nodes()
        new_entities = [item for item in entities if item[0] not in old_entity_vectors_by_id]
        entity_encode_started = time.perf_counter()
        if new_entities:
            new_entity_vectors = _encode_texts(
                entity_model,
                [name for _, name in new_entities],
                entity_batch_size,
                entity_prompt_name,
            )
        else:
            new_entity_vectors = np.empty(
                (0, old_entity_vectors.shape[1]), dtype=np.float32
            )
        for (node_id, _), vector in zip(new_entities, new_entity_vectors):
            old_entity_vectors_by_id[node_id] = vector

        ordered_ids = [node_id for node_id, _ in entities]
        ordered_vectors = np.asarray(
            [old_entity_vectors_by_id[node_id] for node_id in ordered_ids],
            dtype=np.float32,
        )
        graph_store.write_entity_embeddings(
            ordered_ids,
            ordered_vectors,
            metadata=_model_metadata(
                "hnsw_entity_embedding", entity_model_path, entity_prompt_name
            ),
        )
        entity_encode_seconds = time.perf_counter() - entity_encode_started

        hnsw_started = time.perf_counter()
        graph_store.build_hnsw_index(
            ef_construction=hnsw_ef_construction,
            m=hnsw_m,
        )
        hnsw_seconds = time.perf_counter() - hnsw_started

        result = {
            "label": tier.label,
            "dataset_id": tier.dataset_id,
            "dataset_path": str(tier.dataset_path.resolve()),
            "sample_start": tier.sample_start,
            "sample_limit": tier.sample_limit,
            "dataset_bytes": tier.dataset_path.stat().st_size,
            "dataset_sha256": file_sha256(tier.dataset_path),
            "ingest": ingest,
            "new_kv_records": new_kv_rows,
            "new_entities": len(new_entities),
            "counts": {"kv_records": len(kv_store), "graph": graph_store.stats()},
            "timing_seconds": {
                "ingest": ingest_seconds,
                "new_kv_embedding_and_append": kv_encode_seconds,
                "new_entity_embedding_and_write": entity_encode_seconds,
                "hnsw_rebuild": hnsw_seconds,
                "total": time.perf_counter() - started,
            },
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-store", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-label", default="000237-2wiki")
    parser.add_argument(
        "--append-tier",
        action="append",
        type=parse_append_tier,
        default=[],
        metavar="LABEL::DATASET_ID::PATH[::START::LIMIT]",
    )
    parser.add_argument("--hnsw-embedding-model", required=True)
    parser.add_argument("--kv-embedding-model", required=True)
    parser.add_argument("--hnsw-embedding-batch-size", type=int, default=1024)
    parser.add_argument("--kv-embedding-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-prompt-name", default=None)
    parser.add_argument("--kv-prompt-name", default=None)
    parser.add_argument(
        "--kv-encoding-profile",
        choices=["qwen3-embedding-v2", "sentence-transformer"],
        default="qwen3-embedding-v2",
    )
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument(
        "--ingest-commit-interval",
        type=int,
        default=100,
        help="Commit SQLite writes every N appended samples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.append_tier:
        raise ValueError("At least one --append-tier is required")
    if not args.base_store.is_dir():
        raise FileNotFoundError(args.base_store)
    args.output_root.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    entity_model = SentenceTransformer(args.hnsw_embedding_model)
    kv_model, effective_kv_prompt, kv_device = _load_kv_model(
        args.kv_embedding_model,
        args.kv_encoding_profile,
        args.kv_prompt_name,
    )

    base_destination = args.output_root / args.base_label
    copy_store(args.base_store, base_destination)
    manifest: dict[str, Any] = {
        "base_store": str(args.base_store.resolve()),
        "base_label": args.base_label,
        "models": {
            "hnsw_embedding_model": str(Path(args.hnsw_embedding_model).resolve()),
            "hnsw_prompt_name": args.hnsw_prompt_name,
            "kv_embedding_model": str(Path(args.kv_embedding_model).resolve()),
            "kv_prompt_name": effective_kv_prompt,
            "kv_encoding_profile": args.kv_encoding_profile,
        },
        "hnsw": {"ef_construction": args.hnsw_ef_construction, "m": args.hnsw_m},
        "tiers": [
            {
                "label": args.base_label,
                "store_dir": str(base_destination.resolve()),
                "copied_from": str(args.base_store.resolve()),
                "counts": store_counts(base_destination),
            }
        ],
    }

    previous = base_destination
    for tier in args.append_tier:
        destination = args.output_root / tier.label
        print(f"[COPY] {previous} -> {destination}", flush=True)
        copy_store(previous, destination)
        print(f"[APPEND] {tier.dataset_path}", flush=True)
        result = extend_store(
            destination,
            tier,
            entity_model=entity_model,
            entity_model_path=args.hnsw_embedding_model,
            entity_batch_size=args.hnsw_embedding_batch_size,
            entity_prompt_name=args.hnsw_prompt_name,
            kv_model=kv_model,
            kv_model_path=args.kv_embedding_model,
            kv_batch_size=args.kv_embedding_batch_size,
            kv_prompt_name=effective_kv_prompt,
            kv_encoding_profile=args.kv_encoding_profile,
            kv_device=kv_device,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_m=args.hnsw_m,
            ingest_commit_interval=args.ingest_commit_interval,
        )
        result["store_dir"] = str(destination.resolve())
        result["copied_from"] = str(previous.resolve())
        manifest["tiers"].append(result)
        (destination / "scale_tier_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        previous = destination

    manifest_path = args.output_root / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
