#!/usr/bin/env python3
"""Build or extend PathWeaver KVStore and GraphStore from complete triples."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kblam.stores import EntityResolver, GraphStore, KVStore
from kblam.stores.common import normalize_text


@dataclass(frozen=True)
class ParsedTriple:
    triple_type: str
    subject: str
    predicate: str
    object_value: str

    @classmethod
    def from_raw(cls, triple: dict[str, Any]) -> "ParsedTriple | None":
        parsed = cls(
            triple_type=normalize_text(triple.get("type")).upper(),
            subject=normalize_text(triple.get("name")),
            predicate=normalize_text(triple.get("description_type")),
            object_value=normalize_text(triple.get("description")),
        )
        if parsed.triple_type not in {"RELATION", "ATTRIBUTE"}:
            return None
        return parsed if parsed.subject and parsed.predicate and parsed.object_value else None

    def metadata(self) -> dict[str, str]:
        return {
            "triple_type": self.triple_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_value,
        }


@dataclass
class IngestStats:
    samples: int = 0
    triples_seen: int = 0
    triples_added: int = 0
    kv_pairs_seen: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def load_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("data")
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list or an object containing a data list")
        yield from payload
        return
    raise ValueError(f"Unsupported dataset suffix: {path.suffix}")


def iter_triples(sample: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    top_level = sample.get("triple_list") or []
    if top_level:
        for triple in top_level:
            if isinstance(triple, dict):
                yield normalize_text(triple.get("title")), triple
        return

    for paragraph in sample.get("context") or []:
        if not isinstance(paragraph, dict):
            continue
        title = normalize_text(paragraph.get("title"))
        for triple in paragraph.get("triple_list") or []:
            if isinstance(triple, dict):
                yield title, triple


def default_kv_pairs(triple: ParsedTriple) -> list[dict[str, str]]:
    pairs = [{"key_string": f"{triple.subject} {triple.predicate}", "value_string": triple.object_value}]
    if triple.triple_type == "RELATION":
        pairs.append(
            {
                "key_string": f"the entity that {triple.predicate} {triple.object_value} is",
                "value_string": triple.subject,
            }
        )
    return pairs


def load_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Alias file must be a JSON object mapping alias to canonical entity name")
    return {str(alias): str(canonical) for alias, canonical in payload.items()}


def ingest_dataset(
    dataset_path: Path,
    dataset_id: str,
    kv_store: KVStore,
    graph_store: GraphStore,
    *,
    sample_start: int = 0,
    sample_limit: int | None = None,
    commit_interval: int = 1,
) -> dict[str, int]:
    if sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("sample_limit must be positive when provided")
    if commit_interval <= 0:
        raise ValueError("commit_interval must be positive")
    graph_store.register_dataset(dataset_id, str(dataset_path.resolve()))
    stats = IngestStats()
    deferred_commit = commit_interval > 1

    try:
        for source_index, sample in enumerate(load_rows(dataset_path)):
            if source_index < sample_start:
                continue
            if sample_limit is not None and stats.samples >= sample_limit:
                break
            stats.samples += 1
            sample_id = str(sample.get("_id", sample.get("id", source_index)))
            for triple_index, (title, raw_triple) in enumerate(iter_triples(sample)):
                triple = ParsedTriple.from_raw(raw_triple)
                if triple is None:
                    continue
                stats.triples_seen += 1
                kv_offsets = _store_kv_pairs(
                    kv_store,
                    raw_triple.get("kv_lists") or default_kv_pairs(triple),
                    triple,
                    dataset_id,
                    sample_id,
                    triple_index,
                    stats,
                    commit=not deferred_commit,
                )
                if not kv_offsets:
                    continue
                graph_store.add_triple(
                    triple_type=triple.triple_type,
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object_value=triple.object_value,
                    kv_offsets=kv_offsets,
                    dataset_id=dataset_id,
                    sample_id=sample_id,
                    source_index=source_index,
                    triple_index=triple_index,
                    title=title,
                    commit=not deferred_commit,
                )
                stats.triples_added += 1
            if deferred_commit and stats.samples % commit_interval == 0:
                kv_store.commit()
                graph_store.commit()
        if deferred_commit:
            kv_store.commit()
            graph_store.commit()
    except BaseException:
        if deferred_commit:
            kv_store.rollback()
            graph_store.rollback()
        raise
    return stats.as_dict()


def _store_kv_pairs(
    kv_store: KVStore,
    kv_pairs: Iterable[dict[str, Any]],
    triple: ParsedTriple,
    dataset_id: str,
    sample_id: str,
    triple_index: int,
    stats: IngestStats,
    *,
    commit: bool = True,
) -> list[int]:
    offsets = []
    for kv_index, kv in enumerate(kv_pairs):
        key_text = normalize_text(kv.get("key_string"))
        value_text = normalize_text(kv.get("value_string"))
        if not key_text or not value_text:
            continue
        stats.kv_pairs_seen += 1
        offsets.append(
            kv_store.add(
                key_text,
                value_text,
                metadata=triple.metadata(),
                dataset_id=dataset_id,
                sample_id=sample_id,
                triple_index=triple_index,
                kv_index=kv_index,
                commit=commit,
            )
        )
    return offsets


def _encode_texts(model, texts: list[str], batch_size: int, prompt_name: str | None) -> np.ndarray:
    encode_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    return np.asarray(model.encode(texts, **encode_kwargs), dtype=np.float32)


def _safe_encode_texts(model, texts: list[str], batch_size: int, prompt_name: str | None) -> np.ndarray:
    import torch

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings: list[np.ndarray] = []
    offset = 0
    minimum_batch_size = min(8, batch_size)
    while offset < len(texts):
        current_batch_size = batch_size
        while current_batch_size >= minimum_batch_size:
            batch = texts[offset : offset + current_batch_size]
            try:
                embeddings.append(_encode_texts(model, batch, batch_size, prompt_name))
                offset += len(batch)
                torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                current_batch_size //= 2
                torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"KV embedding OOM even at batch size {minimum_batch_size}")
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def build_entity_embeddings(
    graph_store: GraphStore,
    model_path: str,
    batch_size: int,
    prompt_name: str | None = None,
) -> None:
    from sentence_transformers import SentenceTransformer

    entities = graph_store.entity_nodes()
    node_ids = [node_id for node_id, _ in entities]
    names = [name for _, name in entities]
    model = SentenceTransformer(model_path)
    vectors = _encode_texts(model, names, batch_size, prompt_name)
    graph_store.write_entity_embeddings(
        node_ids,
        vectors,
        metadata=_model_metadata("hnsw_entity_embedding", model_path, prompt_name),
    )


def _model_metadata(role: str, model_path: str, prompt_name: str | None) -> dict[str, Any]:
    return {
        "role": role,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "prompt_name": prompt_name,
    }


def _load_kv_model(model_path: str, profile: str, prompt_name: str | None):
    from sentence_transformers import SentenceTransformer
    import torch

    if profile == "sentence-transformer":
        model = SentenceTransformer(model_path)
        return model, prompt_name, str(getattr(model, "device", "auto"))
    if profile != "qwen3-embedding-v2":
        raise ValueError(f"Unsupported KV encoding profile: {profile}")

    # Match the successful SentenceTransformer path in embedding_v2.py. These
    # details materially affect last-token pooled Qwen embeddings.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = SentenceTransformer(
        model_path,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    first_module = model._first_module() if hasattr(model, "_first_module") else None
    if first_module is not None and hasattr(first_module, "half"):
        first_module.half()
    return model, prompt_name or "query", device


def build_kv_embeddings(
    kv_store: KVStore,
    model_path: str,
    batch_size: int,
    prompt_name: str | None = None,
    encoding_profile: str = "qwen3-embedding-v2",
) -> None:
    records = list(kv_store.iter_records())
    if not records:
        raise ValueError("Cannot build KV embeddings for an empty KVStore")

    model, effective_prompt_name, device = _load_kv_model(model_path, encoding_profile, prompt_name)

    key_texts = [record.key_text for record in records]
    value_texts = [record.value_text for record in records]
    concatenated_vectors = _safe_encode_texts(
        model,
        key_texts + value_texts,
        batch_size,
        effective_prompt_name,
    )
    key_vectors = concatenated_vectors[: len(key_texts)]
    value_vectors = concatenated_vectors[len(key_texts) :]
    kv_store.write_tensors(
        key_vectors,
        value_vectors,
        metadata={
            **_model_metadata("kv_base_embedding", model_path, effective_prompt_name),
            "encoding_profile": encoding_profile,
            "padding_side": "left" if encoding_profile == "qwen3-embedding-v2" else "model_default",
            "model_dtype": "float16_first_module" if encoding_profile == "qwen3-embedding-v2" else "model_default",
            "device": device,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Zero-based source row at which ingestion starts.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Maximum rows to ingest after --sample-start.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=1,
        help="Commit every N selected samples; values >1 accelerate large ingests.",
    )
    parser.add_argument("--alias-file", type=Path, default=None)
    parser.add_argument(
        "--hnsw-embedding-model",
        "--entity-embedding-model",
        dest="hnsw_embedding_model",
        type=str,
        default=None,
        help="Local embedding model used for entity vectors and HNSW retrieval.",
    )
    parser.add_argument(
        "--kv-embedding-model",
        type=str,
        default=None,
        help="Independent local embedding model used for offset-aligned KV base embeddings.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--hnsw-embedding-batch-size", type=int, default=None)
    parser.add_argument("--kv-embedding-batch-size", type=int, default=None)
    parser.add_argument("--hnsw-prompt-name", type=str, default=None)
    parser.add_argument("--kv-prompt-name", type=str, default=None)
    parser.add_argument(
        "--kv-encoding-profile",
        choices=["qwen3-embedding-v2", "sentence-transformer"],
        default="qwen3-embedding-v2",
        help="KV loading/encoding behavior; qwen3-embedding-v2 matches docs/scripts/embedding_v2.py.",
    )
    parser.add_argument("--build-hnsw", action="store_true")
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-m", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_id = args.dataset_id or args.dataset_path.stem
    resolver = EntityResolver(load_aliases(args.alias_file))
    kv_dir = args.store_dir / "kv"
    graph_dir = args.store_dir / "graph"

    with KVStore(kv_dir) as kv_store, GraphStore(graph_dir, resolver=resolver) as graph_store:
        before_kv = len(kv_store)
        before_graph = graph_store.stats()
        ingest_counts = ingest_dataset(
            args.dataset_path,
            dataset_id,
            kv_store,
            graph_store,
            sample_start=args.sample_start,
            sample_limit=args.sample_limit,
            commit_interval=args.commit_interval,
        )

        if args.hnsw_embedding_model:
            build_entity_embeddings(
                graph_store,
                args.hnsw_embedding_model,
                args.hnsw_embedding_batch_size or args.embedding_batch_size,
                args.hnsw_prompt_name,
            )
        if args.kv_embedding_model:
            build_kv_embeddings(
                kv_store,
                args.kv_embedding_model,
                args.kv_embedding_batch_size or args.embedding_batch_size,
                args.kv_prompt_name,
                args.kv_encoding_profile,
            )
        if args.build_hnsw:
            if not args.hnsw_embedding_model and not graph_store.entity_vectors_path.exists():
                raise ValueError("--build-hnsw requires existing entity embeddings or --hnsw-embedding-model")
            graph_store.build_hnsw_index(
                ef_construction=args.hnsw_ef_construction,
                m=args.hnsw_m,
            )

        after_graph = graph_store.stats()
        print(json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_path": str(args.dataset_path),
                "sample_start": args.sample_start,
                "sample_limit": args.sample_limit,
                "commit_interval": args.commit_interval,
                "ingest": ingest_counts,
                "kv_records_before": before_kv,
                "kv_records_after": len(kv_store),
                "graph_before": before_graph,
                "graph_after": after_graph,
                "hnsw_embedding_model": args.hnsw_embedding_model,
                "kv_embedding_model": args.kv_embedding_model,
                "store_dir": str(args.store_dir),
            },
            indent=2,
        ))


if __name__ == "__main__":
    main()
