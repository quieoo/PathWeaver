#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from transformers import AutoTokenizer


def read_json_or_jsonl(path: Path) -> List[Any]:
    if path.suffix == ".jsonl":
        rows: List[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list-like dataset in {path}, but got {type(data).__name__}")


def iter_text_parts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        title = value.get("title")
        if isinstance(title, str) and title.strip():
            yield title.strip()
        sentences = value.get("sentences")
        if isinstance(sentences, list):
            for sent in sentences:
                yield from iter_text_parts(sent)
            return
        for item in value.values():
            yield from iter_text_parts(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_text_parts(item)


def sample_to_document_text(sample: Any) -> str:
    if isinstance(sample, str):
        return sample.strip()

    if not isinstance(sample, dict):
        return json.dumps(sample, ensure_ascii=False)

    if "context" in sample:
        parts = list(iter_text_parts(sample["context"]))
        if parts:
            return "\n".join(parts)

    for field in ["paragraphs", "passages"]:
        value = sample.get(field)
        if isinstance(value, list):
            parts = list(iter_text_parts(value))
            if parts:
                return "\n".join(parts)

    preferred_fields = ["document", "doc", "text", "content", "paragraph", "passage"]
    for field in preferred_fields:
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    parts: List[str] = []
    for key in ["title", "question", "answer"]:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if parts:
        return "\n".join(parts)

    return json.dumps(sample, ensure_ascii=False)


def sample_to_paragraph_texts(sample: Any) -> List[str]:
    if isinstance(sample, str):
        text = sample.strip()
        return [text] if text else []

    if not isinstance(sample, dict):
        return [json.dumps(sample, ensure_ascii=False)]

    paragraphs: List[str] = []
    context = sample.get("context")
    if isinstance(context, list):
        for item in context:
            parts = list(iter_text_parts(item))
            text = "\n".join(parts).strip()
            if text:
                paragraphs.append(text)
        if paragraphs:
            return paragraphs

    preferred_fields = ["paragraphs", "passages"]
    for field in preferred_fields:
        value = sample.get(field)
        if isinstance(value, list):
            for item in value:
                text = "\n".join(iter_text_parts(item)).strip()
                if text:
                    paragraphs.append(text)
            if paragraphs:
                return paragraphs

    fallback = sample_to_document_text(sample).strip()
    return [fallback] if fallback else []


def sample_to_documents(sample: Any, document_granularity: str) -> List[str]:
    if document_granularity == "paragraph":
        return sample_to_paragraph_texts(sample)
    if document_granularity == "sample":
        text = sample_to_document_text(sample).strip()
        return [text] if text else []
    raise ValueError(f"Unsupported document granularity: {document_granularity}")


def chunk_tokens(tokens: Sequence[int], chunk_size: int) -> List[Tuple[int, ...]]:
    return [tuple(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)]


def count_recomputed_tokens(
    original_tokens: Sequence[int],
    edited_tokens: Sequence[int],
    chunk_size: int,
) -> int:
    original_chunks = chunk_tokens(original_tokens, chunk_size)
    edited_chunks = chunk_tokens(edited_tokens, chunk_size)
    recomputed = 0

    for idx, edited_chunk in enumerate(edited_chunks):
        original_chunk = original_chunks[idx] if idx < len(original_chunks) else None
        if original_chunk != edited_chunk:
            recomputed += len(edited_chunk)
    return recomputed


def choose_edit_token_id(tokenizer: AutoTokenizer, rng: random.Random) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Tokenizer vocab_size is unavailable.")

    special_ids = set(tokenizer.all_special_ids)
    while True:
        token_id = rng.randrange(vocab_size)
        if token_id not in special_ids:
            return token_id


def simulate_edit(
    tokens: Sequence[int],
    tokenizer: AutoTokenizer,
    rng: random.Random,
) -> Tuple[str, int, List[int]]:
    if not tokens:
        raise ValueError("Cannot edit an empty token sequence.")

    position = rng.randrange(len(tokens))
    op = rng.choice(["replace", "insert"])
    edit_token_id = choose_edit_token_id(tokenizer, rng)

    edited_tokens = list(tokens)
    if op == "replace":
        edited_tokens[position] = edit_token_id
    else:
        edited_tokens.insert(position, edit_token_id)

    return op, position, edited_tokens


def summarize(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect update amplification statistics for editable-memory experiments."
    )
    parser.add_argument("--dataset-path", required=True, help="Path to the input json/jsonl dataset.")
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer path or HF model id.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size used by CacheBlend.")
    parser.add_argument(
        "--document-granularity",
        choices=["sample", "paragraph"],
        default="sample",
        help="Treat each sample as one document, or split each sample into paragraph-level documents.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for edit simulation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on processed samples.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to save per-sample results and summary as json.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")

    dataset_path = Path(args.dataset_path)
    rows = read_json_or_jsonl(dataset_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for tokenization. Please run this script in the experiment env "
            "that already has transformers installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    rng = random.Random(args.seed)

    per_sample: List[Dict[str, Any]] = []
    doc_token_counts: List[int] = []
    recomputed_token_counts: List[int] = []
    op_counter = {"replace": 0, "insert": 0}

    for idx, sample in enumerate(rows):
        sample_id = None
        if isinstance(sample, dict):
            sample_id = sample.get("_id") or sample.get("id")
        documents = sample_to_documents(sample, args.document_granularity)
        for doc_idx, document_text in enumerate(documents):
            token_ids = tokenizer.encode(document_text, add_special_tokens=False)
            if not token_ids:
                continue

            doc_token_count = len(token_ids)
            op, position, edited_token_ids = simulate_edit(token_ids, tokenizer, rng)
            recomputed_tokens = count_recomputed_tokens(token_ids, edited_token_ids, args.chunk_size)

            doc_token_counts.append(doc_token_count)
            recomputed_token_counts.append(recomputed_tokens)
            op_counter[op] += 1

            per_sample.append(
                {
                    "index": idx,
                    "sample_id": sample_id,
                    "document_index": doc_idx,
                    "document_granularity": args.document_granularity,
                    "doc_tokens": doc_token_count,
                    "edit_op": op,
                    "edit_position": position,
                    "chunk_size": args.chunk_size,
                    "recomputed_tokens": recomputed_tokens,
                    "update_amplification": recomputed_tokens,
                }
            )

    summary = {
        "dataset_path": str(dataset_path),
        "tokenizer_path": args.tokenizer_path,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "document_granularity": args.document_granularity,
        "num_input_samples": len(rows),
        "num_documents": len(per_sample),
        "avg_doc_tokens": float(mean(doc_token_counts)) if doc_token_counts else 0.0,
        "avg_recomputed_tokens": float(mean(recomputed_token_counts)) if recomputed_token_counts else 0.0,
        "doc_tokens_stats": summarize(doc_token_counts),
        "recomputed_tokens_stats": summarize(recomputed_token_counts),
        "edit_ops": op_counter,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "samples": per_sample}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
