#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from extractor import (
    ExtractionConfig,
    NoLLMTripleExtractor,
    append_jsonl,
    load_existing_output_ids,
    read_json_or_jsonl,
    safe_sample_id,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="No-LLM stage1 triple extraction baseline")
    ap.add_argument("--input", type=str, required=True, help="Input .json or .jsonl")
    ap.add_argument("--output", type=str, required=True, help="Output .jsonl")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N samples")
    ap.add_argument("--resume", action="store_true", help="Skip samples already in output")
    ap.add_argument("--overwrite", action="store_true", help="Ignore existing output file contents")
    ap.add_argument(
        "--supporting-pages-only",
        action="store_true",
        help="Use only supporting pages when supporting_facts exist",
    )
    ap.add_argument(
        "--include-question-entities",
        action="store_true",
        help="Merge question-side entity candidates into entity_list",
    )
    ap.add_argument(
        "--use-spacy",
        action="store_true",
        help="Use spaCy if installed and en_core_web_sm is available",
    )
    ap.add_argument(
        "--max-triples-per-sentence",
        type=int,
        default=32,
        help="Cap candidate triples kept from one sentence",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    samples = read_json_or_jsonl(args.input)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if args.resume and not args.overwrite:
        existing_ids = load_existing_output_ids(args.output)

    extractor = NoLLMTripleExtractor(
        ExtractionConfig(
            supporting_pages_only=args.supporting_pages_only,
            include_question_entities=args.include_question_entities,
            use_spacy=args.use_spacy,
            max_triples_per_sentence=max(1, args.max_triples_per_sentence),
        )
    )

    processed = 0
    skipped = 0
    for idx, sample in enumerate(samples):
        sample_id = safe_sample_id(sample, idx)
        if sample_id in existing_ids:
            skipped += 1
            continue
        graph = extractor.build_graph(sample, idx=idx)
        append_jsonl(args.output, graph)
        processed += 1
        if processed % 100 == 0:
            print(
                f"processed={processed} skipped={skipped} "
                f"spacy_enabled={extractor.spacy_enabled} last={sample_id}"
            )

    print(
        f"done processed={processed} skipped={skipped} "
        f"spacy_enabled={extractor.spacy_enabled} output={args.output}"
    )


if __name__ == "__main__":
    main()
