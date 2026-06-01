#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from extractor_openie import OpenIEConfig, StanfordOpenIEExtractor

import sys
from pathlib import Path as _Path

_NO_LLM_DIR = _Path(__file__).resolve().parent.parent / "triple_gen_no_llm"
if str(_NO_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_NO_LLM_DIR))

from extractor import append_jsonl, load_existing_output_ids, read_json_or_jsonl, safe_sample_id  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stanford OpenIE stage1 triple extraction baseline")
    ap.add_argument("--input", type=str, required=True, help="Input .json or .jsonl")
    ap.add_argument("--output", type=str, required=True, help="Output .jsonl")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N samples")
    ap.add_argument("--resume", action="store_true", help="Skip samples already in output")
    ap.add_argument("--overwrite", action="store_true", help="Ignore existing output file contents")
    ap.add_argument("--skip-comparison", action="store_true", help="Skip samples whose type is comparison")
    ap.add_argument(
        "--skip-comprision",
        action="store_true",
        help="Compatibility alias for --skip-comparison",
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed for shuffling samples")
    ap.add_argument(
        "--supporting-pages-only",
        action="store_true",
        help="Use only supporting pages when supporting_facts exist",
    )
    ap.add_argument(
        "--include-question-entities",
        action="store_true",
        help="Merge lightweight question-side entity candidates into entity_list",
    )
    ap.add_argument("--corenlp-url", type=str, default="http://localhost:9000", help="Stanford CoreNLP server URL")
    ap.add_argument("--request-timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    ap.add_argument("--openie-min-confidence", type=float, default=0.35, help="Discard triples below this OpenIE confidence")
    ap.add_argument("--openie-max-entailments-per-clause", type=int, default=150, help="Maps to openie.max_entailments_per_clause")
    ap.add_argument("--openie-no-strict", action="store_true", help="Set openie.triple.strict=false")
    ap.add_argument("--openie-all-nominals", action="store_true", help="Set openie.triple.all_nominals=true")
    ap.add_argument("--openie-resolve-coref", action="store_true", help="Set openie.resolve_coref=true")
    ap.add_argument("--with-ner", action="store_true", help="Also request NER tokens from CoreNLP for entity harvesting")
    ap.add_argument("--max-triples-per-sentence", type=int, default=8, help="Cap triples kept from one sentence")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    samples = read_json_or_jsonl(args.input)
    samples = [s for s in samples if s.get("answerable", True) is not False]
    if args.skip_comparison or args.skip_comprision:
        samples = [s for s in samples if s.get("type", "") != "comparison"]
    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(samples)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and output_path.exists() and output_path.stat().st_size > 0:
        output_path.write_text("", encoding="utf-8")

    existing_ids = set()
    if args.resume and not args.overwrite:
        existing_ids = load_existing_output_ids(args.output)

    extractor = StanfordOpenIEExtractor(
        OpenIEConfig(
            corenlp_url=args.corenlp_url,
            supporting_pages_only=args.supporting_pages_only,
            include_question_entities=args.include_question_entities,
            request_timeout=args.request_timeout,
            strict=not args.openie_no_strict,
            max_entailments_per_clause=max(1, args.openie_max_entailments_per_clause),
            triple_all_nominals=args.openie_all_nominals,
            resolve_coref=args.openie_resolve_coref,
            with_ner=args.with_ner,
            min_confidence=args.openie_min_confidence,
            max_triples_per_sentence=max(1, args.max_triples_per_sentence),
        )
    )

    processed = 0
    skipped = 0
    started_at = time.perf_counter()
    for idx, sample in enumerate(samples):
        sample_id = safe_sample_id(sample, idx)
        if sample_id in existing_ids:
            skipped += 1
            continue
        graph = extractor.build_graph(sample, idx=idx)
        append_jsonl(args.output, graph)
        processed += 1
        if processed % 100 == 0:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            doc_rate = processed / elapsed
            print(f"processed={processed} skipped={skipped} doc/s={doc_rate:.2f} last={sample_id}")

    elapsed = max(time.perf_counter() - started_at, 1e-9)

    print(
        json.dumps(
            {
                "processed": processed,
                "skipped": skipped,
                "input_samples": len(samples),
                "docs_per_second": processed / elapsed if processed > 0 else 0.0,
                "output": args.output,
                "corenlp_url": args.corenlp_url,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
