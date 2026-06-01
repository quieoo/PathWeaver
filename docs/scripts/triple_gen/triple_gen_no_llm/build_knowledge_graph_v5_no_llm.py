#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, Optional

from extractor import (
    ExtractionConfig,
    NoLLMTripleExtractor,
    append_jsonl,
    build_programmatic_final_from_graph,
    cache_path,
    heuristic_graph_revision,
    load_cached_json,
    load_existing_output_ids,
    merge_final_stage_into_sample,
    merge_stage1_and_stage2_graph,
    normalize_final_sample_output,
    read_json_or_jsonl,
    safe_sample_id,
    save_cached_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="No-LLM replacement pipeline for build_knowledge_graph_v5.py"
    )
    ap.add_argument("--input", type=str, required=True, help="Input .json or .jsonl file")
    ap.add_argument("--output", type=str, required=True, help="Output .jsonl file")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N samples")
    ap.add_argument("--resume", action="store_true", help="Skip samples already present in output jsonl")
    ap.add_argument("--overwrite", action="store_true", help="Ignore caches and recompute")
    ap.add_argument(
        "--stage-cache-dir",
        type=str,
        default=None,
        help="Optional root dir for stage caches; uses stage1/stage2/final subdirs",
    )
    ap.add_argument(
        "--error-log",
        type=str,
        default="./kg_extract_no_llm_errors.log",
        help="Error log path",
    )
    ap.add_argument(
        "--supporting-pages-only",
        action="store_true",
        help="Use only supporting pages when supporting_facts exist",
    )
    ap.add_argument(
        "--include-question-entities",
        action="store_true",
        help="Merge question-side entity candidates into stage1 entity_list",
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
    ap.add_argument(
        "--disable-answer-aware",
        action="store_true",
        help="Skip heuristic stage2 answer-aware repair and only keep stage1 graph",
    )
    ap.add_argument(
        "--max-hops",
        type=int,
        default=4,
        help="Maximum relation hops for answer bridge checking",
    )
    return ap


def stage_cache_subdir(root: Optional[str], stage_name: str) -> Optional[str]:
    if not root:
        return None
    return str(Path(root) / stage_name)


def process_one_sample(
    sample: Dict,
    idx: int,
    extractor: NoLLMTripleExtractor,
    args: argparse.Namespace,
) -> Dict:
    sample_id = safe_sample_id(sample, idx)
    sample["_id"] = sample_id

    stage1_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "stage1"), sample_id, "stage1")
    stage2_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "stage2"), sample_id, "stage2")
    final_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "final"), sample_id, "final")

    stage1 = None if args.overwrite else load_cached_json(stage1_path)
    if stage1 is None:
        stage1 = extractor.build_graph(sample, idx=idx)
        save_cached_json(stage1_path, stage1)

    stage2 = None if args.overwrite else load_cached_json(stage2_path)
    if stage2 is None:
        if args.disable_answer_aware:
            stage2 = {
                "_id": sample_id,
                "entity_list": stage1.get("entity_list", []) or [],
                "triples": stage1.get("triples", []) or [],
                "answer_sufficient": False,
                "missing_links": [],
                "revision_notes": [],
            }
        else:
            stage2 = heuristic_graph_revision(
                sample,
                stage1,
                extractor,
                max_hops=max(1, args.max_hops),
            )
        save_cached_json(stage2_path, stage2)

    stage_final = None if args.overwrite else load_cached_json(final_path)
    if stage_final is None:
        graph_for_kv = merge_stage1_and_stage2_graph(stage1, stage2)
        stage_final = build_programmatic_final_from_graph(sample, graph_for_kv)
        save_cached_json(final_path, stage_final)

    final_sample = merge_final_stage_into_sample(sample, stage_final)
    return normalize_final_sample_output(final_sample)


def main() -> None:
    args = build_arg_parser().parse_args()
    samples = read_json_or_jsonl(args.input)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.error_log:
        Path(args.error_log).parent.mkdir(parents=True, exist_ok=True)

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
    errors = 0
    for idx, sample in enumerate(samples):
        sample_id = safe_sample_id(sample, idx)
        if sample_id in existing_ids:
            skipped += 1
            continue
        try:
            final_sample = process_one_sample(sample, idx, extractor, args)
            append_jsonl(args.output, final_sample)
            processed += 1
            if processed % 100 == 0:
                print(
                    f"processed={processed} skipped={skipped} errors={errors} "
                    f"spacy_enabled={extractor.spacy_enabled} last={sample_id}"
                )
        except Exception as exc:
            errors += 1
            msg = (
                f"sample={sample_id} idx={idx} error={exc}\n"
                f"{traceback.format_exc()}\n"
            )
            print(msg)
            if args.error_log:
                with open(args.error_log, "a", encoding="utf-8") as f:
                    f.write(msg)

    summary = {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "spacy_enabled": extractor.spacy_enabled,
        "output": args.output,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
