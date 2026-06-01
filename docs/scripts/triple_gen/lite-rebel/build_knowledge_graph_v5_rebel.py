#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_NO_LLM_DIR = Path(__file__).resolve().parent.parent / "triple_gen_no_llm"
if str(_NO_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_NO_LLM_DIR))

from extractor import (  # type: ignore
    append_jsonl,
    build_programmatic_final_from_graph,
    cache_path,
    load_cached_json,
    load_existing_output_ids,
    merge_final_stage_into_sample,
    merge_stage1_and_stage2_graph,
    normalize_final_sample_output,
    read_json_or_jsonl,
    safe_sample_id,
    save_cached_json,
)

from extractor_rebel import RebelConfig, RebelTripleExtractor, heuristic_graph_revision_rebel


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="REBEL-based replacement for build_knowledge_graph_v5.py"
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
        default="./kg_extract_rebel_errors.log",
        help="Error log path",
    )
    ap.add_argument(
        "--use-all-context-pages",
        action="store_true",
        help="Use all context/pages; default is supporting pages only",
    )
    ap.add_argument(
        "--include-question-entities",
        action="store_true",
        help="Merge lightweight question-side entity candidates into entity_list",
    )
    ap.add_argument(
        "--enable-answer-aware",
        action="store_true",
        help="Explicitly enable rebel-specific heuristic stage2 repair (enabled by default)",
    )
    ap.add_argument(
        "--disable-answer-aware",
        action="store_true",
        help="Skip stage2 and keep only stage1 graph",
    )
    ap.add_argument(
        "--max-hops",
        type=int,
        default=4,
        help="Maximum relation hops for answer bridge checking when stage2 is enabled",
    )
    ap.add_argument("--skip-comparison", action="store_true", help="Skip samples whose type is comparison")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for shuffling samples")
    ap.add_argument("--progress-every", type=int, default=100, help="Print progress every N processed samples")

    ap.add_argument("--model-name", type=str, default="Babelscape/rebel-large", help="Hugging Face model id or local path")
    ap.add_argument("--hf-cache-dir", type=str, default=None, help="Optional Hugging Face cache dir")
    ap.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    ap.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Model dtype for GPU inference; auto prefers bf16/fp16 on CUDA",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="Sentence batch size for generation")
    ap.add_argument("--sample-batch-size", type=int, default=4, help="Number of samples to batch together for stage1 inference")
    ap.add_argument("--max-input-length", type=int, default=256, help="Tokenizer truncation length")
    ap.add_argument("--max-new-tokens", type=int, default=192, help="Max new tokens for decoder")
    ap.add_argument("--num-beams", type=int, default=3, help="Beam size for generation")
    ap.add_argument("--max-triples-per-sentence", type=int, default=16, help="Cap triples kept from one sentence")
    return ap


def stage_cache_subdir(root: Optional[str], stage_name: str) -> Optional[str]:
    if not root:
        return None
    return str(Path(root) / stage_name)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress_line(
    *,
    processed: int,
    total: int,
    skipped: int,
    errors: int,
    started_at: float,
    last_sample_id: str,
    runtime_stats: Dict[str, Any],
    width: int = 24,
) -> str:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    done = processed + skipped + errors
    pct = (done / total) if total > 0 else 1.0
    filled = min(width, int(width * pct))
    bar = "#" * filled + "-" * (width - filled)
    doc_rate = processed / elapsed if processed > 0 else 0.0
    remaining = max(0, total - done)
    eta = (remaining / doc_rate) if doc_rate > 0 else 0.0
    return (
        f"[{bar}] {done}/{total} {pct * 100:5.1f}% "
        f"ok={processed} skip={skipped} err={errors} "
        f"doc/s={doc_rate:.2f} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"last={last_sample_id}"
    )


def prepare_stage1_batch(
    batch_items: Sequence[Tuple[int, Dict]],
    extractor: RebelTripleExtractor,
    args: argparse.Namespace,
) -> Dict[str, Dict]:
    stage1_by_id: Dict[str, Dict] = {}
    pending: List[Tuple[Dict, int]] = []

    for idx, sample in batch_items:
        sample_id = safe_sample_id(sample, idx)
        sample["_id"] = sample_id
        stage1_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "stage1"), sample_id, "stage1")
        stage1 = None if args.overwrite else load_cached_json(stage1_path)
        if stage1 is None:
            pending.append((sample, idx))
            continue
        stage1_by_id[sample_id] = stage1

    if pending:
        built = extractor.build_graphs(pending)
        for sample, idx in pending:
            sample_id = safe_sample_id(sample, idx)
            stage1 = built[sample_id]
            stage1_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "stage1"), sample_id, "stage1")
            save_cached_json(stage1_path, stage1)
            stage1_by_id[sample_id] = stage1

    return stage1_by_id


def process_one_sample(
    sample: Dict,
    idx: int,
    extractor: RebelTripleExtractor,
    args: argparse.Namespace,
    stage1: Optional[Dict] = None,
) -> Dict:
    sample_id = safe_sample_id(sample, idx)
    sample["_id"] = sample_id

    stage2_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "stage2"), sample_id, "stage2")
    final_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "final"), sample_id, "final")

    if stage1 is None:
        stage1_path = cache_path(stage_cache_subdir(args.stage_cache_dir, "stage1"), sample_id, "stage1")
        stage1 = None if args.overwrite else load_cached_json(stage1_path)
        if stage1 is None:
            stage1 = extractor.build_graph(sample, idx=idx)
            save_cached_json(stage1_path, stage1)

    stage2 = None if args.overwrite else load_cached_json(stage2_path)
    answer_aware_enabled = not bool(args.disable_answer_aware)
    if stage2 is None:
        if not answer_aware_enabled:
            stage2 = {
                "_id": sample_id,
                "entity_list": stage1.get("entity_list", []) or [],
                "triples": stage1.get("triples", []) or [],
                "answer_sufficient": False,
                "missing_links": [],
                "revision_notes": ["stage2 skipped: answer-aware repair disabled"],
            }
        else:
            stage2 = heuristic_graph_revision_rebel(
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
    samples = [s for s in samples if s.get("answerable", True) is not False]
    if args.skip_comparison:
        samples = [s for s in samples if s.get("type", "") != "comparison"]
    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(samples)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.error_log:
        Path(args.error_log).parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and output_path.exists() and output_path.stat().st_size > 0:
        output_path.write_text("", encoding="utf-8")

    existing_ids = set()
    if args.resume and not args.overwrite:
        existing_ids = load_existing_output_ids(args.output)

    extractor = RebelTripleExtractor(
        RebelConfig(
            model_name=args.model_name,
            supporting_pages_only=not args.use_all_context_pages,
            include_question_entities=args.include_question_entities,
            batch_size=max(1, args.batch_size),
            sample_batch_size=max(1, args.sample_batch_size),
            max_input_length=max(32, args.max_input_length),
            max_new_tokens=max(32, args.max_new_tokens),
            num_beams=max(1, args.num_beams),
            max_triples_per_sentence=max(1, args.max_triples_per_sentence),
            hf_cache_dir=args.hf_cache_dir,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
    )

    processed = 0
    skipped = 0
    errors = 0
    started_at = time.perf_counter()
    pending_batch: List[Tuple[int, Dict]] = []
    sample_batch_size = max(1, args.sample_batch_size)
    total_to_visit = len(samples)

    def print_progress(last_sample_id: str) -> None:
        runtime_stats = extractor.get_runtime_stats()
        line = render_progress_line(
            processed=processed,
            total=total_to_visit,
            skipped=skipped,
            errors=errors,
            started_at=started_at,
            last_sample_id=last_sample_id,
            runtime_stats=runtime_stats,
        )
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def flush_pending(batch_items: Sequence[Tuple[int, Dict]]) -> None:
        nonlocal processed, errors
        if not batch_items:
            return
        stage1_by_id = prepare_stage1_batch(batch_items, extractor, args)
        for batch_idx, batch_sample in batch_items:
            sample_id = safe_sample_id(batch_sample, batch_idx)
            try:
                final_sample = process_one_sample(
                    batch_sample,
                    batch_idx,
                    extractor,
                    args,
                    stage1=stage1_by_id.get(sample_id),
                )
                append_jsonl(args.output, final_sample)
                processed += 1
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print_progress(sample_id)
            except Exception as exc:
                errors += 1
                msg = f"sample={sample_id} idx={batch_idx} error={exc}\n{traceback.format_exc()}\n"
                sys.stdout.write("\n")
                print(msg)
                if args.error_log:
                    with open(args.error_log, "a", encoding="utf-8") as f:
                        f.write(msg)
                print_progress(sample_id)

    for idx, sample in enumerate(samples):
        sample_id = safe_sample_id(sample, idx)
        if sample_id in existing_ids:
            skipped += 1
            if args.progress_every > 0 and (processed + skipped + errors) % args.progress_every == 0:
                print_progress(sample_id)
            continue
        pending_batch.append((idx, sample))
        if len(pending_batch) >= sample_batch_size:
            flush_pending(pending_batch)
            pending_batch = []

    flush_pending(pending_batch)
    if total_to_visit > 0:
        last_sample_id = safe_sample_id(samples[-1], len(samples) - 1)
        print_progress(last_sample_id)
        sys.stdout.write("\n")

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    runtime_stats = extractor.get_runtime_stats()
    summary = {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "output": args.output,
        "model_name": args.model_name,
        "answer_aware_enabled": not bool(args.disable_answer_aware),
        "elapsed_seconds": elapsed,
        "docs_per_second": processed / elapsed if processed > 0 else 0.0,
        "triples_per_second": runtime_stats.get("triples_built", 0) / elapsed if processed > 0 else 0.0,
        "sample_batch_size": sample_batch_size,
        "sentence_batch_size": max(1, args.batch_size),
        "runtime_stats": runtime_stats,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
