#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from pathlib import Path as _Path

from extractor_rebel import RebelConfig, RebelTripleExtractor


_NO_LLM_DIR = _Path(__file__).resolve().parent.parent / "triple_gen_no_llm"
if str(_NO_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_NO_LLM_DIR))

from extractor import append_jsonl, load_existing_output_ids, read_json_or_jsonl, safe_sample_id  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="REBEL stage1 triple extraction baseline")
    ap.add_argument("--input", type=str, required=True, help="Input .json or .jsonl")
    ap.add_argument("--output", type=str, required=True, help="Output .jsonl")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N samples")
    ap.add_argument("--resume", action="store_true", help="Skip samples already in output")
    ap.add_argument("--overwrite", action="store_true", help="Ignore existing output file contents")
    ap.add_argument(
        "--use-all-context-pages",
        action="store_true",
        help="Use all context/pages; default is supporting pages only",
    )
    ap.add_argument(
        "--supporting-pages-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--include-question-entities",
        action="store_true",
        help="Merge lightweight question-side entity candidates into entity_list",
    )
    ap.add_argument("--skip-comparison", action="store_true", help="Skip samples whose type is comparison")
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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress_line(*, processed, skipped, total, started_at, last_sample_id, runtime_stats, width=24) -> str:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    done = processed + skipped
    pct = (done / total) if total > 0 else 1.0
    filled = min(width, int(width * pct))
    bar = "#" * filled + "-" * (width - filled)
    sample_rate = processed / elapsed if processed > 0 else 0.0
    sentence_rate = runtime_stats.get("sentences_per_second", 0.0)
    remaining = max(0, total - done)
    eta = (remaining / sample_rate) if sample_rate > 0 else 0.0
    return (
        f"[{bar}] {done}/{total} {pct * 100:5.1f}% "
        f"ok={processed} skip={skipped} "
        f"samples={sample_rate:.2f}/s sent={sentence_rate:.2f}/s "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"last={last_sample_id}"
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    samples = read_json_or_jsonl(args.input)
    samples = [s for s in samples if s.get("answerable", True) is not False]
    if args.skip_comparison:
        samples = [s for s in samples if s.get("type", "") != "comparison"]
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    started_at = time.perf_counter()
    pending = []
    sample_batch_size = max(1, args.sample_batch_size)
    total_to_visit = len(samples)

    def print_progress(last_sample_id):
        runtime_stats = extractor.get_runtime_stats()
        line = render_progress_line(
            processed=processed,
            skipped=skipped,
            total=total_to_visit,
            started_at=started_at,
            last_sample_id=last_sample_id,
            runtime_stats=runtime_stats,
        )
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def flush_pending(batch_items):
        nonlocal processed
        if not batch_items:
            return
        built = extractor.build_graphs([(sample, idx) for idx, sample in batch_items])
        for idx, sample in batch_items:
            sample_id = safe_sample_id(sample, idx)
            graph = built[sample_id]
            append_jsonl(args.output, graph)
            processed += 1
            if processed % 100 == 0:
                print_progress(sample_id)

    for idx, sample in enumerate(samples):
        sample_id = safe_sample_id(sample, idx)
        if sample_id in existing_ids:
            skipped += 1
            if (processed + skipped) % 100 == 0:
                print_progress(sample_id)
            continue
        pending.append((idx, sample))
        if len(pending) >= sample_batch_size:
            flush_pending(pending)
            pending = []

    flush_pending(pending)
    if total_to_visit > 0:
        last_sample_id = safe_sample_id(samples[-1], len(samples) - 1)
        print_progress(last_sample_id)
        sys.stdout.write("\n")

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    runtime_stats = extractor.get_runtime_stats()
    print(
        json.dumps(
            {
                "processed": processed,
                "skipped": skipped,
                "output": args.output,
                "model_name": args.model_name,
                "elapsed_seconds": elapsed,
                "samples_per_second": processed / elapsed if processed > 0 else 0.0,
                "triples_per_second": runtime_stats.get("triples_built", 0) / elapsed if processed > 0 else 0.0,
                "sample_batch_size": sample_batch_size,
                "sentence_batch_size": max(1, args.batch_size),
                "runtime_stats": runtime_stats,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
