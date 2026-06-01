#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_NO_LLM_DIR = Path(__file__).resolve().parent.parent / "triple_gen_no_llm"
if str(_NO_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_NO_LLM_DIR))

from extractor import (  # type: ignore
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

from extractor_openie import OpenIEConfig, StanfordOpenIEExtractor


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="LLM-free Stanford OpenIE replacement for build_knowledge_graph_v5.py"
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
        default="./kg_extract_openie_errors.log",
        help="Error log path",
    )
    ap.add_argument(
        "--use-all-context-pages",
        action="store_true",
        help="Use all context/pages; default is supporting pages only",
    )
    ap.add_argument(
        "--supporting-pages-only",
        action="store_true",
        help="Compatibility alias; default behavior already uses only supporting pages",
    )
    ap.add_argument(
        "--include-question-entities",
        action="store_true",
        help="Merge lightweight question-side entity candidates into entity_list",
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
    ap.add_argument("--skip-comparison", action="store_true", help="Skip samples whose type is comparison")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for shuffling samples")
    ap.add_argument("--progress-every", type=int, default=100, help="Print progress every N processed samples")
    ap.add_argument("--workers", type=int, default=4, help="Number of concurrent samples to process")

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
    triples_done: int,
    started_at: float,
    last_sample_id: str,
    width: int = 24,
) -> str:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    done = processed + skipped + errors
    pct = (done / total) if total > 0 else 1.0
    filled = min(width, int(width * pct))
    bar = "#" * filled + "-" * (width - filled)
    doc_rate = processed / elapsed if processed > 0 else 0.0
    triple_rate = triples_done / elapsed if triples_done > 0 else 0.0
    remaining = max(0, total - done)
    eta = (remaining / doc_rate) if doc_rate > 0 else 0.0
    return (
        f"[{bar}] {done}/{total} {pct * 100:5.1f}% "
        f"ok={processed} skip={skipped} err={errors} "
        f"doc/s={doc_rate:.2f} triples={triple_rate:.2f}/s "
        f"triples_done={triples_done} elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)} last={last_sample_id}"
    )


def emit_progress(line: str, *, final: bool = False) -> None:
    suffix = "\n" if final else "\r"
    print(line, end=suffix, flush=True)


def process_one_sample(
    sample: Dict,
    idx: int,
    extractor: StanfordOpenIEExtractor,
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


def process_sample_task(
    idx: int,
    sample: Dict[str, Any],
    extractor: StanfordOpenIEExtractor,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Any]]:
    sample_id = safe_sample_id(sample, idx)
    final_sample = process_one_sample(sample, idx, extractor, args)
    return sample_id, final_sample


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.use_all_context_pages and args.supporting_pages_only:
        raise SystemExit("Cannot use both --use-all-context-pages and --supporting-pages-only")

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

    extractor = StanfordOpenIEExtractor(
        OpenIEConfig(
            corenlp_url=args.corenlp_url,
            supporting_pages_only=(not args.use_all_context_pages) or args.supporting_pages_only,
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

    total = len(samples)
    processed = 0
    skipped = 0
    errors = 0
    triples_done = 0
    started_at = time.perf_counter()
    last_sample_id = "-"
    final_progress_emitted = False

    pending_items: List[Tuple[int, Dict[str, Any]]] = []
    for idx, sample in enumerate(samples):
        sample_id = safe_sample_id(sample, idx)
        if sample_id in existing_ids:
            skipped += 1
            last_sample_id = sample_id
            continue
        pending_items.append((idx, sample))

    if total > 0:
        emit_progress(
            render_progress_line(
                processed=processed,
                total=total,
                skipped=skipped,
                errors=errors,
                triples_done=triples_done,
                started_at=started_at,
                last_sample_id=last_sample_id,
            )
        )

    workers = max(1, args.workers)
    progress_every = max(1, args.progress_every)
    futures: Dict[Future[Tuple[str, Dict[str, Any]]], Tuple[int, str]] = {}

    if workers == 1:
        for idx, sample in pending_items:
            sample_id = safe_sample_id(sample, idx)
            try:
                _, final_sample = process_sample_task(idx, sample, extractor, args)
                append_jsonl(args.output, final_sample)
                processed += 1
                triples_done += len(final_sample.get("triple_list", []) or [])
                last_sample_id = sample_id
            except Exception as exc:
                errors += 1
                last_sample_id = sample_id
                msg = f"sample={sample_id} idx={idx} error={exc}\n{traceback.format_exc()}\n"
                print(msg)
                if args.error_log:
                    with open(args.error_log, "a", encoding="utf-8") as f:
                        f.write(msg)
            done = processed + skipped + errors
            if done % progress_every == 0 or done == total:
                emit_progress(
                    render_progress_line(
                        processed=processed,
                        total=total,
                        skipped=skipped,
                        errors=errors,
                        triples_done=triples_done,
                        started_at=started_at,
                        last_sample_id=last_sample_id,
                    ),
                    final=(done == total),
                )
                final_progress_emitted = (done == total)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, sample in pending_items:
                sample_id = safe_sample_id(sample, idx)
                future = executor.submit(process_sample_task, idx, sample, extractor, args)
                futures[future] = (idx, sample_id)

            for future in as_completed(futures):
                idx, sample_id = futures[future]
                last_sample_id = sample_id
                try:
                    _, final_sample = future.result()
                    append_jsonl(args.output, final_sample)
                    processed += 1
                    triples_done += len(final_sample.get("triple_list", []) or [])
                except Exception as exc:
                    errors += 1
                    msg = f"sample={sample_id} idx={idx} error={exc}\n{traceback.format_exc()}\n"
                    print(msg)
                    if args.error_log:
                        with open(args.error_log, "a", encoding="utf-8") as f:
                            f.write(msg)

                done = processed + skipped + errors
                if done % progress_every == 0 or done == total:
                    emit_progress(
                        render_progress_line(
                            processed=processed,
                            total=total,
                            skipped=skipped,
                            errors=errors,
                            triples_done=triples_done,
                            started_at=started_at,
                            last_sample_id=last_sample_id,
                        ),
                        final=(done == total),
                    )
                    final_progress_emitted = (done == total)

    if total > 0 and not final_progress_emitted:
        emit_progress(
            render_progress_line(
                processed=processed,
                total=total,
                skipped=skipped,
                errors=errors,
                triples_done=triples_done,
                started_at=started_at,
                last_sample_id=last_sample_id,
            ),
            final=True,
        )
    else:
        print()

    summary = {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "output": args.output,
        "corenlp_url": args.corenlp_url,
        "workers": workers,
        "triples_done": triples_done,
        "elapsed_seconds": round(max(0.0, time.perf_counter() - started_at), 3),
        "docs_per_second": round(
            (processed / max(time.perf_counter() - started_at, 1e-9)) if processed > 0 else 0.0,
            6,
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
