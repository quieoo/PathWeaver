#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import requests
import re
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque

# ============================================================
# FIXED PROMPT PREFIX (PREFIX-CACHE FRIENDLY)
# ⚠️ Do NOT change spaces / newlines casually
# ============================================================

PROMPT_PREFIX = (
    "You are a knowledge extraction system.\n\n"
    "Your task is to extract factual triples from ONE context.\n"
    "============================================================\n"
    "ENTITY AWARENESS\n"
    "============================================================\n"
    "- The FIRST element of the context is the entity name (title).\n"
    "- This entity MUST be treated as a real-world entity node.\n"
    "- In every triple, the head (name) MUST be an entity.\n"
    "- Use surface names exactly as they appear in the text.\n"
    "- Do NOT perform entity linking or disambiguation.\n\n"
    "============================================================\n"
    "TRIPLE TYPES\n"
    "============================================================\n"
    "Each extracted triple MUST be classified as exactly one type:\n\n"
    "ATTRIBUTE\n"
    "- Describes an intrinsic property, role, or attribute of an entity.\n"
    "- The tail is usually a literal value (not an entity).\n"
    "- Semantic form: the <description_type> of <name> is <description>\n\n"
    "RELATION\n"
    "- Describes a semantic relation between TWO entities.\n"
    "- The tail SHOULD be another entity whenever possible.\n"
    "- The description_type is usually a verb phrase or contains a preposition.\n"
    "- Semantic form: <name> <description_type> <description>\n\n"
    "============================================================\n"
    "AUXILIARY ATTRIBUTE VIEW (FOR RELATION ONLY)\n"
    "============================================================\n"
    "For EACH RELATION triple, you SHOULD derive an attribute-style description\n"
    "that can be used to answer attribute-form questions.\n\n"
    "- The attribute_desc_alias MUST be derived from the same fact.\n"
    "- It MUST NOT introduce new information.\n"
    "- If no clear attribute-style description exists, leave it empty.\n\n"
    "Examples:\n"
    "- RELATION:  Mina Gerhardsen | is the daughter of | Rune Gerhardsen\n"
    "  attribute_desc_alias: father\n\n"
    "- RELATION:  Book X | was written by | Author Y\n"
    "  attribute_desc_alias: author\n\n"
    "============================================================\n"
    "OUTPUT FORMAT (STRICT)\n"
    "============================================================\n"
    "Output PLAIN TEXT ONLY. Do NOT output JSON or explanations.\n\n"
    "Each triple MUST be written on ONE line.\n\n"
    "ATTRIBUTE | name | description_type | description\n"
    "RELATION  | name | description_type | description | attribute_desc_alias\n\n"
    "============================================================\n"
    "EXAMPLES\n"
    "============================================================\n"
    "[CTX 0]\n"
    "ATTRIBUTE | Pamela Jain | occupation | Indian playback singer\n"
    "ATTRIBUTE | Pamela Jain | date of birth | 16 March\n\n"
    "[CTX 0]\n"
    "RELATION | Mina Gerhardsen | is the daughter of | Rune Gerhardsen | father\n\n"
    "============================================================\n"
    "RULES\n"
    "============================================================\n"
    "- Do NOT invent facts.\n"
    "- Do NOT merge information across contexts.\n"
    "- Keep the original textual order.\n"
    "- Do NOT create multi-hop or inferred relations.\n\n"
    "============================================================\n"
    "Convert the following context:\n"
)

# ============================================================
# Build prompt for ONE context
# context format: [title, [sentences]]
# ============================================================


def build_ctx_prompt(title: str, sentences: list[str]) -> str:
    ctx = [title, sentences]
    # IMPORTANT: separators ensures prefix-cache stability
    return PROMPT_PREFIX + json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))

# ============================================================
# OpenAI-compatible batch call (/v1/completions)
# ============================================================

def call_batch(
    endpoint: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    timeout: int,
):
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    outs = []
    choices = data.get("choices", [])
    for i in range(len(prompts)):
        if i < len(choices):
            c = choices[i]
            if isinstance(c, dict) and "text" in c:
                outs.append((c.get("text") or "").strip())
            else:
                outs.append("")
        else:
            outs.append("")
    return outs



# ============================================================
# Parse typed triples (single-context)
# ============================================================

CTX_RE = re.compile(r"^\s*\[CTX\s+(\d+)\]\s*$", re.IGNORECASE)

def build_key_string(triple_type: str, name: str, desc_type: str) -> str:
    if triple_type == "ATTRIBUTE":
        return f"the {desc_type} of {name}"
    return f"{name} {desc_type}"

def parse_typed_triples(text: str):
    """
    Parse output for ONE context.
    """
    if not text:
        return []

    triples = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue

        triple_type = parts[0].upper()
        if triple_type not in {"ATTRIBUTE", "RELATION"}:
            continue

        name = parts[1]
        desc_type = parts[2]
        desc = parts[3]

        if not name or not desc_type or not desc:
            continue

        triple = {
            "type": triple_type,
            "name": name,
            "description_type": desc_type,
            "description": desc,
            "key_string": build_key_string(triple_type, name, desc_type),
        }

        if triple_type == "RELATION":
            if len(parts) >= 5 and parts[4]:
                triple["attribute_desc_alias"] = parts[4]
                triple["key_string_alias"] = build_key_string(
                    "ATTRIBUTE", name, parts[4]
                )
            else:
                triple["attribute_desc_alias"] = None

        triples.append(triple)

    return triples

# ============================================================
# Flatten samples into context-level tasks
# ============================================================

# def flatten_samples(samples):
#     flat = []
#     for sid, s in enumerate(samples):
#         for cid, (title, sentences) in enumerate(s["context"]):
#             flat.append({
#                 "sample_id": sid,
#                 "ctx_id": cid,
#                 "title": title,
#                 "sentences": sentences,
#             })
#     return flat

def filter_sample_hotpot(samples):
    return [s for s in samples if s["type"] == "bridge"]

def flatten_samples(samples, supporting_only: bool):
    """
    Return:
      flat_ctxs: list of ctx tasks
      total_ctx_per_sample: dict[sample_id] -> number of ctxs that must be completed
    """
    flat = []
    total_ctx_per_sample = defaultdict(int)

    for sid, sample in enumerate(samples):
        # -------- hotpot supporting titles --------
        supporting_titles = None
        if supporting_only:
            supporting_titles = {
                t for (t, _) in sample.get("supporting_facts", [])
            }

        for cid, (title, sentences) in enumerate(sample["context"]):
            if supporting_titles is not None and title not in supporting_titles:
                continue

            flat.append({
                "sample_id": sid,
                "ctx_id": cid,
                "title": title,
                "sentences": sentences,
            })
            total_ctx_per_sample[sid] += 1

    return flat, total_ctx_per_sample

def flatten_samples_musique(samples, supporting_only: bool = False):
    """
    Flatten Musique dataset samples for triple extraction.

    Args:
        samples: List of Musique samples with 'paragraphs' field
        supporting_only: If True, only process supporting paragraphs

    Return:
      flat_ctxs: list of ctx tasks
      total_ctx_per_sample: dict[sample_id] -> number of ctxs that must be completed
    """
    flat = []
    total_ctx_per_sample = defaultdict(int)

    for sid, sample in enumerate(samples):
        negative_sample=True
        for paragraph in sample.get("paragraphs", []):
            if supporting_only and not paragraph.get("is_supporting", False):
                continue
            negative_sample=False
            flat.append({
                "sample_id": sid,
                "ctx_id": paragraph.get("idx", len(flat)),
                "title": paragraph.get("title", ""),
                "sentences": paragraph.get("paragraph_text", ""),
            })
            total_ctx_per_sample[sid] += 1
        

        if negative_sample:
            for i in range(1):
                paragraph=sample.get("paragraphs", [])[i]
                flat.append({
                    "sample_id": sid,
                    "ctx_id": paragraph.get("idx", len(flat)),
                    "title": paragraph.get("title", ""),
                    "sentences": paragraph.get("paragraph_text", ""),
                })
                total_ctx_per_sample[sid] += 1

    return flat, total_ctx_per_sample

# ============================================================
# Main extraction loop (context-level batching + retry)
# ============================================================

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_triples_sequentially(samples, args, output_path):

    if args.dataset == "2wiki" or args.dataset=="hotpot":
        if args.dataset == "hotpot":
            samples = filter_sample_hotpot(samples)
            print(f"Filtered {len(samples)} samples for hotpot")

        flat_ctxs, total_ctx_per_sample = flatten_samples(
            samples=samples,
            supporting_only=args.supporting_only,
        )
    elif args.dataset == "musique":
        flat_ctxs, total_ctx_per_sample = flatten_samples_musique(
            samples=samples,
            supporting_only=args.supporting_only,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Total {len(flat_ctxs)} contexts to process")

    # sample_id -> { ctx_id -> triples }
    results = defaultdict(dict)

    # retry queue is context-level
    retry_queue = flat_ctxs
    
    written_samples = set()

    for round_id in range(args.retries + 1):
        if not retry_queue:
            break
        print(f"Round {round_id + 1}: {len(retry_queue)} contexts to retry")

        next_retry = []

        for i in tqdm(
            range(0, len(retry_queue), args.batch_size),
            desc=f"Round {round_id + 1}",
            unit="batch",
        ):
            batch = retry_queue[i:i + args.batch_size]

            prompts = [
                build_ctx_prompt(c["title"], c["sentences"])
                for c in batch
            ]

            try:
                outputs = call_batch(
                    endpoint=args.endpoint,
                    model=args.model,
                    prompts=prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            except Exception:
                # entire batch failed → retry all contexts
                next_retry.extend(batch)
                continue

            # -------- parse & collect (context-level) --------
            touched_sample_ids = set()

            for ctx, out in zip(batch, outputs):
                triples = parse_typed_triples(out)
                if triples:
                    results[ctx["sample_id"]][ctx["ctx_id"]] = triples
                    touched_sample_ids.add(ctx["sample_id"])
                else:
                    next_retry.append(ctx)

            # -------- ⭐ sample-level safe write-back ⭐ --------
            for sid in touched_sample_ids:
                if sid in written_samples:
                    continue

                # 核心判定：该 sample 的所有 ctx 是否都已完成
                if len(results.get(sid, {})) < total_ctx_per_sample[sid]:
                    continue  # 还没齐，不能写

                sample = samples[sid]
                new_ctx = []
                if args.dataset=="2wiki" or args.dataset=="hotpot":
                    for cid, (title, sentences) in enumerate(sample["context"]):
                        # 此时一定存在，不会是 []
                        new_ctx.append({
                            "title": title,
                            "sentences": sentences,
                            "triple_list": results.get(sid, {}).get(cid, []),
                        })

                    sample["context"] = new_ctx
                elif args.dataset=="musique":
                    for cid, para in enumerate(sample["paragraphs"]):
                        para["triple_list"] = results.get(sid, {}).get(cid, [])

                append_jsonl(output_path, sample)
                written_samples.add(sid)
                results.pop(sid, None)

        retry_queue = next_retry

        if args.sleep > 0:
            time.sleep(args.sleep)

    # 可选：打印未完成的 sample（用于 debug / 统计）
    unfinished = [
        sid for sid in range(len(samples))
        if sid not in written_samples
    ]
    if unfinished:
        print(
            f"⚠️ {len(unfinished)} samples not fully extracted "
            f"(contexts missing after retries)"
        )


# ============================================================
# Merge context-level results back to samples
# ============================================================

def merge_back(samples, results):
    for sid, s in enumerate(samples):
        new_ctx = []
        for cid, (title, sentences) in enumerate(s["context"]):
            new_ctx.append({
                "title": title,
                "sentences": sentences,
                "triple_list": results.get(sid, {}).get(cid, []),
            })
        s["context"] = new_ctx
    return samples

# ============================================================
# Buffered JSONL writer
# ============================================================

def write_jsonl(path, samples, buf_size=100):
    with open(path, "w", encoding="utf-8") as f:
        buf = []
        for s in samples:
            buf.append(json.dumps(s, ensure_ascii=False))
            if len(buf) >= buf_size:
                f.write("\n".join(buf) + "\n")
                buf.clear()
        if buf:
            f.write("\n".join(buf) + "\n")

# ============================================================
# Entry
# ============================================================

def main(args):
    if args.input.endswith(".jsonl"):
        samples = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
    elif args.input.endswith(".json"):
        with open(args.input, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        raise ValueError(f"Unknown input file format: {args.input}")

    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"✅ Loaded {len(samples)} samples")

    # ⭐ 先清空输出文件
    open(args.output, "w").close()

    extract_triples_sequentially(
        samples=samples,
        args=args,
        output_path=args.output,
    )

    print(f"✅ Done. Output written incrementally to {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("2Wiki triple extractor (optimized v3)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--endpoint", default="http://127.0.0.1:7000/v1/completions")
    ap.add_argument("--model", default="deepseek")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument(
        "--supporting-only",
        action="store_true",
        help="Only extract triples for supporting_facts contexts (Hotpot-style)",
    )

    args = ap.parse_args()

    main(args)
