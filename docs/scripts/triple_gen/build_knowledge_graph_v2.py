#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DAG-KV knowledge extraction script (DeepSeek-V3 friendly).

This script calls an OpenAI-compatible /v1/completions endpoint to extract
bidirectional KV pairs (key | value) from each context paragraph.

It supports 2Wiki/HotpotQA-style samples (sample['context'] = [[title,[sentences]],...])
(and optionally Hotpot 'bridge' filtering), plus a best-effort Musique format.

Output:
- For 2wiki/hotpot: sample['context'] becomes list of dicts {title, sentences, kv_list}
- For musique: each sample['paragraphs'][i]['kv_list'] is filled

The prompt is designed for DeepSeek-V3 but works with any instruction-following LLM.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import requests
from tqdm import tqdm
import random
import os
import concurrent.futures as cf


# high recall
PROMPT_PREFIX_BIDIRECTIONAL_V2 = ("""
You are a knowledge graph extraction system optimized for MAXIMUM RECALL.

Your goal is to extract as many valid entities and explicit relations as possible.

============================================================
I. ENTITY EXTRACTION
============================================================

Identify ALL named entities in the sentences, including:
   - books
   - series
   - people
   - characters
   - species
   - organizations
   - locations

Entities inside clauses MUST be extracted.

============================================================
II. RELATION EXTRACTION (entity → entity)
============================================================

1. Extract ALL explicitly stated relations.
2. Do NOT merge multiple entities into one node.
3. Decompose complex clauses into multiple simple relations.

Example:
"Aldrea and Dak Hamee tried to save their world"
→ (Aldrea, tried to save, their world)
→ (Dak Hamee, tried to save, their world)

============================================================
III. ATTRIBUTE EXTRACTION (entity → value)
============================================================

use ATTRIBUTE triples for literal facts such as:
   - dates
   - numbers
   - rankings
   - descriptive phrases that are not entities

Example:
(Albert Einstein, date of birth, 14 March 1879)

============================================================
IV. KV GENERATION RULES (BIDIRECTIONAL)
============================================================

For EACH extracted fact, generate TWO key/value pairs.

A. For attribute fact (entity, attribute, value):

1) Forward KV:
   key_string = natural_forward_string(entity, attribute)
   value_string = value

2) Reverse KV:
   key_string = natural_reverse_string(value, attribute)
   value_string = entity

B. For relation fact (entity1, relation, entity2):

1) Forward KV:
   key_string = natural_forward_string(entity1, relation)
   value_string = entity2

2) Reverse KV:
   key_string = natural_reverse_string(entity2, relation)
   value_string = entity1

------------------------------------------------------------
Reverse Equivalence Constraint
------------------------------------------------------------

natural_reverse_string MUST:
- Be a logically equivalent restatement of the SAME fact
- Use neutral wording if needed

If strict equivalence cannot be achieved, use a safe neutral reverse phrasing.

------------------------------------------------------------
Examples
------------------------------------------------------------

Attribute fact:
(India, location, South Asia)

Output:
('the location of India', 'South Asia')
('South Asia is the location of', 'India')

Relation fact:
(Bill Gates, founded, Microsoft)

Output:
('Bill Gates founded', 'Microsoft')
('Microsoft is founded by', 'Bill Gates')

============================================================
V. RELATION-DERIVED ATTRIBUTE
============================================================

For each relation (entity1, relation, entity2), derive ONE equivalent attribute fact (entity1, attribute_alias, entity2)

Rules:
- The derived attribute_alias must not change meaning.
- It must not introduce new type information.
- It must be interchangeable with the relation in question-answer form.

For each valid derived attribute fact, generate TWO additional KV pairs:

1) Forward:
   key = natural_forward_string(entity1, attribute_alias)
   value = entity2

2) Reverse:
   key = natural_reverse_string(entity2, attribute_alias)
   value = entity1

------------------------------------------------------------
Example
------------------------------------------------------------

Relation:
(Albert Einstein, was born in, Ulm)

Derived attribute:
(Albert Einstein, birthplace, Ulm)

Additional KV:
('the birthplace of Albert Einstein', 'Ulm')
('the person born in Ulm', 'Albert Einstein')

============================================================
VI. OUTPUT FORMAT (STRICT BLOCKS)
============================================================

Output plain text only.
Do NOT output JSON.
Do NOT output explanations.

For EACH extracted fact, output EXACTLY ONE BLOCK.

A BLOCK has:
1) one TRIPLE line (typed triple)
2) then KV lines (2 lines for ATTRIBUTE fact; 4 lines for RELATION fact)
3) then a separator line: ----

------------------------------------------------------------
TRIPLE LINE FORMAT
------------------------------------------------------------

ATTRIBUTE triple:
TRIPLE | ATTRIBUTE | entity | attribute | value

RELATION triple:
TRIPLE | RELATION | entity1 | relation | entity2 | attribute_alias

------------------------------------------------------------
KV LINES FORMAT
------------------------------------------------------------

Each KV line:
KV | key_string | value_string

For ATTRIBUTE fact: output EXACTLY 2 KV lines
- KV | natural_forward_string(entity, attribute) | value
- KV | natural_reverse_string(value, attribute) | entity

For RELATION fact : output EXACTLY 4 KV lines
- KV | natural_forward_string(entity1, relation) | entity2
- KV | natural_reverse_string(entity2, relation) | entity1
- KV | natural_forward_string(entity1, attribute_alias) | entity2
- KV | natural_reverse_string(entity2, attribute_alias) | entity1

------------------------------------------------------------
BLOCK SEPARATOR
------------------------------------------------------------

After finishing the KV lines for this fact, output:
----

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

- The TRIPLE line and the KV lines MUST describe the SAME fact.
- Keep surface names exactly as written.
- Do NOT invent facts.
- Do NOT merge contexts.

============================================================
Convert the following context:
"""
)
# ============================================================
# Prompt builder
# context format: [title, [sentences]]
# ============================================================

def build_ctx_prompt(title: str, sentences: List[str]) -> str:
    ctx = [title, sentences]
    # separators ensure prefix-cache stability
    return PROMPT_PREFIX_BIDIRECTIONAL_V2 + json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))


# ============================================================
# OpenAI-compatible batch call (/v1/completions)
# ============================================================

def call_batch(
    endpoint: str,
    model: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> List[str]:
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["Convert the following context:"]
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    outs: List[str] = []
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
# Parse KV lines (single-context)
# ============================================================

_QUOTE_WRAP_RE = re.compile(r"^\s*[\(\[]?\s*['\"]?(.*?)['\"]?\s*[\)\]]?\s*$")


def _strip_wrappers(s: str) -> str:
    """Remove surrounding quotes / brackets commonly emitted by LLMs."""
    s = s.strip()
    m = _QUOTE_WRAP_RE.match(s)
    return (m.group(1) if m else s).strip()


def parse_blocks(text: str) -> List[Dict[str, Any]]:
    """Parse LLM output for ONE context.
    Expect repeated blocks:
      TRIPLE | ATTRIBUTE | entity | attribute | value
      KV | key_string | value_string
      KV | key_string | value_string
      ----
    or:
      TRIPLE | RELATION | entity1 | relation | entity2 | attribute_alias
      KV ... (4 lines)
      ----
    """
    if not text:
        return []

    triples: List[Dict[str, Any]] = []

    cur_triple: Dict[str, Any] | None = None
    cur_kvs: List[Dict[str, str]] = []

    def flush():
        nonlocal cur_triple, cur_kvs
        if cur_triple is None:
            cur_kvs = []
            return
        cur_triple["kv_lists"] = cur_kvs
        triples.append(cur_triple)
        cur_triple = None
        cur_kvs = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "----":
            flush()
            continue

        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue

        tag = parts[0].upper()

        # ---------- TRIPLE ----------
        if tag == "TRIPLE":
            # start a new triple; if previous block didn't end with ----, flush it
            if cur_triple is not None:
                flush()

            if len(parts) < 2:
                continue

            ttype = parts[1].upper()
            if ttype == "ATTRIBUTE":
                # TRIPLE | ATTRIBUTE | entity | attribute | value
                if len(parts) < 5:
                    continue
                entity = parts[2]
                attr = parts[3]
                value = parts[4]
                if not entity or not attr or not value:
                    continue
                cur_triple = {
                    "type": "ATTRIBUTE",
                    "name": entity,
                    "description_type": attr,
                    "description": value,
                    "attribute_desc_alias": None,
                }
                cur_kvs = []
                continue

            if ttype == "RELATION":
                # TRIPLE | RELATION | entity1 | relation | entity2 | attribute_alias
                if len(parts) < 6:
                    continue
                e1 = parts[2]
                rel = parts[3]
                e2 = parts[4]
                alias = parts[5] if len(parts) >= 6 else ""
                if not e1 or not rel or not e2:
                    continue
                cur_triple = {
                    "type": "RELATION",
                    "name": e1,
                    "description_type": rel,
                    "description": e2,
                    "attribute_desc_alias": (alias.strip() if alias and alias.strip() else None),
                }
                cur_kvs = []
                continue

            # unknown triple type
            continue

        # ---------- KV ----------
        if tag == "KV":
            # KV | key_string | value_string
            if cur_triple is None:
                continue
            if len(parts) < 3:
                continue
            k = _strip_wrappers(parts[1])
            v = _strip_wrappers(parts[2])
            if not k or not v:
                continue
            cur_kvs.append({"key_string": k, "value_string": v})
            continue

        # ignore other lines

    # flush tail if missing separator
    if cur_triple is not None:
        flush()

    # de-dup kv within each triple (keep order)
    # for tr in triples:
    #     seen = set()
    #     deduped = []
    #     for kv in tr.get("kv_lists", []):
    #         t = (kv["key_string"], kv["value_string"])
    #         if t in seen:
    #             continue
    #         seen.add(t)
    #         deduped.append(kv)
    #     tr["kv_lists"] = deduped

    return triples

# def parse_kvs(text: str) -> List[Dict[str, str]]:
#     """Parse output for ONE context.

#     Expected per-line format: key | value
#     We ignore malformed lines.
#     """
#     if not text:
#         return []

#     kvs: List[Dict[str, str]] = []
#     for raw_line in text.splitlines():
#         line = raw_line.strip()
#         if not line:
#             continue
#         if "|" not in line:
#             continue
#         k, v = line.split("|", 1)
#         key = _strip_wrappers(k)
#         value = _strip_wrappers(v)
#         if not key or not value:
#             continue
#         kvs.append({"key": key, "value": value})

#     # de-dup (keep order)
#     seen = set()
#     deduped = []
#     for kv in kvs:
#         t = (kv["key"], kv["value"])
#         if t in seen:
#             continue
#         seen.add(t)
#         deduped.append(kv)
#     return deduped


# ============================================================
# Dataset flattening helpers
# ============================================================

def _build_supporting_map(sample: Dict[str, Any]) -> Dict[str, set]:
    """Hotpot/2Wiki style: sample['supporting_facts'] = [[title, sent_id], ...]."""
    m: Dict[str, set] = defaultdict(set)
    for item in sample.get("supporting_facts", []) or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title, sid = item[0], item[1]
        if isinstance(title, str) and isinstance(sid, int):
            m[title].add(sid)
    return m


def flatten_samples(samples: List[Dict[str, Any]], supporting_only: bool = False) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """2Wiki/Hotpot: sample['context'] is list of [title, [sentences]]."""
    flat: List[Dict[str, Any]] = []
    total_ctx_per_sample: Dict[int, int] = {}

    for sid, s in enumerate(samples):
        supp_map = _build_supporting_map(s) if supporting_only else None

        ctxs = s.get("context", [])
        kept = []
        for cid, ctx in enumerate(ctxs):
            if not isinstance(ctx, (list, tuple)) or len(ctx) != 2:
                continue
            title, sentences = ctx[0], ctx[1]
            if not isinstance(title, str) or not isinstance(sentences, list):
                continue

            if supporting_only and supp_map is not None:
                idxs = sorted(list(supp_map.get(title, set())))
                if not idxs:
                    continue
                sentences = [sentences[i] for i in idxs if 0 <= i < len(sentences)]
                if not sentences:
                    continue

            kept.append((title, sentences, cid))

        total_ctx_per_sample[sid] = len(kept)
        for new_cid, (title, sentences, orig_cid) in enumerate(kept):
            flat.append({
                "sample_id": sid,
                "ctx_id": orig_cid,
                "title": title,
                "sentences": sentences,
            })

    return flat, total_ctx_per_sample


def flatten_samples_musique(samples: List[Dict[str, Any]], supporting_only: bool = False) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """Best-effort Musique: sample['paragraphs'] is list of dict.

    We try to use fields:
    - title (str)
    - sentences (list[str]) OR paragraph_text/text (str)

    supporting_only: if sample has supporting_facts compatible with (title, sent_id), we filter.
    """
    flat: List[Dict[str, Any]] = []
    total_ctx_per_sample: Dict[int, int] = {}
    answerable_cnt=0

    for sid, s in enumerate(samples):
        paras = s.get("paragraphs", []) or []
        if s.get("answerable"):
            answerable_cnt+=1
        kept = []
        for cid, p in enumerate(paras):
            if not isinstance(p, dict):
                continue
            if supporting_only and not p.get("is_supporting"):
                continue
            title = p.get("title") or p.get("heading") or p.get("entity") or f"para_{cid}"
            if not isinstance(title, str):
                title = str(title)

            sentences = p.get("sentences")
            if isinstance(sentences, list) and all(isinstance(x, str) for x in sentences):
                pass
            else:
                text = p.get("paragraph_text") or p.get("text") or p.get("context")
                if isinstance(text, list):
                    # already tokenized
                    sentences = [str(x) for x in text]
                elif isinstance(text, str):
                    sentences = [text]
                else:
                    sentences = []

            # if supporting_only and supp_map is not None:
            #     idxs = sorted(list(supp_map.get(title, set())))
            #     if idxs and sentences:
            #         sentences = [sentences[i] for i in idxs if 0 <= i < len(sentences)]
            #     if no idxs, we keep paragraph as-is (musique often lacks title alignment)

            if not sentences:
                continue
            kept.append((title, sentences, cid))

        total_ctx_per_sample[sid] = len(kept)
        for (title, sentences, orig_cid) in kept:
            flat.append({
                "sample_id": sid,
                "ctx_id": orig_cid,
                "title": title,
                "sentences": sentences,
            })
    print(f"Only extract answerable sample. Answerable ratio: {answerable_cnt/len(samples)}")
    return flat, total_ctx_per_sample


def filter_sample_hotpot(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in samples if s.get("type") == "bridge"]

def filter_sample_2wiki(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in samples if s.get("type") == "compositional"]

# ============================================================
# Incremental JSONL writer
# ============================================================

def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# Main extraction loop (context-level batching + retry)
# ============================================================

def extract_kvs_sequentially(samples: List[Dict[str, Any]], args, output_path: str, written_samples: Dict[str, Dict[str, Any]]):
    if args.dataset in {"2wiki", "hotpot"}:
        if args.dataset == "hotpot" and args.hotpot_bridge_only:
            samples = filter_sample_hotpot(samples)
            print(f"Filtered {len(samples)} samples for hotpot (bridge only)")
        if args.dataset == "2wiki" and args.compositional_only:
            samples = filter_sample_2wiki(samples)
            print(f"Filtered {len(samples)} samples for 2wiki (compositional only)")

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

    print(f"{len(flat_ctxs)} contexts to process")


    # sample_id -> { ctx_id -> kv_list }
    results: Dict[int, Dict[int, List[Dict[str, str]]]] = defaultdict(dict)

    retry_queue = flat_ctxs
    written_samples = set()

    for round_id in range(args.retries + 1):
        if not retry_queue:
            break
        print(f"Round {round_id + 1}: {len(retry_queue)} contexts to retry")

        next_retry = []
        # -------------------------------
        # Concurrent pipelined mini-batches
        # -------------------------------
        # mini-batch size derived from batch_size and num_mini_batch
        num_mini = max(1, int(args.concurrent_requests))
        mini_bs = max(1, args.batch_size // num_mini)

        # Slice retry_queue into mini-batches
        mini_batches = [retry_queue[j : j + mini_bs] for j in range(0, len(retry_queue), mini_bs)]

        # A tiny helper so we can submit to thread pool
        def _run_one(mini_batch):
            prompts = [build_ctx_prompt(c["title"], c["sentences"]) for c in mini_batch]
            outs = call_batch(
                endpoint=args.endpoint,
                model=args.model,
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            return mini_batch, outs

        max_inflight = max(1, int(args.concurrent_requests))
        pbar = tqdm(total=len(mini_batches), desc=f"Round {round_id + 1}", unit="mini_batch")

        idx = 0
        inflight = {}

        with cf.ThreadPoolExecutor(max_workers=max_inflight) as ex:
            # prime the pipeline
            while idx < len(mini_batches) and len(inflight) < max_inflight:
                mb = mini_batches[idx]
                fut = ex.submit(_run_one, mb)
                inflight[fut] = mb
                idx += 1

            # drain + keep pipeline full
            while inflight:
                done, _ = cf.wait(inflight.keys(), return_when=cf.FIRST_COMPLETED)

                for fut in done:
                    mb = inflight.pop(fut)

                    try:
                        batch, outputs = fut.result()
                    except Exception:
                        # request failed: retry all contexts in this mini-batch
                        next_retry.extend(mb)
                        pbar.update(1)
                        # refill pipeline
                        if idx < len(mini_batches):
                            nb = mini_batches[idx]
                            nfut = ex.submit(_run_one, nb)
                            inflight[nfut] = nb
                            idx += 1
                        continue

                    touched_sample_ids = set()

                    for ctx, out in zip(batch, outputs):
                        triple_list = parse_blocks(out)
                        if triple_list:
                            results[ctx["sample_id"]][ctx["ctx_id"]] = triple_list
                            touched_sample_ids.add(ctx["sample_id"])
                        else:
                            next_retry.append(ctx)

                    # sample-level safe write-back: only write when all contexts for that sample are done
                    for sid in touched_sample_ids:
                        if sid in written_samples:
                            continue
                        if len(results.get(sid, {})) < total_ctx_per_sample.get(sid, 0):
                            continue

                        sample = samples[sid]
                        if args.dataset in {"2wiki", "hotpot"}:
                            new_ctx = []
                            for cid, (title, sentences) in enumerate(sample.get("context", [])):
                                new_ctx.append({
                                    "title": title,
                                    "sentences": sentences,
                                    "triple_list": results.get(sid, {}).get(cid, [])
                                })
                            sample["context"] = new_ctx
                        elif args.dataset == "musique":
                            for cid, para in enumerate(sample.get("paragraphs", []) or []):
                                if isinstance(para, dict):
                                    para["triple_list"] = results.get(sid, {}).get(cid, [])

                        append_jsonl(output_path, sample)
                        written_samples.add(sid)
                        results.pop(sid, None)

                    pbar.update(1)

                    # refill pipeline immediately
                    if idx < len(mini_batches):
                        nb = mini_batches[idx]
                        nfut = ex.submit(_run_one, nb)
                        inflight[nfut] = nb
                        idx += 1

        pbar.close()

        retry_queue = next_retry
        if args.sleep > 0:
            time.sleep(args.sleep)

    unfinished = [sid for sid in range(len(samples)) if sid not in written_samples]
    if unfinished:
        print(
            f"⚠️ {len(unfinished)} samples not fully extracted (contexts missing after retries)"
        )


# ============================================================
# Entry
# ============================================================

def main(args):
    if args.input.endswith(".jsonl"):
        samples = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    elif args.input.endswith(".json"):
        with open(args.input, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        raise ValueError(f"Unknown input file format: {args.input}")
    print(f"Input file has {len(samples)} samples")

    # 如果output已经存在，则读取成map，key是"_id"字段
    written_samples = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                written_samples[sample["_id"]] = sample
        print(f"Output file already has {len(written_samples)} samples")
        

    if args.limit > 0:
        if args.random_seed is not None:
            random.seed(args.random_seed)
            samples = random.sample(samples, args.limit)
        else:
            samples = samples[: args.limit]

    # 过滤出不在written_samples中的样本
    samples = [s for s in samples if s["_id"] not in written_samples]
    print(f"After filtering, {len(samples)} samples to process")

    # clear output file
    # open(args.output, "w", encoding="utf-8").close()

    extract_kvs_sequentially(samples=samples, args=args, output_path=args.output, written_samples=written_samples)

    print(f"✅ Done. Output written incrementally to {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("DAG-KV extractor (bidirectional KV)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", default="2wiki", choices=["2wiki", "hotpot", "musique"])

    # OpenAI-compatible completion endpoint
    ap.add_argument("--endpoint", default="http://127.0.0.1:7000/v1/completions")
    ap.add_argument("--model", default="deepseek-v3")

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)

    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--random-seed", type=int, default=None)

    ap.add_argument("--supporting-only", action="store_true", help="Only extract from supporting sentences when available")
    ap.add_argument("--hotpot-bridge-only", action="store_true", help="For hotpot, only keep 'bridge' samples")
    ap.add_argument("--compositional-only", action="store_true", help="For 2wiki, only keep 'compositional' samples")

    ap.add_argument("--concurrent-requests", type=int, default=1, help="Number of concurrent HTTP requests in flight")
    
    main(ap.parse_args())
