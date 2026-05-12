#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_knowledge_graph_v5.py

V5 graph-first KG extraction pipeline for QA datasets using OpenAI-compatible APIs.

Stage 1:
    Build an evidence-grounded flat graph skeleton from the selected context: nodes + typed triples.
Stage 2:
    Optional answer-aware graph revision / repair over the stage-1 graph, preserving triple type.
Final KV:
    Deterministically generate forward/reverse KV pairs from typed triples.

Key features:
- Async concurrent requests with asyncio + httpx
- Structured Outputs via Responses API or chat.completions JSON schema
- Resume from output jsonl and optional per-stage cache
- Local validation + deduplication
- Incremental JSONL writing

Designed for samples shaped like HotpotQA / 2Wiki / MuSiQue style examples:
{
  "_id": "...",
  "question": "...",
  "answer": "...",
  "supporting_facts": [[title, sent_id], ...],
  "context": [
      {"title": "...", "sentences": [...], "triple_list": [...]},
      ...
  ]
}

Environment:
    export OFOXAI_API_KEY=...
    export OPENAI_BASE_URL=https://api.ofox.ai/v1

Example:
    python3 build_knowledge_graph_v5.py \
        --input hotpot_train.jsonl \
        --output hotpot_train_triples.jsonl \
        --model openai/gpt-5.2 \
        --api-mode chat \
        --concurrency 8 \
        --stage-cache-dir ./stage_cache \
        --answer-aware
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import httpx
except ImportError:
    httpx = None


# ============================================================
# IO
# ============================================================


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    raise ValueError("Unsupported JSON root format. Expect list or {data:[...]}.")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Helpers
# ============================================================

_SPACE_RE = re.compile(r"\s+")


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def safe_sample_id(sample: Dict[str, Any], idx: int) -> str:
    sid = norm_text(sample.get("_id", "")) or norm_text(sample.get("id", ""))
    if sid:
        return sid
    return f"sample_{idx:08d}"


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_progress_message(
    *,
    stats: Dict[str, Any],
    failure_stats: Dict[str, int],
    last_sample_id: str,
) -> str:
    done = int(stats.get("done", 0))
    total = int(stats.get("total", 0))
    started_at = float(stats.get("started_at", time.time()))
    elapsed_s = max(0.0, time.time() - started_at)
    rate = (done / elapsed_s) if elapsed_s > 0 and done > 0 else 0.0
    remaining = max(0, total - done)
    eta_s = (remaining / rate) if rate > 0 else 0.0
    pct = (100.0 * done / total) if total > 0 else 100.0
    return (
        f"[{now_ts()}] progress {done}/{total} ({pct:.1f}%) "
        f"ok={stats['ok']} err={stats['err']} retried={stats['retried']} "
        f"dropped={stats['dropped']} rate={rate:.2f}/s "
        f"elapsed={format_duration(elapsed_s)} eta={format_duration(eta_s)} "
        f"last={last_sample_id} failures={dict(sorted(failure_stats.items()))}"
    )


def make_request_id(stage: str, sample_id: str) -> str:
    return f"{stage}-{sample_id}-{uuid.uuid4().hex[:12]}"


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def infer_triple_type_from_tail_kind(tail_kind: Any) -> str:
    kind = norm_text(tail_kind)
    if kind == "entity":
        return "RELATION"
    if kind == "literal":
        return "ATTRIBUTE"
    return ""


def get_triple_type(tri: Dict[str, Any]) -> str:
    triple_type = norm_text(tri.get("triple_type", ""))
    if triple_type in {"ATTRIBUTE", "RELATION"}:
        return triple_type
    return infer_triple_type_from_tail_kind(tri.get("tail_kind", ""))


def programmatic_forward_kv(tri: Dict[str, Any]) -> Dict[str, str]:
    triple_type = get_triple_type(tri)
    head = norm_text(tri.get("head", tri.get("name", "")))
    rel = norm_text(tri.get("relation", tri.get("description_type", "")))
    tail = norm_text(tri.get("tail", tri.get("description", "")))
    if triple_type == "ATTRIBUTE":
        return {
            "key_string": f"the {rel} of {head} is",
            "value_string": tail,
        }
    return {
        "key_string": f"{head} {rel}",
        "value_string": tail,
    }


def programmatic_reverse_kv(tri: Dict[str, Any]) -> Dict[str, str]:
    triple_type = get_triple_type(tri)
    head = norm_text(tri.get("head", tri.get("name", "")))
    rel = norm_text(tri.get("relation", tri.get("description_type", "")))
    tail = norm_text(tri.get("tail", tri.get("description", "")))
    if triple_type == "ATTRIBUTE":
        return {
            "key_string": f"{tail} is the {rel} of",
            "value_string": head,
        }
    return {
        # "key_string": f'the entity that is related to {tail} by "{rel}" is',
        "key_string": f'the entity that {rel} {tail} is',
        "value_string": head,
    }


# ============================================================
# Schemas for structured outputs
# ============================================================

STAGE1_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "_id": {"type": "string"},
        "entity_list": {
            "type": "array",
            "items": {"type": "string"}
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "head": {"type": "string"},
                    "relation": {"type": "string"},
                    "tail": {"type": "string"},
                    "triple_type": {
                        "type": "string",
                        "enum": ["ATTRIBUTE", "RELATION"]
                    }
                },
                "required": ["head", "relation", "tail", "triple_type"]
            }
        }
    },
    "required": ["_id", "entity_list", "triples"]
}

# ============================================================
# Prompts
# ============================================================

STAGE1_SYSTEM_PROMPT = """You are an information extraction system for building a structured knowledge graph from the given context.

Your task is to extract a flat, evidence-grounded graph with:
1. a clean list of graph-worthy entities,
2. all entity-to-entity relation triples,
3. all entity-to-value attribute triples.

You must follow the definitions and rules below exactly.

==================================================
GOAL
==================================================

Build a compact but useful knowledge graph from the given context.

The graph should contain:
- entity_list: reusable graph-worthy entities
- triples: a flat list of triples
- each triple must have:
  - head
  - relation
  - tail
  - triple_type

Do NOT output free-form explanations.
Do NOT output any text outside the required JSON object.

==================================================
REQUIRED WORKFLOW
==================================================

Follow this workflow strictly:

Step 1. Extract all graph-worthy entities from the context.
Step 2. Extract all entity-to-entity facts as RELATION triples.
Step 3. Extract all entity-to-value facts as ATTRIBUTE triples.

Important:
Always determine triple_type from whether the tail is an entity or a value.

- entity -> entity = RELATION
- entity -> value = ATTRIBUTE

Do NOT decide based only on wording style.
Do NOT decide based only on whether the relation sounds abstract.
The key question is whether the tail should be a reusable graph node.

==================================================
CORE DEFINITIONS
==================================================

1. GRAPH-WORTHY ENTITY

A graph-worthy entity is an object that:
- can be referred to independently,
- has relatively clear identity in the context,
- is worth becoming a reusable graph node,
- may participate in multiple facts or reasoning chains.

Typical graph-worthy entities include:
- person names
- locations
- organizations
- countries / cities / regions
- works (books, films, albums, songs)
- events, movements, wars, marches, campaigns
- species / taxa / named biological entities
- named awards, laws, institutions, titles
- other clearly referable named or uniquely identifiable objects

Important:
A graph-worthy entity is NOT limited to standard named entities.
If something is not a person/location/organization but is still a clearly referable object that should act as a node in a graph, include it.

A tail should be treated as an entity if it can itself serve as a reusable node and plausibly connect to other triples.

2. VALUE

A value is NOT a reusable graph node.
A value is usually:
- a number
- a date
- a year
- a time span
- a measurement
- a quantity
- a percentage
- a short descriptive phrase
- a category label
- a literal value
- a property value that is not worth becoming its own graph node

Examples of values:
- 1930
- 12 March 1930
- 4
- 17 km
- science fantasy
- first person
- annual award

3. RELATION TRIPLE

A relation triple is:
- head = graph-worthy entity
- relation = semantic relation
- tail = graph-worthy entity
- triple_type = RELATION

Form:
(head entity, relation, tail entity)

Use a relation triple only when the tail should also be a graph node.

4. ATTRIBUTE TRIPLE

An attribute triple is:
- head = graph-worthy entity
- relation = attribute / property name
- tail = non-entity value
- triple_type = ATTRIBUTE

Form:
(head entity, attribute, value)

Use an attribute triple when the right side is not worth representing as a graph node.

==================================================
ENTITY SELECTION RULES
==================================================

Extract entities conservatively but usefully.

Include an entity if:
- it is clearly referred to in the context, and
- representing it as a node would help preserve structure or support multi-hop reasoning.

Do NOT include items that are only:
- raw numbers
- plain dates
- simple adjectives
- generic descriptors
- ordinary common nouns with no clear node identity
- value phrases that are better treated as attribute values

Do NOT include duplicate entities.
If multiple mentions refer to the same entity, merge them into one canonical entity.

If needed, use disambiguated names so that each entity is self-contained and clear.
Prefer a fuller name when the short name is ambiguous.

Use grounded surface strings from the context only.
Do not use IDs such as E1, E2, etc.

==================================================
TRIPLE EXTRACTION RULES
==================================================

A. RELATION TRIPLES

Extract all factual entity-to-entity relations that are explicitly supported by the context.

Requirements:
- both head and tail must appear in entity_list
- relation must be concise and faithful to the text
- relation should preserve the original fact, not over-abstract it
- relation must be a clean, self-contained relation phrase
- relation may be specific, but it should still read like a proper edge label rather than a sentence fragment
- do not use weak standalone prepositions such as "in", "on", "under", "of" as the full relation unless they are part of a stable phrase such as "located in" or "born on"
- do not use temporary reasoning phrases or bridge-only phrases such as "leads to"
- do not invent missing links

B. ATTRIBUTE TRIPLES

Extract all factual entity-to-value properties that are explicitly supported by the context.

Requirements:
- head must appear in entity_list
- tail must NOT be an entity from entity_list
- relation must be a concise attribute name
- relation must read naturally in the form: "the <relation> of <head> is <tail>"
- use attribute labels such as "publication date", "population", "height", "role", "series number", "start year"
- do NOT use event-style or clause-style relations such as "published on", "formed", "became part of", "served as", or other relations that do not fit the pattern "the <relation> of <head> is <tail>"
- tail should be a short, precise value span

==================================================
ATOMICITY RULES
==================================================

Every triple must be a minimal unit.

- If one phrase can be split into multiple entities, split it.
- If one fact contains multiple relations, split them into separate triples.
- If one attribute contains multiple values, split them into separate triples.
- Split conjunctions, coordinated phrases, and lists into separate triples when supported by the context.
- Do not compress multiple tails into one comma-joined tail when separate triples are possible.
- Do not use oversized tails when a shorter atomic tail is possible.
- Do not use sentence-like relations when a shorter reusable relation is possible.

==================================================
CANONICALIZATION RULES
==================================================

1. ENTITY NAMES
- Use a canonical name that is clear and unambiguous in context.
- Preserve the actual entity identity from the text.
- Do not replace entities with IDs.

2. RELATIONS
- Use concise natural-language relation names.
- Keep them semantically close to the source text.
- Do not make them overly generic like "is related to".
- Do not make them unnecessarily long.
- Do not use sentence fragments such as "calls target countries" or "is one of pioneers of" as relations.
- Prefer a clean relation phrase that expresses the semantic link clearly by itself.
- Use lowercase relation names.
- Use spaces only, with no underscores, camelCase, or unnecessary punctuation.
- For ATTRIBUTE triples specifically, relation must be an attribute name rather than an event phrase, and must fit: "the <relation> of <head> is <tail>".

3. VALUES
- Keep values short and precise.
- Do not add extra explanation.
- Do not convert values into full sentences.

==================================================
DEDUPLICATION RULES
==================================================

- Do not output duplicate entities.
- Do not output duplicate triples.
- If the same fact is repeated multiple times, keep one clean version.
- If multiple mentions express the same fact, merge them.

==================================================
STRICT CONSTRAINTS
==================================================

- Only use facts supported by the provided context.
- Do not use outside knowledge.
- Do not infer speculative facts.
- Do not output any triple whose evidence is not present in the context.
- Do not create a RELATION triple whose tail is not in entity_list.
- Do not place a graph-worthy entity into an ATTRIBUTE triple as the tail.
- Do not place a plain value into a RELATION triple as the tail.

==================================================
FINAL DECISION CHECK
==================================================

Before output, verify:

- entity_list contains reusable entities only, not plain values
- every RELATION triple is entity -> relation -> entity
- every ATTRIBUTE triple is entity -> attribute -> value
- every RELATION relation is a clean self-contained phrase, not a sentence fragment, weak preposition, or bridge-only phrase
- every ATTRIBUTE relation reads naturally as: "the <relation> of <head> is <tail>"
- every triple is explicit, grounded, deduplicated, and atomic

When uncertain whether the tail is an entity or a value, use this test:
Can the tail be independently referred to and reused as its own graph node in other triples?

- If yes, use RELATION.
- If no, use ATTRIBUTE.

==================================================
OUTPUT FORMAT
==================================================

Return valid JSON only and follow the schema exactly.
"""

# ============================================================
# OpenAI-compatible API client (async, raw HTTP)
# ============================================================

@dataclass
class OpenAIConfig:
    api_key: str
    api_base: str
    model: str
    timeout: float
    max_connections: int
    max_keepalive_connections: int
    max_retries: int
    retry_base: float
    retry_max: float
    store: bool
    verbosity: Optional[str]
    api_mode: str


class ResponsesAPIError(RuntimeError):
    pass


class OpenAIResponsesClient:
    def __init__(self, config: OpenAIConfig):
        if httpx is None:
            raise RuntimeError(
                "httpx is required for async HTTP transport in build_knowledge_graph_v5.py. "
                "Install it with `pip install httpx`."
            )
        self.config = config
        self._base_url = config.api_base.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"
        limits = httpx.Limits(
            max_connections=max(1, self.config.max_connections),
            max_keepalive_connections=max(1, self.config.max_keepalive_connections),
        )
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=self._headers,
            limits=limits,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema_name: str,
        schema: Dict[str, Any],
        request_id: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                data = await self._post_structured(
                    request_id=request_id,
                    payload=payload,
                    schema_name=schema_name,
                    schema=schema,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                )
                text = extract_text_from_api_response(data, self.config.api_mode)
                if not text:
                    raise ResponsesAPIError(
                        f"No output text found in response for request_id={request_id}."
                    )
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ResponsesAPIError(
                        f"Response text is not valid JSON for request_id={request_id}: {text[:1000]}"
                    ) from e
                return parsed
            except Exception as e:
                last_err = e
                if attempt >= self.config.max_retries:
                    break
                sleep_s = min(
                    self.config.retry_max,
                    self.config.retry_base * (2 ** (attempt - 1)) * (1.0 + random.random() * 0.2),
                )
                await asyncio.sleep(sleep_s)
        raise ResponsesAPIError(f"Request failed after retries: {last_err}")

    async def _post_structured(
        self,
        *,
        request_id: str,
        payload: Dict[str, Any],
        schema_name: str,
        schema: Dict[str, Any],
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.config.api_mode == "responses":
            return await self._post_responses(
                request_id=request_id,
                payload=self._build_responses_payload(payload, schema_name, schema),
            )
        if self.config.api_mode == "chat":
            return await self._post_chat_completions(
                request_id=request_id,
                payload=self._build_chat_payload(system_prompt, user_payload, schema_name, schema),
            )
        raise ResponsesAPIError(f"Unsupported api_mode={self.config.api_mode}")

    def _build_responses_payload(
        self,
        payload: Dict[str, Any],
        schema_name: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = dict(payload)
        out["store"] = self.config.store
        out["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        }
        if self.config.verbosity:
            out["text"]["verbosity"] = self.config.verbosity
        return out

    def _build_chat_payload(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema_name: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        return payload

    async def _post_responses(self, *, request_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = await self._client.post(
                f"{self._base_url}/responses",
                json=payload,
                headers={"X-Client-Request-Id": request_id},
            )
            resp.raise_for_status()
            body = resp.text
        except httpx.HTTPStatusError as e:
            body = e.response.text
            raise ResponsesAPIError(f"HTTP {e.response.status_code}: {body[:2000]}") from e
        except httpx.RequestError as e:
            raise ResponsesAPIError(f"Network error: {e}") from e

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ResponsesAPIError(
                f"Response body is not valid JSON for request_id={request_id}: {body[:1000]}"
            ) from e

    async def _post_chat_completions(self, *, request_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"X-Client-Request-Id": request_id},
            )
            resp.raise_for_status()
            body = resp.text
        except httpx.HTTPStatusError as e:
            body = e.response.text
            raise ResponsesAPIError(f"HTTP {e.response.status_code}: {body[:2000]}") from e
        except httpx.RequestError as e:
            raise ResponsesAPIError(f"Network error: {e}") from e

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ResponsesAPIError(
                f"Response body is not valid JSON for request_id={request_id}: {body[:1000]}"
            ) from e


def extract_text_from_response(data: Dict[str, Any]) -> str:
    # SDK helper output_text is not available here, so parse raw response.
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"]

    out_parts: List[str] = []
    for item in data.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            ctype = content.get("type")
            if ctype == "output_text" and isinstance(content.get("text"), str):
                out_parts.append(content["text"])
            elif ctype == "refusal":
                raise ResponsesAPIError(f"Model refusal: {content}")
    return "\n".join(p for p in out_parts if p).strip()


def extract_text_from_chat_completion(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    refusal = message.get("refusal")
    if refusal:
        raise ResponsesAPIError(f"Model refusal: {refusal}")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                out_parts.append(item["text"])
        return "\n".join(p for p in out_parts if p).strip()
    return ""


def extract_text_from_api_response(data: Dict[str, Any], api_mode: str) -> str:
    if api_mode == "responses":
        return extract_text_from_response(data)
    if api_mode == "chat":
        return extract_text_from_chat_completion(data)
    raise ResponsesAPIError(f"Unsupported api_mode={api_mode}")


# ============================================================
# Sample preparation
# ============================================================


def normalize_context_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, dict):
        raw_sentences = entry.get("sentences")
        if isinstance(raw_sentences, list):
            sentences = [norm_text(s) for s in raw_sentences]
        else:
            text = entry.get("paragraph_text") or entry.get("text") or entry.get("context")
            if isinstance(text, list):
                sentences = [norm_text(s) for s in text]
            elif isinstance(text, str):
                sentences = [norm_text(text)]
            else:
                sentences = []

        return {
            "title": norm_text(entry.get("title") or entry.get("heading") or entry.get("entity") or ""),
            "sentences": sentences,
            "triple_list": entry.get("triple_list", []) or [],
            "is_supporting": bool(entry.get("is_supporting", False)),
        }

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        title = norm_text(entry[0])
        raw_sentences = entry[1] if isinstance(entry[1], list) else []
        triple_list = entry[2] if len(entry) >= 3 and isinstance(entry[2], list) else []
        return {
            "title": title,
            "sentences": [norm_text(s) for s in raw_sentences],
            "triple_list": triple_list,
            "is_supporting": False,
        }

    return None


def iter_sample_context_entries(sample: Dict[str, Any]) -> Iterable[Any]:
    raw_context = sample.get("context")
    if isinstance(raw_context, list):
        return raw_context
    raw_paragraphs = sample.get("paragraphs")
    if isinstance(raw_paragraphs, list):
        return raw_paragraphs
    return []


def get_context_map(sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_para in iter_sample_context_entries(sample):
        para = normalize_context_entry(raw_para)
        if para is None:
            continue
        title = para["title"]
        if title:
            out[title] = para
    return out


def get_supporting_titles(sample: Dict[str, Any]) -> List[str]:
    titles: List[str] = []
    seen = set()
    for item in sample.get("supporting_facts", []) or []:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        title = norm_text(item[0])
        if title and title not in seen:
            titles.append(title)
            seen.add(title)
    if titles:
        return titles

    for raw_para in iter_sample_context_entries(sample):
        para = normalize_context_entry(raw_para)
        if para is None or not para.get("is_supporting"):
            continue
        title = para["title"]
        if title and title not in seen:
            titles.append(title)
            seen.add(title)
    return titles


def has_supporting_context(sample: Dict[str, Any]) -> bool:
    return len(get_supporting_titles(sample)) > 0


def build_stage1_input(
    sample: Dict[str, Any],
    *,
    include_question: bool,
    include_answer: bool,
    supporting_pages_only: bool,
) -> Dict[str, Any]:
    ctx_map = get_context_map(sample)
    supporting_titles = get_supporting_titles(sample)

    pages: List[Dict[str, Any]] = []
    if supporting_pages_only:
        for title in supporting_titles:
            para = ctx_map.get(title)
            if para is None:
                continue
            pages.append(
                {
                    "title": title,
                    "sentences": [norm_text(s) for s in (para.get("sentences", []) or [])],
                }
            )
    else:
        for raw_para in iter_sample_context_entries(sample):
            para = normalize_context_entry(raw_para)
            if para is None:
                continue
            pages.append(
                {
                    "title": para["title"],
                    "sentences": para["sentences"],
                }
            )

    payload: Dict[str, Any] = {
        "_id": safe_sample_id(sample, 0),
        "pages": pages,
    }
    if include_question:
        payload["question"] = norm_text(sample.get("question", ""))
    if include_answer:
        payload["answer"] = norm_text(sample.get("answer", ""))
    return payload


def build_stage2_question_decomposition(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    item_id = 0
    for item in sample.get("question_decomposition", []) or []:
        if not isinstance(item, dict):
            continue

        question = norm_text(item.get("question", ""))
        answer = norm_text(item.get("answer", ""))
        if not (question or answer):
            continue

        row: Dict[str, Any] = {
            f"sub_question_{item_id}": question,
            f"sub_answer_{item_id}": answer,
        }
        item_id += 1

        out.append(row)
    return out


def normalize_stage2_revision(stage2: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    if verbose:
        print(f"[stage2_revision] {stage2}")
    out = dedupe_stage1(stage2)
    out["answer_sufficient"] = bool(stage2.get("answer_sufficient", False))
    out["missing_links"] = [
        norm_text(x) for x in (stage2.get("missing_links", []) or []) if norm_text(x)
    ]
    out["revision_notes"] = [
        norm_text(x) for x in (stage2.get("revision_notes", []) or []) if norm_text(x)
    ]
    return out


def validate_stage2_revision(stage1: Dict[str, Any], stage2: Dict[str, Any], verbose: bool = False) -> None:
    validate_stage1(stage2)
    if norm_text(stage2.get("_id", "")) != norm_text(stage1.get("_id", "")):
        raise ValueError("stage2_revision._id does not match stage1._id")
    if not isinstance(stage2.get("answer_sufficient"), bool):
        raise ValueError("stage2_revision.answer_sufficient missing or invalid")
    if not isinstance(stage2.get("missing_links"), list):
        raise ValueError("stage2_revision.missing_links missing or invalid")
    if not isinstance(stage2.get("revision_notes"), list):
        raise ValueError("stage2_revision.revision_notes missing or invalid")




# ============================================================
# Validation and cleanup
# ============================================================
def _iter_stage_graph_triples(stage_graph: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for tri in stage_graph.get("triples", []) or []:
        if isinstance(tri, dict):
            yield tri


def normalize_relation_phrase(text: Any) -> str:
    s = norm_text(text).lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_ENTITY_CODE_RE = re.compile(r"^(?:E|T)\d+$", re.IGNORECASE)


def is_placeholder_entity_code(text: Any) -> bool:
    return bool(_ENTITY_CODE_RE.fullmatch(norm_text(text)))


def normalize_triple_type(text: Any) -> str:
    s = norm_text(text).upper()
    if s in {"ATTRIBUTE", "RELATION"}:
        return s
    return ""


def _normalize_stage_graph_triple(tri: Dict[str, Any]) -> Optional[Dict[str, str]]:
    head = norm_text(tri.get("head", tri.get("name", "")))
    relation = normalize_relation_phrase(tri.get("relation", tri.get("description_type", "")))
    tail = norm_text(tri.get("tail", tri.get("description", "")))
    triple_type = normalize_triple_type(tri.get("triple_type", tri.get("type", "")))
    if not triple_type:
        triple_type = get_triple_type(tri)
    if not (head and relation and tail and triple_type in {"ATTRIBUTE", "RELATION"}):
        return None
    if is_placeholder_entity_code(head):
        return None
    if triple_type == "RELATION" and is_placeholder_entity_code(tail):
        return None
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "triple_type": triple_type,
    }


def dedupe_stage1(stage1: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "_id": norm_text(stage1.get("_id", "")),
        "entity_list": [],
        "triples": [],
    }

    entity_seen = set()
    for ent in stage1.get("entity_list", []) or []:
        name = norm_text(ent)
        if name and name not in entity_seen and not is_placeholder_entity_code(name):
            out["entity_list"].append(name)
            entity_seen.add(name)

    seen = set()
    for tri in _iter_stage_graph_triples(stage1):
        tri2 = _normalize_stage_graph_triple(tri)
        if tri2 is None:
            continue
        sig = (tri2["head"], tri2["relation"], tri2["tail"], tri2["triple_type"])
        if sig in seen:
            continue
        seen.add(sig)
        out["triples"].append(tri2)
    return out


def dedupe_stage2_input_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for tri in triples:
        tri2 = _normalize_stage_graph_triple(tri)
        if tri2 is None:
            continue
        sig = (
            tri2["head"],
            tri2["relation"],
            tri2["tail"],
            tri2["triple_type"],
        )
        if sig not in seen:
            out.append(dict(tri2))
            seen.add(sig)
    return out


def validate_stage1(stage1: Dict[str, Any]) -> None:
    if not isinstance(stage1, dict):
        raise ValueError("stage1 must be a dict")
    if not isinstance(stage1.get("_id"), str):
        raise ValueError("stage1._id missing or invalid")
    if not isinstance(stage1.get("entity_list"), list):
        raise ValueError("stage1.entity_list missing or invalid")
    if not isinstance(stage1.get("triples"), list):
        raise ValueError("stage1.triples missing or invalid")
    for ent in stage1["entity_list"]:
        if not isinstance(ent, str) or not ent.strip():
            raise ValueError("stage1.entity_list item invalid")
        if is_placeholder_entity_code(ent):
            raise ValueError(f"stage1.entity_list contains placeholder code: {ent!r}")
    for tri in stage1["triples"]:
        tri2 = _normalize_stage_graph_triple(tri if isinstance(tri, dict) else {})
        if tri2 is None:
            raise ValueError("stage1 triple invalid")
        if tri2["relation"] != normalize_relation_phrase(tri2["relation"]):
            raise ValueError("stage1 relation format invalid")


def validate_triple(tri: Dict[str, Any]) -> None:
    if not isinstance(tri, dict):
        raise ValueError("triple must be object")
    if tri.get("type") not in {"ATTRIBUTE", "RELATION"}:
        raise ValueError("triple.type invalid")
    for k in ["name", "description_type", "description"]:
        if not isinstance(tri.get(k), str) or not tri[k].strip():
            raise ValueError(f"triple.{k} invalid")
    kvs = tri.get("kv_lists")
    if not isinstance(kvs, list) or len(kvs) < 2:
        raise ValueError("triple.kv_lists invalid")
    for kv in kvs:
        if not isinstance(kv, dict):
            raise ValueError("kv must be object")
        if not isinstance(kv.get("key_string"), str) or not kv["key_string"].strip():
            raise ValueError("kv.key_string invalid")
        if not isinstance(kv.get("value_string"), str) or not kv["value_string"].strip():
            raise ValueError("kv.value_string invalid")


_ANCHOR_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_ANCHOR_NON_WORD_RE = re.compile(r"[^0-9a-z]+")
_ANCHOR_MULTI_SPACE_RE = re.compile(r"\s+")
_ANCHOR_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "to", "from", "with",
    "and", "or", "is", "was", "were", "are", "be", "as",
}


def normalize_anchor_text(text: Any) -> str:
    s = norm_text(text).lower()
    s = _ANCHOR_PAREN_RE.sub("", s)
    s = _ANCHOR_NON_WORD_RE.sub(" ", s)
    s = _ANCHOR_MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def anchor_token_set(text: Any) -> set[str]:
    s = normalize_anchor_text(text)
    return {
        tok for tok in s.split()
        if tok and tok not in _ANCHOR_STOPWORDS and len(tok) > 1
    }


def has_anchor_in_key(key_string: str, anchor_text: str) -> bool:
    key_norm = normalize_anchor_text(key_string)
    if not key_norm:
        return False

    anchor_norm = normalize_anchor_text(anchor_text)
    if not anchor_norm:
        return False

    if anchor_norm in key_norm:
        return True

    key_tokens = anchor_token_set(key_string)
    anchor_tokens = anchor_token_set(anchor_text)
    if not anchor_tokens:
        return False

    overlap = len(key_tokens & anchor_tokens)
    if len(anchor_tokens) == 1:
        return overlap == 1
    if len(anchor_tokens) == 2:
        return overlap >= 1
    return overlap >= 2


def contain_anchor_in_value(anchor_text: str, value_text: str) -> bool:
    anchor_raw = norm_text(anchor_text).lower()
    value_raw = norm_text(value_text).lower()
    anchor_norm = normalize_anchor_text(anchor_text)
    value_norm = normalize_anchor_text(value_text)
    if not anchor_raw or not value_raw or not anchor_norm or not value_norm:
        return False
    if anchor_norm == value_norm:
        return True

    # Allow limited alias expansion such as "Jacksonville" <-> "Jacksonville, Florida"
    # or a parenthetical qualifier added to one side.
    for short_text, long_text in ((anchor_raw, value_raw), (value_raw, anchor_raw)):
        if long_text.startswith(short_text + ",") or long_text.endswith(", " + short_text):
            return True
        if long_text.startswith(short_text + " (") or long_text.endswith(" (" + short_text + ")"):
            return True
    return False



_LEAK_COPULA_RE = re.compile(r"\b(?:is|was|were|are|be|been|being)\b\s*$", re.IGNORECASE)


def cleanup_key_string_after_anchor_removal(text: str) -> str:
    s = norm_text(text)
    s = re.sub(r"[,:;.\-–—]+\s*$", "", s)
    s = _LEAK_COPULA_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def remove_value_anchor_from_key(key_string: str, forbidden_anchor_text: str) -> str:
    key = norm_text(key_string)
    anchor = norm_text(forbidden_anchor_text)
    if not key or not anchor:
        return key

    escaped_anchor = re.escape(anchor)

    s = re.sub(rf"\s+{escaped_anchor}\s*$", "", key, flags=re.IGNORECASE)
    if s != key:
        return cleanup_key_string_after_anchor_removal(s)

    s = re.sub(
        rf"\s+(?:is|was|were|are|be|been|being)\s+{escaped_anchor}\s*$",
        "",
        key,
        flags=re.IGNORECASE,
    )
    if s != key:
        return cleanup_key_string_after_anchor_removal(s)

    s = re.sub(
        rf"^\s*{escaped_anchor}\s+(?:is|was|were|are|be|been|being)\s+",
        "",
        key,
        flags=re.IGNORECASE,
    )
    if s != key:
        return cleanup_key_string_after_anchor_removal(s)

    s = re.sub(
        rf"^\s*{escaped_anchor}\s*[,:\-]\s*",
        "",
        key,
        flags=re.IGNORECASE,
    )
    if s != key:
        return cleanup_key_string_after_anchor_removal(s)

    return key

def has_full_anchor_leak(key_string: str, forbidden_anchor_text: str) -> bool:
    key_norm = normalize_anchor_text(key_string)
    forbidden_norm = normalize_anchor_text(forbidden_anchor_text)
    if not key_norm or not forbidden_norm:
        return False
    return forbidden_norm in key_norm

def postprocess_triple_kv_lists(tri: Dict[str, Any]) -> Dict[str, Any]:
    name = norm_text(tri.get("name", ""))
    description = norm_text(tri.get("description", ""))
    if not name or not description:
        return tri

    repaired_kvs: List[Dict[str, str]] = []
    for kv in tri.get("kv_lists", []) or []:
        key_string = norm_text(kv.get("key_string", ""))
        value_string = norm_text(kv.get("value_string", ""))
        if not key_string or not value_string:
            continue

        repaired_key = key_string
        if value_string == description and has_full_anchor_leak(repaired_key, description):
            candidate = remove_value_anchor_from_key(repaired_key, description)
            if candidate:
                repaired_key = candidate
        elif value_string == name and has_full_anchor_leak(repaired_key, name):
            candidate = remove_value_anchor_from_key(repaired_key, name)
            if candidate:
                repaired_key = candidate

        repaired_kvs.append(
            {
                "key_string": repaired_key,
                "value_string": value_string,
            }
        )

    tri["kv_lists"] = repaired_kvs
    return tri


def validate_kv_anchor_and_value(tri: Dict[str, Any]) -> None:
    name = norm_text(tri.get("name", ""))
    description = norm_text(tri.get("description", ""))
    if not name or not description:
        raise ValueError("triple name/description missing for kv anchor validation")

    for idx, kv in enumerate(tri.get("kv_lists", []) or []):
        key_string = norm_text(kv.get("key_string", ""))
        value_string = norm_text(kv.get("value_string", ""))
        expected_forward = (idx % 2 == 0)

        if expected_forward and not contain_anchor_in_value(value_string, description):
            raise ValueError(
                f"kv_lists must alternate forward/reverse starting with forward: "
                f"index={idx} expected_value={description!r} got={value_string!r}"
            )
        if (not expected_forward) and not contain_anchor_in_value(value_string, name):
            print(tri)
            raise ValueError(
                f"kv_lists must alternate forward/reverse starting with forward: "
                f"index={idx} expected_value={name!r} got={value_string!r}"
            )

        if contain_anchor_in_value(value_string, description):
            if not has_anchor_in_key(key_string, name):
                raise ValueError(
                    f"forward key_string missing subject anchor: key={key_string!r}, name={name!r}, description={description!r}"
                )

        elif contain_anchor_in_value(value_string, name):
            if not has_anchor_in_key(key_string, description):
                raise ValueError(
                    f"reverse key_string missing object/value anchor: key={key_string!r}, name={name!r}, description={description!r}"
                )
        else:
            raise ValueError(
                f"kv.value_string must equal either description or name: value={value_string!r}, name={name!r}, description={description!r}"
            )

# ============================================================
# Output resume / cache helpers
# ============================================================


def cache_path(cache_dir: Optional[str], sample_id: str, suffix: str) -> Optional[str]:
    if not cache_dir:
        return None
    return os.path.join(cache_dir, f"{sample_id}.{suffix}.json")


def stage_cache_subdir(stage_cache_root: Optional[str], stage_name: str) -> Optional[str]:
    if not stage_cache_root:
        return None
    return os.path.join(stage_cache_root, stage_name)


def load_cached_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cached_json(path: Optional[str], obj: Dict[str, Any]) -> None:
    if not path:
        return
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_existing_output_ids(path: str) -> set[str]:
    ids = set()
    if not path or not os.path.exists(path):
        return ids

    # load jsonl data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sample_id = obj.get("_id")
                if isinstance(sample_id, str):
                    ids.add(sample_id)
            except json.JSONDecodeError:
                continue
    return ids


# ============================================================
# Merge stage1 + stage2 revision graph
# ============================================================


def merge_stage1_and_stage2_graph(
    stage1: Dict[str, Any],
    stage2: Dict[str, Any],
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "_id": norm_text(stage2.get("_id", "")) or norm_text(stage1.get("_id", "")),
        "entity_list": [],
        "triples": [],
        "answer_sufficient": bool(stage2.get("answer_sufficient", False)),
        "missing_links": [
            norm_text(x) for x in (stage2.get("missing_links", []) or []) if norm_text(x)
        ],
        "revision_notes": [
            norm_text(x) for x in (stage2.get("revision_notes", []) or []) if norm_text(x)
        ],
    }

    entity_seen = set()
    for ent in (stage1.get("entity_list", []) or []) + (stage2.get("entity_list", []) or []):
        name = norm_text(ent)
        if name and name not in entity_seen and not is_placeholder_entity_code(name):
            out["entity_list"].append(name)
            entity_seen.add(name)

    existing_sigs = set()
    for tri in list(_iter_stage_graph_triples(stage1)) + list(_iter_stage_graph_triples(stage2)):
        tri2 = _normalize_stage_graph_triple(tri)
        if tri2 is None:
            continue
        sig = (tri2["head"], tri2["relation"], tri2["tail"], tri2["triple_type"])
        if sig in existing_sigs:
            continue
        out["triples"].append(tri2)
        existing_sigs.add(sig)

    out = dedupe_stage1(out)
    out["answer_sufficient"] = bool(stage2.get("answer_sufficient", False))
    out["missing_links"] = [
        norm_text(x) for x in (stage2.get("missing_links", []) or []) if norm_text(x)
    ]
    out["revision_notes"] = [
        norm_text(x) for x in (stage2.get("revision_notes", []) or []) if norm_text(x)
    ]
    if verbose:
        print(f"[stage2_merged_graph] {out}")
    return out


# ============================================================
# Merge model output back into sample
# ============================================================


def merge_final_stage_into_sample(sample: Dict[str, Any], stage_final: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    out["triple_list"] = copy.deepcopy(stage_final.get("triple_list", []) or [])
    out["answer_sufficient"] = bool(stage_final.get("answer_sufficient", False))
    out["missing_links"] = [
        norm_text(x) for x in (stage_final.get("missing_links", []) or []) if norm_text(x)
    ]
    out["revision_notes"] = [
        norm_text(x) for x in (stage_final.get("revision_notes", []) or []) if norm_text(x)
    ]
    return out


def normalize_final_sample_output(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    normalized_context: List[Dict[str, Any]] = []

    for raw_para in iter_sample_context_entries(out):
        para = normalize_context_entry(raw_para)
        if para is None:
            continue
        normalized_context.append(
            {
                "title": para["title"],
                "sentences": [norm_text(x) for x in (para.get("sentences", []) or [])],
            }
        )

    out["context"] = normalized_context
    out["triple_list"] = out.get("triple_list", []) or []
    out["answer_sufficient"] = bool(out.get("answer_sufficient", False))
    out["missing_links"] = [
        norm_text(x) for x in (out.get("missing_links", []) or []) if norm_text(x)
    ]
    out["revision_notes"] = [
        norm_text(x) for x in (out.get("revision_notes", []) or []) if norm_text(x)
    ]
    out.pop("paragraphs", None)
    return out


class SampleProcessingError(RuntimeError):
    def __init__(self, stage: str, sample_id: str, cause: Exception):
        self.stage = stage
        self.sample_id = sample_id
        self.cause = cause
        super().__init__(f"{stage} failed for {sample_id}: {cause}")


def classify_error_location(exc: Exception) -> str:
    target = exc.cause if isinstance(exc, SampleProcessingError) and exc.cause is not None else exc
    tb_entries = traceback.extract_tb(target.__traceback__) if target.__traceback__ else []
    current_file = os.path.abspath(__file__)

    for frame in reversed(tb_entries):
        if os.path.abspath(frame.filename) == current_file:
            return f"{frame.name}:{frame.lineno}"
    if tb_entries:
        frame = tb_entries[-1]
        return f"{os.path.basename(frame.filename)}:{frame.name}:{frame.lineno}"
    return target.__class__.__name__


@dataclass
class QueueItem:
    idx: int
    sample: Dict[str, Any]
    sample_id: str


async def queue_worker(
    *,
    worker_id: int,
    queue: asyncio.Queue,
    client: OpenAIResponsesClient,
    cfg: WorkerConfig,
    output_path: str,
    error_log: Optional[str],
    progress_every: int,
    sample_retry_limit: int,
    stats: Dict[str, int],
    failure_stats: Dict[str, int],
    retry_counts: Dict[str, int],
    progress_lock: asyncio.Lock,
) -> None:
    while True:
        item: QueueItem = await queue.get()
        try:
            sample_id, final_sample, _meta = await process_one_sample(
                idx=item.idx,
                sample=item.sample,
                client=client,
                cfg=cfg,
                verbose=cfg.verbose,
            )
            append_jsonl(output_path, final_sample)
            async with progress_lock:
                stats["done"] += 1
                stats["ok"] += 1
                done_cnt = stats["done"]
                if done_cnt % max(1, progress_every) == 0 or done_cnt == stats["total"]:
                    print(build_progress_message(
                        stats=stats,
                        failure_stats=failure_stats,
                        last_sample_id=sample_id,
                    ))
        except Exception as e:
            tb = traceback.format_exc()
            async with progress_lock:
                error_location = classify_error_location(e)
                failure_stats[error_location] = failure_stats.get(error_location, 0) + 1
                retry_counts[item.sample_id] = retry_counts.get(item.sample_id, 0) + 1
                attempts = retry_counts[item.sample_id]
                if attempts <= sample_retry_limit:
                    stats["retried"] += 1
                    print(
                        f"[{now_ts()}] retry sample={item.sample_id} "
                        f"attempt={attempts}/{sample_retry_limit} location={error_location}"
                    )
                    await queue.put(item)
                else:
                    stats["done"] += 1
                    stats["err"] += 1
                    stats["dropped"] += 1
                    done_cnt = stats["done"]
                    msg = (
                        f"[{now_ts()}] ERROR sample={item.sample_id} "
                        f"attempts_exhausted={attempts - 1} worker={worker_id} location={error_location} {e}\n{tb}"
                    )
                    print(msg, file=sys.stderr)
                    if error_log:
                        with open(error_log, "a", encoding="utf-8") as f:
                            f.write(msg + "\n")
                    if done_cnt % max(1, progress_every) == 0 or done_cnt == stats["total"]:
                        print(build_progress_message(
                            stats=stats,
                            failure_stats=failure_stats,
                            last_sample_id=item.sample_id,
                        ))
        finally:
            queue.task_done()


# ============================================================
# Stage schemas / prompts
# ============================================================

STAGE2_REVISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "_id": {"type": "string"},
        "answer_sufficient": {"type": "boolean"},
        "missing_links": {
            "type": "array",
            "items": {"type": "string"}
        },
        "revision_notes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "entity_list": {
            "type": "array",
            "items": {"type": "string"}
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "head": {"type": "string"},
                    "relation": {"type": "string"},
                    "tail": {"type": "string"},
                    "triple_type": {
                        "type": "string",
                        "enum": ["ATTRIBUTE", "RELATION"]
                    }
                },
                "required": ["head", "relation", "tail", "triple_type"]
            }
        }
    },
    "required": [
        "_id",
        "answer_sufficient",
        "missing_links",
        "revision_notes",
        "entity_list",
        "triples"
    ]
}

STAGE2_REVISION_SYSTEM_PROMPT = """You are a graph revision system for answer-aware knowledge graph repair.

You are given:
- a question
- the gold answer
- an optional question decomposition
- the source pages
- a stage1 graph containing an entity list and a triple list

Your job is to revise the stage1 graph so that it is as faithful, and answer-useful as possible.

You must:
1. check whether the current graph already contains an explicit bridge chain from the question-referred object(s) to the answer,
2. if not, minimally add missing entities and triples from the provided pages,
3. decide whether the revised graph is sufficient to answer the question,
4. normalize triple wording so that the triples are structurally clean and linguistically natural,
5. return the revised graph in the required JSON schema.

You must NOT output any text outside the required JSON object.

==================================================
PRIMARY OBJECTIVE
==================================================

Revise the stage1 graph so that it can explicitly support answering the given question using only facts grounded in the provided pages.

The revised graph should:
- preserve as much correct stage1 content as possible,
- add only facts necessary for answerability,
- avoid unnecessary graph expansion,
- keep entity names and triple wording clean and canonical,
- satisfy the required output schema exactly.

This is a revision task, NOT a full re-extraction task.
Prefer minimal edits over rewriting everything.

==================================================
CORE CONCEPT: ANSWER-SUPPORTING BRIDGE CHAIN
==================================================

A graph is answer-sufficient only if it contains an explicit bridge chain from the question target to the answer.

A bridge chain is a short sequence of grounded facts that connects:
- the entity or object referred to by the question
to
- the gold answer
possibly through one or more intermediate entities.

Every step in the chain must be explicitly supported by the provided pages.

Do NOT treat vague topical relatedness as a valid bridge chain.
Do NOT rely on outside knowledge.
Do NOT rely on implicit commonsense leaps.
The connection must be recoverable from the graph through explicit facts.

==================================================
INPUT INTERPRETATION
==================================================

You will receive:
- question
- answer
- question_decomposition (optional)
- pages
- stage1 graph

Interpret the stage1 graph as an initial draft.
It may contain:
- missing entities
- missing triples
- wrong triple types
- awkward relation phrases
- answer-insufficient structure

Your task is to repair these issues minimally.

==================================================
REVISION PROCEDURE
==================================================

Follow this procedure internally before producing the final JSON:

Step 1. Identify the question anchor(s)
- Determine the main entity, event, or object referred to by the question.
- Determine what information the question is asking for.
- Determine what the gold answer refers to: an entity or a value.

Step 2. Check current graph sufficiency
Examine whether the current stage1 graph already contains an explicit answer-supporting bridge chain.
A graph is sufficient only if:
- the question anchor is represented in the graph, and
- the answer is represented in the graph, and
- the connection between them is explicitly recoverable through the graph’s triples.

Step 3. If insufficient, minimally repair the graph
Use the provided pages and optional question_decomposition to add only the missing core entities and triples required to complete the answer-supporting bridge chain.

Important:
- Add the minimum necessary facts.
- Do not expand the graph with unrelated background facts.
- Do not add entities or triples that are not useful for answering the current question.
- Prefer preserving stage1 items when they are correct.
- If question_decomposition is provided, use it as a hint for the intended bridge chain and missing intermediate steps.
- Do not treat question_decomposition as independent evidence; only keep a repaired step if it is explicitly supported by the provided pages.

Step 4. Re-check sufficiency
After repair, determine whether the revised graph is now sufficient to answer the question.
Set:
- answer_sufficient = true, if the revised graph contains an explicit bridge chain to the answer
- answer_sufficient = false, otherwise

If still insufficient, list the remaining core missing links in missing_links.

Step 5. Normalize the final graph
Before output, revise entity names and triple wording so the graph is clean and consistent.
- After deciding the final triples, run one final wording check:
  - every ATTRIBUTE triple should read naturally under "the <relation> of <head> is <tail>"
  - every RELATION triple should read naturally under "<head> <relation> <tail>"
- If a triple sounds awkward under its required template, rewrite that triple minimally without changing the grounded fact.
- For RELATION triples specifically, rewrite any relation that is a sentence fragment, a weak standalone preposition, or a temporary bridge-only phrase into a clean self-contained relation phrase when the pages support that wording.

==================================================
ENTITY REVISION RULES
==================================================

The final entity_list should contain graph-worthy entities only.

A graph-worthy entity is an independently referable object that is worth being a reusable graph node.

Include entities such as:
- people
- places
- organizations
- works
- events
- movements
- taxa/species/genus/family
- named institutions
- clearly referable named or uniquely identifiable objects

Do NOT include:
- plain numbers
- dates
- years
- measurements
- generic descriptive values
- ordinary non-node value phrases

Entity list rules:
- Keep entity names canonical and self-contained.
- Merge duplicate mentions referring to the same entity.
- Use disambiguated names when needed.
- Do not use IDs like E1, E2, T1, etc.
- Every entity appearing in a RELATION triple must appear in entity_list.
- ATTRIBUTE tails should usually NOT appear in entity_list unless they are truly graph-worthy entities.

==================================================
TRIPLE REVISION RULES
==================================================

Each triple must be one of two types:

1. RELATION
A RELATION triple connects:
- head = entity
- tail = entity

Use RELATION only when the tail is a graph-worthy entity that should be a node.

2. ATTRIBUTE
An ATTRIBUTE triple connects:
- head = entity
- tail = value

Use ATTRIBUTE only when the tail is not a graph-worthy entity and is better treated as a literal/value.

Examples of ATTRIBUTE tails:
- dates
- years
- numbers
- quantities
- measurements
- short category labels
- short descriptive values

==================================================
TRIPLE TYPE DECISION RULE
==================================================

Use this rule strictly:

- If the tail should be a reusable graph node, set triple_type = "RELATION".
- If the tail should be a value/literal rather than a reusable graph node, set triple_type = "ATTRIBUTE".

So:
- entity -> entity = RELATION
- entity -> value = ATTRIBUTE

Do NOT classify based only on wording style.
Decide based on whether the tail is graph-worthy.

==================================================
TRIPLE WORDING NORMALIZATION
==================================================

The final triples must be wording-clean.

For ATTRIBUTE triples:
When inserted into the template
    "the <relation> of <head> is <tail>"
the expression should sound natural and semantically correct.

For RELATION triples:
When inserted into the template
    "<head> <relation> <tail>"
the expression should sound natural and semantically correct.

For RELATION triples, the relation itself should be a clean self-contained phrase.
Do NOT leave relations as sentence fragments such as "calls target countries" or "is one of pioneers of".
Do NOT use weak standalone prepositions such as "in", "on", "under", "of" as the full relation unless they are part of a stable phrase such as "located in" or "born on".
Do NOT create temporary bridge-only relations such as "leads to" just to connect the answer chain.

If a triple is semantically correct but awkwardly phrased, rewrite only as much as needed to make it natural.

Important constraints:
- Do NOT change the fact itself.
- Do NOT change the entity identity.
- Do NOT introduce new information.
- Do NOT over-rewrite.
- Prefer minimal phrase repair.

In most cases:
- for ATTRIBUTE triples, mainly refine the relation phrase
- for RELATION triples, mainly refine the relation phrase
Avoid changing head or tail unless the original name is clearly non-canonical or duplicated.

==================================================
EVIDENCE AND GROUNDING RULES
==================================================

All retained or added graph content must be grounded in the provided pages.

You may use question_decomposition only as a hint for identifying missing reasoning steps.
You may also use it as a hint for the likely order of the bridge chain.
You may NOT use it as independent evidence.

Do NOT:
- invent facts,
- import outside knowledge,
- add plausible but unsupported bridge steps,
- make speculative corrections.

==================================================
MISSING_LINKS RULES
==================================================

missing_links should be a concise list of the still-missing core facts or bridge steps preventing answer sufficiency.

Use short strings, for example:
- "missing entity linking X to Y"
- "missing fact connecting A to B"
- "missing attribute giving the birth date of C"

Only include the essential remaining gaps.
If answer_sufficient is true, missing_links should usually be an empty list.

==================================================
REVISION_NOTES RULES
==================================================

revision_notes should briefly summarize what was changed.

Use short strings such as:
- "added missing entity: ..."
- "added missing relation triple: ..."
- "added missing attribute triple: ..."
- "reclassified triple from ATTRIBUTE to RELATION"
- "rewrote awkward relation phrase for naturalness"
- "merged duplicate entities: X and Y"

Keep notes concise and factual.
Do not include long explanations.

==================================================
MINIMALITY RULES
==================================================

This is crucial.

You are revising a stage1 graph, not rebuilding from scratch.

Therefore:
- keep correct stage1 entities when possible
- keep correct stage1 triples when possible
- only add missing facts required for answerability
- only rewrite wording when necessary
- do not introduce unrelated graph content

==================================================
OUTPUT REQUIREMENTS
==================================================

You must satisfy all of the following:
- Output valid JSON only.
- Do not add any extra fields.
- _id must be preserved from the input sample.
- entity_list must be deduplicated.
- All RELATION triple heads and tails must be in entity_list.
- ATTRIBUTE triple heads must be in entity_list.
- Triple wording must be natural under the required templates.
- If the graph is still insufficient, answer_sufficient must be false.
- If the graph is sufficient, answer_sufficient must be true.

==================================================
FINAL CHECK BEFORE OUTPUT
==================================================

Before producing the final JSON, verify:

1. Is there now an explicit bridge chain from the question anchor to the answer?
2. If yes, set answer_sufficient = true.
3. If no, set answer_sufficient = false and fill missing_links with the remaining core gaps.
4. Are all entities deduplicated and canonical?
5. Are all triple types correct?
6. For every ATTRIBUTE triple, does
   "the <relation> of <head> is <tail>"
   sound natural?
7. For every RELATION triple, does
   "<head> <relation> <tail>"
   sound natural?
8. Is every fact grounded in the provided pages?
9. Is the revision minimal rather than a full rewrite?
10. Is every RELATION relation a clean self-contained phrase rather than a sentence fragment, weak preposition, or bridge-only phrase?


Return valid JSON only and follow the schema exactly.
"""


def build_programmatic_final_from_graph(
    sample: Dict[str, Any],
    graph_for_kv: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "_id": norm_text(graph_for_kv.get("_id", "")) or safe_sample_id(sample, 0),
        "triple_list": [],
        "answer_sufficient": bool(graph_for_kv.get("answer_sufficient", False)),
        "missing_links": [
            norm_text(x) for x in (graph_for_kv.get("missing_links", []) or []) if norm_text(x)
        ],
        "revision_notes": [
            norm_text(x) for x in (graph_for_kv.get("revision_notes", []) or []) if norm_text(x)
        ],
    }
    for tri in dedupe_stage2_input_triples(graph_for_kv.get("triples", []) or []):
        forward_kv = programmatic_forward_kv(tri)
        reverse_kv = programmatic_reverse_kv(tri)
        out["triple_list"].append(
            {
                "type": get_triple_type(tri),
                "name": norm_text(tri.get("head", "")),
                "description_type": norm_text(tri.get("relation", "")),
                "description": norm_text(tri.get("tail", "")),
                "kv_lists": [
                    {
                        "key_string": norm_text(forward_kv.get("key_string", "")),
                        "value_string": norm_text(forward_kv.get("value_string", "")),
                    },
                    {
                        "key_string": norm_text(reverse_kv.get("key_string", "")),
                        "value_string": norm_text(reverse_kv.get("value_string", "")),
                    },
                ],
            }
        )
    return out


def validate_final_stage(stage_final: Dict[str, Any], verbose: bool = False) -> None:
    if verbose:
        return
    if not isinstance(stage_final, dict):
        raise ValueError("final stage must be a dict")
    if not isinstance(stage_final.get("_id"), str):
        raise ValueError("final._id missing or invalid")
    if not isinstance(stage_final.get("answer_sufficient"), bool):
        raise ValueError("final.answer_sufficient missing or invalid")
    if not isinstance(stage_final.get("missing_links"), list):
        raise ValueError("final.missing_links missing or invalid")
    if not isinstance(stage_final.get("revision_notes"), list):
        raise ValueError("final.revision_notes missing or invalid")
    if not isinstance(stage_final.get("triple_list"), list):
        raise ValueError("final.triple_list missing or invalid")
    for tri in stage_final["triple_list"]:
        validate_triple(tri)
        validate_kv_anchor_and_value(tri)


@dataclass
class WorkerConfig:
    include_question: bool
    include_answer: bool
    answer_aware: bool
    supporting_pages_only: bool
    stage_cache_dir: Optional[str]
    overwrite: bool
    verbose: bool = False


async def process_one_sample(
    *,
    idx: int,
    sample: Dict[str, Any],
    client: OpenAIResponsesClient,
    cfg: WorkerConfig,
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    sample_id = safe_sample_id(sample, idx)
    sample["_id"] = sample_id

    stage1_path = cache_path(stage_cache_subdir(cfg.stage_cache_dir, "stage1"), sample_id, "stage1")
    stage2_path = cache_path(stage_cache_subdir(cfg.stage_cache_dir, "stage2"), sample_id, "stage2")

    # ---------------- stage 1: graph extraction ----------------
    stage1_input = build_stage1_input(
        sample,
        include_question=cfg.include_question,
        include_answer=cfg.include_answer,
        supporting_pages_only=cfg.supporting_pages_only,
    )
    try:
        stage1 = None if cfg.overwrite else load_cached_json(stage1_path)
        if stage1 is None:
            if verbose:
                print(f"[stage1_input] {stage1_input}")
            stage1 = await client.create_structured_response(
                system_prompt=STAGE1_SYSTEM_PROMPT,
                user_payload=stage1_input,
                schema_name="stage1_graph",
                schema=STAGE1_SCHEMA,
                request_id=make_request_id("stage1", sample_id),
            )
    except Exception as e:
        raise SampleProcessingError("stage1_api", sample_id, e) from e
    try:
        stage1 = dedupe_stage1(stage1)
        validate_stage1(stage1)
        save_cached_json(stage1_path, stage1)
    except Exception as e:
        raise SampleProcessingError("stage1_validation", sample_id, e) from e

    # ---------------- stage 2: answer-aware graph revision ----------------
    stage2_default = {
        "_id": norm_text(stage1.get("_id", "")),
        "answer_sufficient": False,
        "missing_links": [],
        "revision_notes": [],
        "entity_list": copy.deepcopy(stage1.get("entity_list", []) or []),
        "triples": copy.deepcopy(stage1.get("triples", []) or []),
    }
    stage2 = None if cfg.overwrite else load_cached_json(stage2_path)
    if stage2 is None:
        if cfg.answer_aware and cfg.include_answer:
            stage2_input = stage1_input.copy()
            stage2_input["stage1_graph"] = {
                "entity_list": copy.deepcopy(stage1.get("entity_list", []) or []),
                "triples": copy.deepcopy(stage1.get("triples", []) or []),
            }
            question_decomposition = build_stage2_question_decomposition(sample)
            if question_decomposition:
                stage2_input["question_decomposition"] = question_decomposition

            if verbose:
                print(f"[stage2_input] {stage2_input}")
            try:
                stage2 = await client.create_structured_response(
                    system_prompt=STAGE2_REVISION_SYSTEM_PROMPT,
                    user_payload=stage2_input,
                    schema_name="stage2_answer_aware_revision",
                    schema=STAGE2_REVISION_SCHEMA,
                    request_id=make_request_id("stage2", sample_id),
                )
            except Exception as e:
                raise SampleProcessingError("stage2_api", sample_id, e) from e
        else:
            stage2 = stage2_default
            if verbose:
                print("[stage2_revision] skipped")
    try:
        stage2 = normalize_stage2_revision(stage2, verbose=verbose)
        validate_stage2_revision(stage1, stage2, verbose=verbose)
        save_cached_json(stage2_path, stage2)
    except Exception as e:
        raise SampleProcessingError("stage2_validation", sample_id, e) from e

    graph_revision = stage2
    graph_for_kv = merge_stage1_and_stage2_graph(stage1, stage2, verbose=verbose)

    # ---------------- deterministic KV generation ----------------
    try:
        stage_final = build_programmatic_final_from_graph(sample, graph_for_kv)
    except Exception as e:
        raise SampleProcessingError("deterministic_kv", sample_id, e) from e

    try:
        try:
            validate_final_stage(stage_final, verbose=verbose)
        except Exception as e:
            print(f"[{now_ts()}] WARN sample={sample_id} stage=final dropped_invalid_final {e}", file=sys.stderr)

        final_sample = merge_final_stage_into_sample(sample, stage_final)
        final_sample = normalize_final_sample_output(final_sample)
    except Exception as e:
        raise SampleProcessingError("finalization", sample_id, e) from e

    return sample_id, final_sample, {
        "stage1": stage1,
        "stage2": graph_revision,
        "graph_for_kv": graph_for_kv,
        "stage_final": stage_final,
    }


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and args.api_key_env != "OFOXAI_API_KEY":
        api_key = os.environ.get("OFOXAI_API_KEY", "").strip()
    if not api_key and not args.allow_empty_api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is empty."
            " Set it directly, or export OFOXAI_API_KEY for OfoxAI."
        )

    ensure_dir(os.path.dirname(args.output) or ".")
    if args.error_log:
        ensure_dir(os.path.dirname(args.error_log) or ".")
    for stage_name in ["stage1", "stage2"]:
        ensure_dir(stage_cache_subdir(args.stage_cache_dir, stage_name))

    samples = read_json_or_jsonl(args.input)
    print(f"Load {len(samples)} samples from {args.input}")
    # 过滤掉无法answer的样本
    samples = [s for s in samples if s.get("answerable", True) is not False]
    print(f"Filter {len(samples)} samples after answerable check")

    # 过滤掉comparison样本
    if args.skip_comparison:
        samples = [s for s in samples if s.get("type", "") != "comparison"]
        print(f"Filter {len(samples)} samples after comparison check")

    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(samples)

    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]
    print(f"Process {len(samples)} samples")

    existing_ids = set()
    if args.resume and not args.overwrite:
        existing_ids = load_existing_output_ids(args.output)

    concurrency = max(1, args.concurrency)
    max_connections = args.max_connections if args.max_connections is not None else concurrency
    max_keepalive_connections = (
        args.max_keepalive_connections
        if args.max_keepalive_connections is not None
        else concurrency
    )

    cfg = OpenAIConfig(
        api_key=api_key,
        api_base=args.api_base,
        model=args.model,
        timeout=args.timeout,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        max_retries=args.max_retries,
        retry_base=args.retry_base,
        retry_max=args.retry_max,
        store=not args.no_store,
        verbosity=args.verbosity,
        api_mode=args.api_mode,
    )
    worker_cfg = WorkerConfig(
        include_question=not args.no_question,
        include_answer=not args.no_answer,
        answer_aware=args.answer_aware,
        supporting_pages_only=not args.use_all_context_pages,
        stage_cache_dir=args.stage_cache_dir,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )

    client = OpenAIResponsesClient(cfg)

    try:
        queue: asyncio.Queue = asyncio.Queue()
        retry_counts: Dict[str, int] = {}
        progress_lock = asyncio.Lock()
        total = 0
        skipped_existing = 0
        skipped_no_support = 0
        for idx, sample in enumerate(samples):
            sample_id = safe_sample_id(sample, idx)
            if sample_id in existing_ids:
                skipped_existing += 1
                continue
            if worker_cfg.supporting_pages_only and not has_supporting_context(sample):
                skipped_no_support += 1
                continue
            total += 1
            await queue.put(QueueItem(idx=idx, sample=sample, sample_id=sample_id))

        print(
            f"[{now_ts()}] Loaded {len(samples)} samples; pending={total}; "
            f"skipped_existing={skipped_existing}; skipped_no_support={skipped_no_support}"
        )

        stats = {
            "total": total,
            "done": 0,
            "ok": 0,
            "err": 0,
            "retried": 0,
            "dropped": 0,
            "started_at": time.time(),
        }
        failure_stats: Dict[str, int] = {}

        workers = [
            asyncio.create_task(
                queue_worker(
                    worker_id=wid,
                    queue=queue,
                    client=client,
                    cfg=worker_cfg,
                    output_path=args.output,
                    error_log=args.error_log,
                    progress_every=args.progress_every,
                    sample_retry_limit=args.sample_retries,
                    stats=stats,
                    failure_stats=failure_stats,
                    retry_counts=retry_counts,
                    progress_lock=progress_lock,
                )
            )
            for wid in range(concurrency)
        ]

        await queue.join()

        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        print(build_progress_message(
            stats=stats,
            failure_stats=failure_stats,
            last_sample_id="<done>",
        ))
        print(
            f"[{now_ts()}] DONE total={total} ok={stats['ok']} err={stats['err']} "
            f"retried={stats['retried']} dropped={stats['dropped']} "
            f"elapsed={format_duration(time.time() - stats['started_at'])} "
            f"failures={dict(sorted(failure_stats.items()))}"
        )
    finally:
        await client.aclose()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="V5 graph-first KG extraction with typed triples and deterministic KV generation")
    ap.add_argument("--input", type=str, required=True, help="Input .json or .jsonl file")
    ap.add_argument("--output", type=str, required=True, help="Output .jsonl file")
    ap.add_argument("--model", type=str, default="openai/gpt-5.2", help="Model id, e.g. openai/gpt-5.2 or anthropic/claude-sonnet-4")
    ap.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", os.environ.get("OFOXAI_BASE_URL", "https://api.ofox.ai/v1")),
        help="OpenAI-compatible API base URL",
    )
    ap.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="API key environment variable; also falls back to OFOXAI_API_KEY",
    )
    ap.add_argument(
        "--api-mode",
        type=str,
        default="chat",
        choices=["responses", "chat"],
        help="Use /v1/responses or /v1/chat/completions",
    )
    ap.add_argument(
        "--allow-empty-api-key",
        action="store_true",
        help="Allow empty API key for local servers such as vLLM",
    )

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="HTTP client max concurrent connections; defaults to --concurrency",
    )
    ap.add_argument(
        "--max-keepalive-connections",
        type=int,
        default=None,
        help="HTTP client keepalive pool size; defaults to --concurrency",
    )
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--retry-base", type=float, default=2.0)
    ap.add_argument("--retry-max", type=float, default=30.0)
    ap.add_argument("--verbosity", type=str, default="low", choices=["low", "medium", "high"])
    ap.add_argument("--no-store", action="store_true", help="Set store=false on Responses API")

    ap.add_argument("--resume", action="store_true", help="Skip samples already present in output jsonl")
    ap.add_argument("--stage-cache-dir", type=str, default=None, help="Optional root dir for per-stage caches; uses stage1/stage2 subdirs")
    ap.add_argument("--overwrite", action="store_true", help="Ignore caches and recompute")
    ap.add_argument("--error-log", type=str, default="./kg_extract_errors.log")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--sample-retries", type=int, default=2, help="Max times to requeue a failed sample after the initial attempt")

    ap.add_argument("--no-question", action="store_true", help="Do not include question in stage1 prompt")
    ap.add_argument("--no-answer", action="store_true", help="Do not include answer in stage1 prompt")
    ap.add_argument("--answer-aware", action="store_true", help="Enable answer-aware stage2 graph revision before deterministic KV generation")
    ap.add_argument("--use-all-context-pages", action="store_true", help="Use all context/pages end-to-end; default is supporting pages only")
    ap.add_argument("--skip-comparison", action="store_true", help="Add SKTP comparison description type to stage2 style hints")
    ap.add_argument("--verbose", action="store_true", help="Print verbose output")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for shuffling samples")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
