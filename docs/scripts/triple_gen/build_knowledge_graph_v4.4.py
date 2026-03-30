#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_knowledge_graph_v4.4.py

Four-stage graph-first KG extraction pipeline for QA datasets using OpenAI-compatible APIs.

Stage 1:
    Build an evidence-grounded graph skeleton from supporting pages: nodes + triples.
Stage 2:
    Optional answer-aware graph revision / repair over the stage-1 graph.
Stage 3:
    Convert the selected graph into dataset-compatible triple_list JSON with forward KV only.
Stage 4:
    Generate reverse KV only.

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
    python3 build_knowledge_graph_v4.4.py \
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
from collections import defaultdict
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


# ============================================================
# Schemas for structured outputs
# ============================================================

STAGE1_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "_id": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entity_id": {"type": "string"},
                    "name": {"type": "string"},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["entity_id", "name", "aliases"],
            },
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "triple_id": {"type": "string"},
                    "head": {"type": "string"},
                    "relation": {"type": "string"},
                    "tail": {"type": "string"},
                    "tail_kind": {"type": "string", "enum": ["entity", "literal"]},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title": {"type": "string"},
                                "sentence_id": {"type": "integer"},
                            },
                            "required": ["title", "sentence_id"],
                        },
                    },
                },
                "required": [
                    "triple_id",
                    "head",
                    "relation",
                    "tail",
                    "tail_kind",
                    "evidence",
                ],
            },
        },
    },
    "required": ["_id", "entities", "triples"],
}

# ============================================================
# Prompts
# ============================================================

STAGE1_SYSTEM_PROMPT = """You are a knowledge graph construction engine.

Your task is to construct a complete, faithful, evidence-grounded graph skeleton from the provided context.

Primary goal:
1. Identify all minimal reusable nodes needed to represent the explicit facts in context.
2. Identify all explicit edges among those nodes.
3. Output a complete triple list that preserves graph connectivity for downstream multi-hop reasoning.
4. Do not hallucinate or infer unstated facts.

Node rules:
- A node must be a stable, reusable semantic unit.
- Prefer keeping canonical names, titles, organizations, places, works, species, dates, years, quantities, categories, and short literal values as nodes or literal tails.
- Do not split fixed names or official titles into smaller pieces.
- Do not create standalone nodes for clause fragments or long descriptive spans that are not reusable.
- Event-like items may be kept only when they are explicitly named or clearly function as a reusable node in the text.

Triple rules:
- Extract only explicit triples supported by the provided context.
- Split multi-fact sentences into multiple triples.
- If one head links to multiple tails, emit multiple triples.
- Use short, stable relation phrases.
- Prefer preserving bridge nodes and bridge edges needed for graph connectivity.
- Keep the graph as complete as possible over the provided pages, not only the final answer-bearing fact.
- Every triple must have evidence.
- evidence.title must be the page title.
- evidence.sentence_id must be the integer sentence index in that page.

Tail typing rules:
- Use tail_kind='entity' for person, place, organization, work, event-like named node, species, group, concept-like named node, or other reusable graph node.
- Use tail_kind='literal' for date, year, number, profession label, category, nickname, boolean-like value, short descriptive value, or other attribute-like value.

Deduplication rules:
- Do not output exact duplicate triples.
- If the same triple appears in multiple sentences, keep one triple and attach all evidence items.

Output rules:
- Return valid JSON only.
- Follow the supplied JSON schema exactly.
- Do not include explanations.
- Do not include markdown.
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
                "httpx is required for async HTTP transport in build_knowledge_graph_v4.4.py. "
                "Install it with `pip install httpx`."
            )
        self.config = config
        self._base_url = config.api_base.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=self._headers,
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


def build_stage3_forward_input(
    sample: Dict[str, Any],
    stage1: Dict[str, Any],
) -> Dict[str, Any]:
    ctx_map = get_context_map(sample)
    supporting_titles = set(get_supporting_titles(sample))

    title_to_sentences: Dict[str, List[str]] = {}
    for title, para in ctx_map.items():
        if title in supporting_titles:
            title_to_sentences[title] = [norm_text(x) for x in (para.get("sentences", []) or [])]

    triples_by_title: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tri in stage1.get("triples", []) or []:
        for ev in tri.get("evidence", []) or []:
            title = norm_text(ev.get("title", ""))
            if title in title_to_sentences:
                triples_by_title[title].append(tri)

    pages: List[Dict[str, Any]] = []
    for title, sentences in title_to_sentences.items():
        pages.append(
            {
                "title": title,
                "sentences": sentences,
                "triples": dedupe_stage2_input_triples(triples_by_title.get(title, [])),
            }
        )

    return {
        "_id": safe_sample_id(sample, 0),
        "pages": pages,
        "style_hints": {
            "preferred_description_types": [
                "birth date",
                "death date",
                "release year",
                "production location",
                "nickname",
                "occupation",
                "nationality",
                "starring",
                "directed by",
                "based on",
                "composed by",
                "partnered with",
                "voice type",
                "type",
                "located in",
            ],
        },
    }



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


def _iter_stage1_raw_triples(stage1: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(stage1.get("triples"), list):
        return list(stage1.get("triples", []) or [])
    # backward compatibility with older cached / intermediate formats
    if isinstance(stage1.get("facts"), list):
        converted = []
        for fact in stage1.get("facts", []) or []:
            converted.append(
                {
                    "triple_id": norm_text(fact.get("fact_id", "")),
                    "head": norm_text(fact.get("subject", "")),
                    "relation": norm_text(fact.get("predicate", "")),
                    "tail": norm_text(fact.get("object", "")),
                    "tail_kind": norm_text(fact.get("object_kind", "")),
                    "evidence": fact.get("evidence", []) or [],
                }
            )
        return converted
    return []


def dedupe_stage1(stage1: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "_id": norm_text(stage1.get("_id", "")),
        "entities": [],
        "triples": [],
    }

    entity_seen = set()
    for ent in stage1.get("entities", []) or []:
        e = {
            "entity_id": norm_text(ent.get("entity_id", "")),
            "name": norm_text(ent.get("name", "")),
            "aliases": sorted({norm_text(a) for a in (ent.get("aliases", []) or []) if norm_text(a)}),
        }
        sig = (e["entity_id"], e["name"], tuple(e["aliases"]))
        if e["entity_id"] and e["name"] and sig not in entity_seen:
            out["entities"].append(e)
            entity_seen.add(sig)

    triple_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for tri in _iter_stage1_raw_triples(stage1):
        head = norm_text(tri.get("head", ""))
        relation = norm_text(tri.get("relation", ""))
        tail = norm_text(tri.get("tail", ""))
        kind = norm_text(tri.get("tail_kind", ""))
        if not (head and relation and tail and kind in {"entity", "literal"}):
            continue
        sig = (head, relation, tail, kind)
        evs = []
        ev_seen = set()
        for ev in tri.get("evidence", []) or []:
            title = norm_text(ev.get("title", ""))
            try:
                sid = int(ev.get("sentence_id", -1))
            except Exception:
                sid = -1
            if not title or sid < 0:
                continue
            esig = (title, sid)
            if esig not in ev_seen:
                evs.append({"title": title, "sentence_id": sid})
                ev_seen.add(esig)

        if sig not in triple_map:
            triple_map[sig] = {
                "triple_id": norm_text(tri.get("triple_id", "")) or f"G{len(triple_map) + 1}",
                "head": head,
                "relation": relation,
                "tail": tail,
                "tail_kind": kind,
                "evidence": evs,
            }
        else:
            existing_seen = {(x["title"], x["sentence_id"]) for x in triple_map[sig]["evidence"]}
            for ev in evs:
                esig = (ev["title"], ev["sentence_id"])
                if esig not in existing_seen:
                    triple_map[sig]["evidence"].append(ev)
                    existing_seen.add(esig)

    out["triples"] = list(triple_map.values())
    return out


def dedupe_stage2_input_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for tri in triples:
        sig = (
            norm_text(tri.get("head", "")),
            norm_text(tri.get("relation", "")),
            norm_text(tri.get("tail", "")),
            norm_text(tri.get("tail_kind", "")),
        )
        if sig not in seen:
            out.append(
                {
                    "triple_id": norm_text(tri.get("triple_id", "")),
                    "head": sig[0],
                    "relation": sig[1],
                    "tail": sig[2],
                    "tail_kind": sig[3],
                    "evidence": tri.get("evidence", []) or [],
                }
            )
            seen.add(sig)
    return out


def validate_stage1(stage1: Dict[str, Any]) -> None:
    if not isinstance(stage1, dict):
        raise ValueError("stage1 must be a dict")
    if not isinstance(stage1.get("_id"), str):
        raise ValueError("stage1._id missing or invalid")
    if not isinstance(stage1.get("entities"), list):
        raise ValueError("stage1.entities missing or invalid")
    if not isinstance(stage1.get("triples"), list):
        raise ValueError("stage1.triples missing or invalid")
    for ent in stage1["entities"]:
        if not isinstance(ent, dict):
            raise ValueError("stage1 entity must be object")
        if not isinstance(ent.get("entity_id"), str) or not ent["entity_id"].strip():
            raise ValueError("stage1 entity_id missing")
        if not isinstance(ent.get("name"), str) or not ent["name"].strip():
            raise ValueError("stage1 entity name missing")
        if not isinstance(ent.get("aliases"), list):
            raise ValueError("stage1 aliases missing")
    for tri in stage1["triples"]:
        if not isinstance(tri, dict):
            raise ValueError("stage1 triple must be object")
        for k in ["triple_id", "head", "relation", "tail"]:
            if not isinstance(tri.get(k), str) or not tri[k].strip():
                raise ValueError(f"stage1 triple {k} missing")
        if tri.get("tail_kind") not in {"entity", "literal"}:
            raise ValueError("stage1 triple tail_kind invalid")
        if not isinstance(tri.get("evidence"), list) or not tri["evidence"]:
            raise ValueError("stage1 triple evidence missing")
        for ev in tri["evidence"]:
            if not isinstance(ev, dict):
                raise ValueError("stage1 evidence must be object")
            if not isinstance(ev.get("title"), str) or not ev["title"].strip():
                raise ValueError("stage1 evidence.title missing")
            if not isinstance(ev.get("sentence_id"), int):
                raise ValueError("stage1 evidence.sentence_id invalid")


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

def check_supporting_sentence_coverage(sample: Dict[str, Any], stage1: Dict[str, Any]) -> Tuple[bool, List[Tuple[str, int]]]:
    required = []
    for item in sample.get("supporting_facts", []) or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title = norm_text(item[0])
            try:
                sid = int(item[1])
            except Exception:
                continue
            required.append((title, sid))

    covered = set()
    for tri in stage1.get("triples", []) or []:
        for ev in tri.get("evidence", []) or []:
            title = norm_text(ev.get("title", ""))
            sid = ev.get("sentence_id")
            if isinstance(sid, int):
                covered.add((title, sid))

    missing = [x for x in required if x not in covered]
    return (len(missing) == 0), missing


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
        "entities": [],
        "triples": [],
        "answer_sufficient": bool(stage2.get("answer_sufficient", False)),
        "missing_links": [
            norm_text(x) for x in (stage2.get("missing_links", []) or []) if norm_text(x)
        ],
        "revision_notes": [
            norm_text(x) for x in (stage2.get("revision_notes", []) or []) if norm_text(x)
        ],
    }

    entity_map: Dict[str, Dict[str, Any]] = {}
    entity_order: List[str] = []
    for ent in (stage1.get("entities", []) or []) + (stage2.get("entities", []) or []):
        entity_id = norm_text(ent.get("entity_id", ""))
        name = norm_text(ent.get("name", ""))
        aliases = sorted({norm_text(a) for a in (ent.get("aliases", []) or []) if norm_text(a)})
        if not entity_id or not name:
            continue
        if entity_id not in entity_map:
            entity_map[entity_id] = {
                "entity_id": entity_id,
                "name": name,
                "aliases": aliases,
            }
            entity_order.append(entity_id)
        else:
            entity_map[entity_id]["aliases"] = sorted(
                set(entity_map[entity_id]["aliases"]) | set(aliases)
            )
            entity_map[entity_id]["name"] = name

    def _merge_evidence_items(*evidence_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for evidence_list in evidence_lists:
            for ev in evidence_list or []:
                title = norm_text(ev.get("title", ""))
                try:
                    sid = int(ev.get("sentence_id", -1))
                except Exception:
                    sid = -1
                if not title or sid < 0:
                    continue
                sig = (title, sid)
                if sig in seen:
                    continue
                merged.append({"title": title, "sentence_id": sid})
                seen.add(sig)
        return merged

    triple_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    triple_order: List[Tuple[str, str, str, str]] = []

    for tri in stage1.get("triples", []) or []:
        sig = (
            norm_text(tri.get("head", "")),
            norm_text(tri.get("relation", "")),
            norm_text(tri.get("tail", "")),
            norm_text(tri.get("tail_kind", "")),
        )
        if not all(sig[:3]) or sig[3] not in {"entity", "literal"}:
            continue
        triple_map[sig] = {
            "triple_id": norm_text(tri.get("triple_id", "")),
            "head": sig[0],
            "relation": sig[1],
            "tail": sig[2],
            "tail_kind": sig[3],
            "evidence": _merge_evidence_items(tri.get("evidence", []) or []),
        }
        triple_order.append(sig)

    for tri in stage2.get("triples", []) or []:
        sig = (
            norm_text(tri.get("head", "")),
            norm_text(tri.get("relation", "")),
            norm_text(tri.get("tail", "")),
            norm_text(tri.get("tail_kind", "")),
        )
        if not all(sig[:3]) or sig[3] not in {"entity", "literal"}:
            continue
        if sig not in triple_map:
            triple_map[sig] = {
                "triple_id": norm_text(tri.get("triple_id", "")),
                "head": sig[0],
                "relation": sig[1],
                "tail": sig[2],
                "tail_kind": sig[3],
                "evidence": _merge_evidence_items(tri.get("evidence", []) or []),
            }
            triple_order.append(sig)
        else:
            stage2_triple_id = norm_text(tri.get("triple_id", ""))
            if stage2_triple_id:
                triple_map[sig]["triple_id"] = stage2_triple_id
            triple_map[sig]["evidence"] = _merge_evidence_items(
                triple_map[sig].get("evidence", []) or [],
                tri.get("evidence", []) or [],
            )

    out["entities"] = [entity_map[entity_id] for entity_id in entity_order]
    out["triples"] = [triple_map[sig] for sig in triple_order]
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


def merge_final_stage_into_sample(sample: Dict[str, Any], stage2: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    title_to_triples = {
        norm_text(para.get("title", "")): para.get("triple_list", []) or []
        for para in (stage2.get("context", []) or [])
    }

    target_key = "context" if isinstance(out.get("context"), list) else "paragraphs"
    for idx, para in enumerate(out.get(target_key, []) or []):
        norm_para = normalize_context_entry(para)
        if norm_para is None:
            continue

        title = norm_para["title"]
        if title in title_to_triples:
            triple_list = title_to_triples[title]
        else:
            triple_list = norm_para["triple_list"]

        if isinstance(para, dict):
            para["triple_list"] = triple_list
        elif isinstance(para, list):
            if len(para) >= 3:
                para[2] = triple_list
            else:
                out[target_key][idx] = list(para) + [triple_list]
        elif isinstance(para, tuple):
            items = list(para[:2])
            items.append(triple_list)
            out[target_key][idx] = items
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
                "triple_list": para.get("triple_list", []) or [],
            }
        )

    out["context"] = normalized_context
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
            if not cfg.verbose:
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
# V4.4 overrides: four-stage workflow
# Stage 1: semantic fact extraction
# Stage 2: optional answer-aware graph revision
# Stage 3: triple normalization + forward KV only
# Stage 4: reverse KV only
# ============================================================

STAGE2_REVISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "_id": {"type": "string"},
        "answer_sufficient": {"type": "boolean"},
        "missing_links": {
            "type": "array",
            "items": {"type": "string"},
        },
        "revision_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entity_id": {"type": "string"},
                    "name": {"type": "string"},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["entity_id", "name", "aliases"],
            },
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "triple_id": {"type": "string"},
                    "head": {"type": "string"},
                    "relation": {"type": "string"},
                    "tail": {"type": "string"},
                    "tail_kind": {"type": "string", "enum": ["entity", "literal"]},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title": {"type": "string"},
                                "sentence_id": {"type": "integer"},
                            },
                            "required": ["title", "sentence_id"],
                        },
                    },
                },
                "required": [
                    "triple_id",
                    "head",
                    "relation",
                    "tail",
                    "tail_kind",
                    "evidence",
                ],
            },
        },
    },
    "required": ["_id", "answer_sufficient", "missing_links", "revision_notes", "entities", "triples"],
}

STAGE2_REVISION_SYSTEM_PROMPT = """You are an answer-aware graph revision engine.

Your task is to inspect a stage-1 evidence-grounded graph, determine whether it is sufficient to answer the question, and minimally revise it when needed.

Inputs you receive:
- question
- answer
- pages
- stage1 entities
- stage1 triples

Goals:
1. Judge whether the current graph is sufficient to support answering the question.
2. If the graph is already sufficient, keep it as stable as possible.
3. If the graph is insufficient, repair it by adding, removing, disambiguating, or rewriting entities/triples strictly based on the provided context pages.
4. Preserve bridge entities and bridge relations needed for multi-hop reasoning.
5. Never hallucinate facts or create unsupported links just to make the answer reachable.

Revision policy:
- Prefer minimal edits over aggressive rewriting.
- Focus on grounded question-side entities, answer-side entities/values, and bridge nodes that connect them.
- When names are ambiguous, make the graph more specific rather than more vague.
- If stage1 missed an explicit supporting fact from the pages, add it.
- If a stage1 triple is unsupported, malformed, duplicated, or blocks correct reasoning due to ambiguity, you may remove or rewrite it.
- Every output triple must remain evidence-grounded.

Sufficiency policy:
- Set answer_sufficient=true only when the graph contains enough grounded entities and relations to support the question-answer connection.
- Set answer_sufficient=false if key bridge facts are still missing from the provided pages.
- missing_links should briefly name what is missing if sufficiency is false; otherwise return an empty list.
- revision_notes should summarize the main edits you made; if no edits are needed, use an empty list.

Output rules:
- Return valid JSON only.
- Follow the supplied schema exactly.
- Do not include explanations outside the JSON fields.
- Do not include markdown.
"""

# Stage2 不再做过严的字面核对
# 优先判断核心 answer chain 是否已经成立
# 允许 question 中存在未完全落地的修饰语
# 在不 hallucinate 的前提下，尽量把已有 graph 修成更适合多跳 QA 和后续 KV 生成的结构
STAGE2_REVISION_SYSTEM_PROMPT_v2 = """You are an answer-aware graph revision engine.

Your task is to revise an evidence-grounded stage-1 graph for question answering.

You are given:
- the question
- the answer
- the supporting facts / pages
- a stage-1 graph extracted from the provided context

Your job is to determine whether the current graph is sufficient to support answering the question, and if needed, revise the graph conservatively using ONLY the provided context.

Core objective:
- Preserve or improve the graph so that it better supports multi-hop question answering and downstream KV generation.
- Prioritize the core answer-bearing chain and the bridge structure needed to connect the question target to the answer.
- Do NOT hallucinate facts, entities, locations, or links not supported by the provided pages.

Important principle: answer sufficiency is about the core answer chain, not strict literal coverage.
- Do NOT mark the graph insufficient merely because some question phrase, modifier, or location wording is not explicitly copied into the graph.
- A question may use paraphrased, indirect, compressed, or partially underspecified wording.
- If the existing graph already supports the core entity/event alignment and the answer-bearing path, then the graph can still be sufficient even if some surface wording in the question is not explicitly grounded node-by-node.
- Distinguish carefully between:
  1. missing core evidence that truly prevents answering the question
  2. missing surface-form alignment for a question phrase
- Only case (1) should cause answer_sufficient = false.

How to judge answer sufficiency:
- Read the question and answer carefully.
- Identify the core answer target: what must be found to produce the answer.
- Identify the minimum bridge chain needed to connect the question-side target to the answer-side fact.
- If that bridge chain is already supported by the stage-1 graph and grounded in the provided context, mark answer_sufficient = true even if some secondary modifier from the question is not explicitly represented as its own node or triple.
- Use a strict standard against hallucination, but not an overly literal standard against paraphrase.

Revision policy:
- Revise only when doing so improves answerability, graph faithfulness, or structural usefulness.
- Prefer conservative repair over unnecessary rewriting.
- If the stage-1 graph is already sufficient, keep it mostly stable.
- If the graph is insufficient, add or revise only facts that are explicitly supported by the provided pages.
- Never invent a bridge fact just to make the graph answer the question.

What kinds of revisions are encouraged:
1. Bridge-chain completion
- Add or revise grounded bridge triples that connect the question-side entity/event to the answer-bearing fact.
- Preserve intermediate nodes that are useful for multi-hop reasoning.

2. Structural sharpening
- Prefer reusable graph nodes and short bridge relations over long descriptive literal tails when supported by the context.
- If a compound expression can be represented more usefully as a graph path, prefer the graph path.
- Example: if the context supports an intermediate place node, prefer:
  work -> takes place in -> place
  place -> located in -> state
  instead of collapsing everything into one large tail string.

3. Alias and paraphrase alignment
- If the question refers to an entity/event/person using wording that is clearly supported by aliases or nearby grounded facts in the provided context, preserve the grounded entity even if the wording is not identical.
- Do not require exact lexical overlap between the question and the graph when the intended grounded referent is clear from the provided context.

4. De-emphasizing low-utility triples
- Long, overly specific, weakly reusable literal tails are allowed if explicitly supported, but they should not replace shorter, more reusable bridge facts.
- If a long literal is not useful for multi-hop reasoning and a cleaner grounded structure is available, prefer the cleaner structure.

When NOT to mark insufficient:
- Do not mark insufficient solely because a question-side modifier is not explicitly present as a node.
- Do not mark insufficient solely because the question uses a paraphrase of an event or role while the graph already grounds the corresponding entity/event chain.
- Do not mark insufficient solely because one location/detail in the question is not explicitly repeated in the graph, if the existing grounded chain still identifies the answer correctly.

When to mark insufficient:
- Mark answer_sufficient = false only if the provided context truly lacks the grounded facts needed to connect the question target to the answer.
- If the answer would require inventing an unsupported identity resolution, unsupported event equivalence, unsupported location link, or unsupported relation, then mark insufficient.
- In this case, explain the missing core evidence clearly in missing_links and revision_notes.

Entity and triple revision rules:
- Keep entities canonical, specific, and reusable.
- Keep relations short, faithful, and close to the wording supported by the context.
- Prefer graph-friendly relations over vague relations like "was" when a more specific grounded relation is available.
- Preserve evidence for every triple.
- Every added or revised triple must be explicitly supported by the provided pages.
- Do not output duplicate triples.
- Preserve useful aliases when explicitly supported.

Output expectations:
- Return the revised graph.
- Keep the graph stable if no meaningful revision is needed.
- Set answer_sufficient based on core answerability, not strict lexical matching.
- Use missing_links only for genuinely missing core evidence.
- Use revision_notes to briefly explain key revision decisions, especially when a question phrase is only partially grounded but the graph is still sufficient.

Output rules:
- Return valid JSON only.
- Follow the supplied schema exactly.
- Do not include markdown.
- Do not include explanations outside the JSON fields.
"""

# 判断当前 graph 是否已经支持答案
# 如果支持，就把核心桥接链显式化、结构化、KV-friendly 化
# 如果不支持，只指出真正阻断答案的缺失证据，而不是因为题面措辞不完全对齐就判死刑
STAGE2_REVISION_SYSTEM_PROMPT_v3 = """You are a question-oriented graph revision engine.

Your job is to revise a stage-1 evidence-grounded graph so that it becomes a better graph for:
1. answering the given question faithfully from the provided context
2. supporting downstream key-value generation
3. supporting multi-hop reasoning with explicit bridge structure

You are given:
- the question
- the answer
- the supporting pages / evidence context
- a stage-1 graph extracted from that context

You must work ONLY with the provided context and the stage-1 graph.
Do NOT hallucinate facts, entities, identities, locations, events, or relations that are not explicitly supported.

Your responsibilities
1. Determine whether the graph is sufficient to support the answer.
2. If it is sufficient, improve the graph structure when useful.
3. If it is not sufficient, identify only the genuinely missing core evidence.

The main principle
Judge answer sufficiency by grounded answerability through the graph, not by exact lexical overlap with the question.

This means:
- The graph does NOT need to copy every question phrase literally.
- The question may use paraphrase, compression, indirect wording, or role descriptions.
- If the core referent and the answer-bearing path are already grounded in the provided context, the graph may be sufficient.

But:
- A question-side modifier matters if it can change which entity, event, place, or time is being referred to.
- If a missing modifier could introduce a different plausible referent, then it is blocking.
- If the remaining grounded evidence still uniquely identifies the intended referent, then the modifier is non-blocking.

Your revision goal
Do not treat Stage2 as only a verifier.
Treat Stage2 as a graph-cleaning and graph-explicating step for QA.

Even when answer_sufficient = true, you should still revise the graph if doing so makes the core answer chain:
- more explicit
- more reusable
- more graph-structured
- more suitable for downstream KV generation

What a good revised graph looks like
A good revised graph should:
- preserve the core answer-bearing chain
- preserve or add useful intermediate bridge nodes
- prefer short, specific, reusable relations
- prefer explicit graph paths over implicit reasoning hidden only in notes
- avoid long, weakly reusable literal tails when a cleaner grounded structure is available
- stay faithful to the provided context

Preferred revision behavior

A. Make the core bridge chain explicit
If answering the question relies on an intermediate bridge relation that is directly supported by the context, represent it as an explicit triple whenever possible.

Do not leave an important bridge step only in revision_notes if it can be grounded as a triple.

B. Improve graph structure even when the graph is already sufficient
If the current graph can answer the question but the answer path is implicit, overly compressed, or not KV-friendly, revise it into a cleaner explicit graph.

C. Prefer reusable graph structure over collapsed tails
If a compound span can be represented as a grounded multi-step path, prefer the path.
For example, prefer:
work -> takes place in -> city
city -> located in -> state
over a single less reusable tail string when the context supports the intermediate node structure.

D. Make directly supported symmetric or jointly-held relations explicit when useful
If the context directly states that multiple participants jointly hold a relation, and making that explicit helps QA or downstream KV generation, add the explicit relation for each participant when supported.
For example, if the context states that X co-founded Y with Z, it is acceptable to make explicit that Z also co-founded Y, if this is directly supported by the wording.

E. Keep paraphrase grounded, not literal
Do not reject a graph simply because the question wording differs from the graph wording.
Resolve paraphrase to grounded entities/events when the intended referent is clear from the provided context.

How to decide answer_sufficient

Set answer_sufficient = true when:
- the core referent is grounded
- the answer-bearing fact is grounded
- the graph already contains, or can be revised from the provided context into, a complete explicit bridge chain without inventing unsupported facts

Set answer_sufficient = false when:
- a core referent cannot be uniquely grounded
- a core bridge step is truly missing from the provided context
- the answer-bearing fact itself is missing
- answering would require inventing an unsupported identity resolution, event equivalence, place link, time link, or relation

Important distinction
Only missing core evidence should make the graph insufficient.
Missing surface wording alone should not.

What to revise

You may:
- add explicitly supported triples
- rewrite overly vague relations into more specific grounded relations
- replace low-utility long literal facts with shorter, more reusable grounded structure
- preserve helpful aliases
- keep extra non-harmful triples if they are grounded

You should not:
- invent facts
- over-expand irrelevant side details
- keep a graph unnecessarily verbose if a cleaner grounded structure is available
- rely on revision_notes to carry reasoning that should instead be represented in triples

How to use revision_notes
Use revision_notes only to briefly explain:
- why the graph is sufficient or insufficient
- which core bridge chain supports the answer
- why a modifier was judged blocking or non-blocking
- what important structural cleanup you performed

Do not use revision_notes as a substitute for graph structure.

Output standards
- Return a revised graph that is faithful to the provided context.
- Keep the graph stable when no meaningful revision is needed.
- But do perform beneficial structural revisions when they improve explicitness and downstream usefulness.
- Use missing_links only for genuinely missing core evidence.
- Be conservative against hallucination, but proactive about making grounded bridge structure explicit.

Output rules
- Return valid JSON only.
- Follow the provided schema exactly.
- Do not include markdown.
- Do not include explanations outside the JSON fields.
"""

# 比v3更克制
STAGE2_REVISION_SYSTEM_PROMPT_v4 = """You are a question-oriented graph refinement engine.

You are given:
- a question
- an answer
- evidence pages
- a stage-1 graph extracted from those pages

Your task is to refine the stage-1 graph into a better graph for question answering.

You must use ONLY the provided evidence pages.
Do NOT hallucinate any fact, entity, location, event, identity, time, or relation.

Your goals are:
1. decide whether the graph is sufficient to support the answer
2. if it is sufficient, make the answer path explicit when helpful
3. if it is not sufficient, identify only the truly missing core evidence
4. keep the graph faithful, minimal, and structurally useful for downstream KV generation

Core standard
Judge sufficiency by whether the graph can support the answer through grounded reasoning from the provided evidence.

Do NOT require literal overlap between the question and the graph.
The question may use paraphrase, role descriptions, indirect wording, or compressed phrasing.

However, do NOT ignore a question modifier if it can change the intended referent.
A missing modifier is blocking only when it creates real ambiguity about which entity, event, place, or time the question refers to.

Main working rule
Refine the graph toward the smallest explicit grounded answer chain.

This means:
- keep the core answer-bearing path
- make important bridge steps explicit when directly supported
- avoid leaving essential reasoning only in notes
- avoid unnecessary extra bridges
- avoid over-rewriting the graph just to make the path shorter
- preserve the meaning of the evidence faithfully

What to optimize for
A good revised graph should be:
- faithful to the evidence
- sufficient for the question
- explicit at the critical bridge steps
- compact rather than bloated
- reusable for downstream KV generation
- natural in its relations

Preferred behavior

1. Preserve the core answer chain
Identify the smallest set of entities and triples that connect the question target to the answer.
Preserve that chain clearly.

2. Make critical bridge steps explicit
If a bridge step is directly supported by the evidence and is necessary or strongly useful for the answer path, represent it as a triple.
Do not leave a necessary bridge only implicit in revision_notes.

3. Prefer minimal explicit structure
Add the minimum grounded structure needed to clarify the answer path.
Do not add an extra relation if existing or cleaner revised triples already make the path explicit.

4. Prefer natural relations
Use short, faithful, semantically natural relations.
Do not create awkward or overly task-specific relations just to force a direct path.

5. Prefer faithful restructuring over aggressive rewriting
You may split an overly long or low-utility fact into cleaner grounded subfacts when the evidence clearly supports that split.
But do not rewrite a fact into a cleaner-looking structure if that changes or blurs the original meaning.

6. Prefer reusable nodes over collapsed spans when clearly supported
If the evidence directly supports an intermediate reusable node, you may represent:
work -> place
place -> region
instead of collapsing everything into one tail.
But do this only when the intermediate structure is directly grounded and genuinely useful.

7. Materialize directly supported symmetric or joint roles when useful
If the evidence directly states a joint relation such as co-founding, co-authorship, joint membership, or similar, you may make the role explicit for each participant when this improves the answer path and remains faithful to the evidence.

When answer_sufficient should be true
Set answer_sufficient = true when:
- the intended referent can be grounded from the evidence
- the answer-bearing fact is grounded
- the revised graph can express a complete answer path without inventing unsupported facts

When answer_sufficient should be false
Set answer_sufficient = false only when:
- the intended referent cannot be uniquely grounded
- a core bridge step is missing from the evidence
- the answer-bearing fact is missing from the evidence
- answering would require unsupported identity resolution, event equivalence, location linking, time linking, or relation invention

Important boundary
Do not confuse missing wording with missing evidence.
A graph may be sufficient even if the question wording is not mirrored literally.

But also:
do not treat a possibly referent-changing modifier as harmless unless the remaining grounded evidence still uniquely determines the target.

What you may revise
You may:
- add directly supported triples
- split an overly compressed fact into cleaner grounded subfacts
- replace vague relations with more specific grounded ones
- keep useful aliases
- remove or de-emphasize low-utility facts if they are not important to the answer path

What you should avoid
Do not:
- invent facts
- invent bridge relations that are not naturally supported
- over-expand irrelevant side details
- keep important reasoning only in notes
- rewrite evidence into a tidier but less faithful meaning
- add extra bridge triples that are redundant with already explicit structure

How to use revision_notes
Use revision_notes only to briefly record:
- why the graph is sufficient or insufficient
- what the core answer chain is
- what key structural refinements were made
- why a modifier was judged blocking or non-blocking

If you mention a bridge step in revision_notes, it should match the revised graph.

Output expectations
Return a revised graph that is:
- evidence-grounded
- sufficient if possible
- minimally explicit at the key answer path
- faithful in meaning
- useful for downstream KV generation

Keep the graph stable when no meaningful refinement is needed.
But when a small faithful refinement clearly improves the answer path, make that refinement.

Output rules
- Return valid JSON only.
- Follow the schema exactly.
- Do not include markdown.
- Do not include explanations outside the JSON fields.
"""

STAGE3_FORWARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "_id": {"type": "string"},
        "context": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "sentences": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "triple_list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string", "enum": ["ATTRIBUTE", "RELATION"]},
                                "name": {"type": "string"},
                                "description_type": {"type": "string"},
                                "description": {"type": "string"},
                                "kv_lists": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "key_string": {"type": "string"},
                                            "value_string": {"type": "string"},
                                        },
                                        "required": ["key_string", "value_string"],
                                    },
                                },
                            },
                            "required": [
                                "type",
                                "name",
                                "description_type",
                                "description",
                                "kv_lists",
                            ],
                        },
                    },
                },
                "required": ["title", "sentences", "triple_list"],
            },
        },
    },
    "required": ["_id", "context"],
}

STAGE4_REVERSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "_id": {"type": "string"},
        "context": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "reverse_list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "triple_id": {"type": "string"},
                                "key_string": {"type": "string"},
                                "value_string": {"type": "string"},
                            },
                            "required": ["triple_id", "key_string", "value_string"],
                        },
                    },
                },
                "required": ["title", "reverse_list"],
            },
        },
    },
    "required": ["_id", "context"],
}



STAGE3_FORWARD_SYSTEM_PROMPT = """You are a normalization engine for DAG-KV / KBLaM triple construction.

Your task is to convert the provided stage-1 graph triples into dataset-compatible triples and generate ONLY the FORWARD KV for each triple.

Primary goals:
1. Preserve all valid stage-1 triples.
2. Convert each stage-1 triple into exactly one dataset triple.
3. Classify each triple as ATTRIBUTE or RELATION.
4. Generate exactly one FORWARD KV per triple.

============================================================
I. TRIPLE SCHEMA
============================================================

Each triple must follow this schema:
{
  "type": "ATTRIBUTE" or "RELATION",
  "name": "<subject entity>",
  "description_type": "<relation or attribute label>",
  "description": "<object entity or literal value>",
  "kv_lists": [
    {"key_string": "...", "value_string": "..."}
  ]
}

============================================================
II. ATTRIBUTE VS RELATION
============================================================

- Use ATTRIBUTE when the object is literal-like:
  date, year, category, profession label, nickname, time span, quantity, short descriptive phrase, etc.
- Use RELATION when the object is entity-like:
  person, place, organization, work, event, species, group, or another graph node.
- Use stage1.object_kind as the primary signal, but keep compatibility with the dataset style.

============================================================
III. NAMING RULES
============================================================

- Keep name exactly as the subject entity surface form.
- Keep description exactly as the target object/value surface form.
- Prefer short stable description_type values such as:
  birth date, death date, release year, production location, nickname, occupation, nationality,
  starring, directed by, based on, composed by, partnered with, located in, voice type, type.
- Choose description_type values that remain natural in BOTH forward and reverse KV generation.
- Prefer canonical relation labels whose reverse stays semantically clear, such as:
  directed by, published by, located in, based on, takes place in, played by, named after.
- Avoid relation labels whose reverse would require awkward role inversion or encourage forward-style rewriting,
  such as: published, made, did, had, became, was when a more specific grounded relation is available.
- When multiple equivalent phrasings are supported, prefer the one that keeps the object/value in a natural reverse anchor.

EXAMPLE: REVERSE-FRIENDLY RELATION LABEL
- Prefer:
  name="Navajivan", description_type="published by", description="Mohandas Karamchand Gandhi"
- Avoid:
  name="Mohandas Karamchand Gandhi", description_type="published", description="Navajivan"

EXAMPLE: ANOTHER REVERSE-FRIENDLY LABEL
- Prefer:
  name="Sheyann Webb", description_type="played by", description="Jurnee Smollett"
- Avoid replacing it with a less stable label such as:
  name="Jurnee Smollett", description_type="played", description="Sheyann Webb"

============================================================
IV. FORWARD KV RULES
============================================================

For EACH triple, generate exactly ONE KV pair:
- key_string = a natural semantic query or statement fragment anchored on name
- value_string = description

The forward key_string MUST:
- preserve the subject anchor (the name, or a faithful surface form of it)
- preserve the meaning of description_type
- point from the subject toward the missing target phrase
- be concise, stable, and retrieval-friendly
- not invent information
- not contain the full target phrase when avoidable

GOOD EXAMPLE:
name="Norway", description_type="population date", description="August 2018"
[
  {"key_string": "the population date of Norway is", "value_string": "August 2018"}
]

============================================================
V. OUTPUT RULES
============================================================

- Return valid JSON only.
- Follow the supplied schema exactly.
- Do not include explanations.
- Do not include markdown.
- Do not drop stage-1 triples.
- Do not merge different stage-1 triples.
- Do not generate reverse KV in this stage.
"""

STAGE4_REVERSE_SYSTEM_PROMPT = """You generate ONLY reverse KV pairs for already-finalized triples.

Your job:
For each input triple, produce exactly one reverse KV item:
- triple_id
- key_string
- value_string

The reverse KV must help the model retrieve the ORIGINAL SUBJECT (the triple.name)
from the OBJECT / ATTRIBUTE SIDE (the triple.description).

==================================================
CORE DEFINITION OF REVERSE
==================================================

Given a triple with:
- name = SUBJECT
- description_type = RELATION / ATTRIBUTE LABEL
- description = OBJECT / ATTRIBUTE VALUE

You must reverse the retrieval direction.

That means:
- value_string MUST be exactly the original SUBJECT, i.e. exactly triple.name
- key_string MUST be a query anchored on the OBJECT / ATTRIBUTE VALUE side,
  i.e. the query should use triple.description as the anchor of retrieval

Think of the reverse KV as answering:

"Given this description-side clue, who / what is the subject?"

NOT:

"Given the subject, what is the description?"

If the output still asks for the description/object/location/date/etc. of the subject,
then it is still FORWARD and therefore wrong.

==================================================
HARD CONSTRAINTS
==================================================

1. value_string MUST equal triple.name exactly
- copy it exactly
- do not paraphrase
- do not shorten
- do not normalize
- do not replace with description
- do not output any other entity

2. key_string MUST be reverse-oriented
It must retrieve the subject from the description side.

3. key_string MUST be anchored by triple.description
- the semantic center of the query must be the description side
- the query should sound like:
  "the person/place/work/event/... associated with [description] is"
  or an equivalent reverse phrasing

4. key_string MUST NOT simply restate the forward direction
The following pattern is wrong:
- subject-centered question asking for description
Examples of wrong patterns:
- "the place where Bloody Sunday took place is"
- "the party that Jones is a member of is"
- "the date Selma, Lord, Selma premiered on is"
These still retrieve the description, not the subject.

5. key_string should be fluent and concise
- natural English
- one single retrieval prompt
- no extra explanation
- no quotation marks
- no bullet markers
- no trailing punctuation required, but ending with "is" is preferred

==================================================
MANDATORY GENERATION PROCEDURE
==================================================

For every triple, follow these steps strictly:

Step 1. Copy value_string
Set:
value_string = triple.name

Step 2. Identify the description-side anchor
The anchor is:
triple.description

Step 3. Build a reverse retrieval query
Write key_string so that:
- it uses the description side as the clue
- it asks for / points to the subject side
- it does NOT ask for the object side again

Step 4. Final self-check
Before finalizing, verify all of the following:
- Does value_string exactly equal triple.name?
- Is key_string centered on triple.description rather than triple.name?
- If someone reads key_string alone, would the answer be the subject?
- Is this truly reverse rather than forward?

If any answer is "no", rewrite key_string.

==================================================
GOOD INTUITION
==================================================

Forward:
- "Bloody Sunday took place in" -> "Selma"

Reverse:
- "the event that took place in Selma is" -> "Bloody Sunday"

Forward:
- "Jones is a member of" -> "Democratic Party"

Reverse:
- "the person who is a member of Democratic Party is" -> "Jones"

Forward:
- "Selma, Lord, Selma premiered on" -> "January 17, 1999"

Reverse:
- "the film that premiered on January 17, 1999 is" -> "Selma, Lord, Selma"

==================================================
CANONICAL EXAMPLES
==================================================

Example 1
Input triple:
{
  "triple_id": "T1",
  "type": "RELATION",
  "name": "Bloody Sunday",
  "description_type": "took place in",
  "description": "Selma"
}
Correct output:
{
  "triple_id": "T1",
  "key_string": "the event that took place in Selma is",
  "value_string": "Bloody Sunday"
}
Why:
- value_string is exactly the subject
- key_string is anchored on "Selma"
- it retrieves the event, not the place

Wrong output:
{
  "triple_id": "T1",
  "key_string": "the place where Bloody Sunday took place is",
  "value_string": "Selma"
}
Why wrong:
- value_string is not the subject
- query is still asking for the place
- this is forward, not reverse

Example 2
Input triple:
{
  "triple_id": "T2",
  "type": "RELATION",
  "name": "Jones",
  "description_type": "is a member of",
  "description": "Democratic Party"
}
Correct output:
{
  "triple_id": "T2",
  "key_string": "the person who is a member of Democratic Party is",
  "value_string": "Jones"
}

Wrong output:
{
  "triple_id": "T2",
  "key_string": "the party that Jones is a member of is",
  "value_string": "Democratic Party"
}
Why wrong:
- still retrieving the object, not the subject

Example 3
Input triple:
{
  "triple_id": "T3",
  "type": "ATTRIBUTE",
  "name": "Howell Heflin",
  "description_type": "retired in",
  "description": "1997"
}
Correct output:
{
  "triple_id": "T3",
  "key_string": "the person who retired in 1997 is",
  "value_string": "Howell Heflin"
}

Example 4
Input triple:
{
  "triple_id": "T4",
  "type": "ATTRIBUTE",
  "name": "Selma, Lord, Selma",
  "description_type": "premiered on",
  "description": "January 17, 1999"
}
Correct output:
{
  "triple_id": "T4",
  "key_string": "the film that premiered on January 17, 1999 is",
  "value_string": "Selma, Lord, Selma"
}

Example 5
Input triple:
{
  "triple_id": "T5",
  "type": "RELATION",
  "name": "Selma, Lord, Selma",
  "description_type": "directed by",
  "description": "Charles Burnett"
}
Correct output:
{
  "triple_id": "T5",
  "key_string": "the film directed by Charles Burnett is",
  "value_string": "Selma, Lord, Selma"
}

Example 6
Input triple:
{
  "triple_id": "T6",
  "type": "RELATION",
  "name": "Sheyann Webb",
  "description_type": "portrayed by",
  "description": "Jurnee Smollett"
}
Correct output:
{
  "triple_id": "T6",
  "key_string": "the character portrayed by Jurnee Smollett is",
  "value_string": "Sheyann Webb"
}

==================================================
SPECIAL HANDLING RULES
==================================================

A. Dates / years / numbers
Use the date/year/number as the anchor and retrieve the subject.
Examples:
- "the person born in 1965 is"
- "the film released in 1999 is"

B. Locations
Use the location as the anchor and retrieve the subject.
Examples:
- "the event that took place in Selma is"
- "the place located in Alabama is"
- "the film that takes place in Selma is"

C. Membership / affiliation / role
Use the organization / group / role side as anchor.
Examples:
- "the person who is a member of Democratic Party is"
- "the person who became U.S. Senator from Alabama is"

D. Creative-work metadata
Use director / actor / channel / date / source material as the anchor.
Examples:
- "the film directed by Charles Burnett is"
- "the film that premiered as a television movie on ABC is"

E. Descriptive attributes
If description is an attribute-like noun phrase, key_string should still retrieve the subject.
Examples:
- "the 1999 American film is"
- "the 11-year-old African-American girl is"
Only do this when it is reasonably natural and clearly points back to the subject.

==================================================
OUTPUT FORMAT
==================================================

Return valid JSON only.

Schema:
{
  "_id": "...",
  "reverse_kv_list": [
    {
      "triple_id": "T00001",
      "key_string": "...",
      "value_string": "..."
    }
  ]
}

Rules:
- Output one reverse KV item for each input triple
- Preserve triple_id exactly
- value_string must equal the corresponding triple.name exactly
- Do not include explanations
- Do not include markdown
- Do not omit any triple
"""

# 更紧凑版本

STAGE4_REVERSE_SYSTEM_PROMPT_v2 = """You generate ONLY reverse KV pairs for finalized triples.

For each triple:
- name = subject
- description_type = relation / attribute label
- description = object / attribute value

Your task is to generate exactly one reverse KV item:
- triple_id
- key_string
- value_string

Core rule:
- value_string MUST equal triple.name exactly
- key_string MUST retrieve the subject from the description side

A reverse KV means:
- use triple.description as the anchor
- make the answer be triple.name

So the reverse direction is:
given description -> retrieve subject

Not:
given subject -> retrieve description

Hard rules:
1. value_string = triple.name exactly
2. key_string must be centered on triple.description
3. key_string must NOT still ask for the description/object/location/date of the subject
4. Preserve triple_id exactly
5. Output one item per input triple

Good examples:
- Bloody Sunday / took place in / Selma
  -> key_string: "the event that took place in Selma is"
  -> value_string: "Bloody Sunday"

- Jones / is a member of / Democratic Party
  -> key_string: "the person who is a member of Democratic Party is"
  -> value_string: "Jones"

- Selma, Lord, Selma / premiered on / January 17, 1999
  -> key_string: "the film that premiered on January 17, 1999 is"
  -> value_string: "Selma, Lord, Selma"

Wrong example:
- Bloody Sunday / took place in / Selma
  -> "the place where Bloody Sunday took place is" -> "Selma"
This is wrong because it is still forward, and value_string is not the subject.

Before finalizing each item, check:
- Does value_string exactly equal name?
- Is key_string anchored on description?
- Would the answer to key_string be the subject?

Return valid JSON only.

Schema:
{
  "_id": "...",
  "reverse_kv_list": [
    {
      "triple_id": "T00001",
      "key_string": "...",
      "value_string": "..."
    }
  ]
}

Do not output explanations.
Do not output markdown.
"""

def _triple_sig(tri: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        norm_text(tri.get("type", "")),
        norm_text(tri.get("name", "")),
        norm_text(tri.get("description_type", "")),
        norm_text(tri.get("description", "")),
    )


def _repair_single_kv(tri: Dict[str, Any], kv: Dict[str, str]) -> Dict[str, str]:
    tmp = {
        "name": norm_text(tri.get("name", "")),
        "description": norm_text(tri.get("description", "")),
        "kv_lists": [
            {
                "key_string": norm_text(kv.get("key_string", "")),
                "value_string": norm_text(kv.get("value_string", "")),
            }
        ],
    }
    tmp = postprocess_triple_kv_lists(tmp)
    if tmp.get("kv_lists"):
        return tmp["kv_lists"][0]
    return {
        "key_string": norm_text(kv.get("key_string", "")),
        "value_string": norm_text(kv.get("value_string", "")),
    }


def _choose_forward_kv(
    *,
    tri: Dict[str, Any],
    kvs: List[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    description = norm_text(tri.get("description", ""))
    name = norm_text(tri.get("name", ""))
    cleaned = []
    for kv in kvs:
        key_string = norm_text(kv.get("key_string", ""))
        value_string = norm_text(kv.get("value_string", ""))
        if key_string and value_string:
            cleaned.append({"key_string": key_string, "value_string": value_string})
    if not cleaned:
        return None

    preferred = [kv for kv in cleaned if contain_anchor_in_value(value_string, description) and has_anchor_in_key(kv["key_string"], name)]
    if preferred:
        return _repair_single_kv(tri, preferred[0])

    preferred = [kv for kv in cleaned if contain_anchor_in_value(value_string, description)]
    if preferred:
        return _repair_single_kv(tri, preferred[0])

    return _repair_single_kv(tri, cleaned[0])


def normalize_stage3_forward(stage3: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    out = {"_id": norm_text(stage3.get("_id", "")), "context": []}
    triple_counter = 1

    for para in stage3.get("context", []) or []:
        triples = []
        seen = set()
        for tri in para.get("triple_list", []) or []:
            tri2 = {
                "triple_id": "",
                "type": norm_text(tri.get("type", "")),
                "name": norm_text(tri.get("name", "")),
                "description_type": norm_text(tri.get("description_type", "")),
                "description": norm_text(tri.get("description", "")),
                "kv_lists": [],
            }

            candidate_kvs = []
            for kv in tri.get("kv_lists", []) or []:
                key_string = norm_text(kv.get("key_string", ""))
                value_string = norm_text(kv.get("value_string", ""))
                if key_string and value_string:
                    candidate_kvs.append({
                        "key_string": key_string,
                        "value_string": value_string,
                    })

            chosen_forward = _choose_forward_kv(tri=tri2, kvs=candidate_kvs)
            if chosen_forward is not None:
                tri2["kv_lists"] = [chosen_forward]

            sig = _triple_sig(tri2)
            if sig in seen:
                continue
            seen.add(sig)
            tri2["triple_id"] = norm_text(tri.get("triple_id", "")) or f"T{triple_counter:05d}"
            triple_counter += 1
            if verbose:
                print(f"[stage3] {tri2}")
            triples.append(tri2)

        out["context"].append(
            {
                "title": norm_text(para.get("title", "")),
                "sentences": [norm_text(x) for x in (para.get("sentences", []) or [])],
                "triple_list": triples,
            }
        )
    return out


def build_stage4_reverse_input(stage3: Dict[str, Any]) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    for para in stage3.get("context", []) or []:
        triples = []
        for tri in para.get("triple_list", []) or []:
            triples.append(
                {
                    "triple_id": norm_text(tri.get("triple_id", "")),
                    "type": norm_text(tri.get("type", "")),
                    "name": norm_text(tri.get("name", "")),
                    "description_type": norm_text(tri.get("description_type", "")),
                    "description": norm_text(tri.get("description", "")),
                }
            )
        pages.append(
            {
                "title": norm_text(para.get("title", "")),
                "sentences": [norm_text(x) for x in (para.get("sentences", []) or [])],
                "triples": triples,
            }
        )
    return {
        "_id": norm_text(stage3.get("_id", "")),
        "pages": pages,
    }


def build_empty_stage4_from_stage3(stage3: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": norm_text(stage3.get("_id", "")),
        "context": [
            {
                "title": norm_text(para.get("title", "")),
                "reverse_list": [],
            }
            for para in (stage3.get("context", []) or [])
        ],
    }


def normalize_stage4_reverse(stage4: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    out = {"_id": norm_text(stage4.get("_id", "")), "context": []}
    for para in stage4.get("context", []) or []:
        reverse_list = []
        seen_ids = set()
        for item in para.get("reverse_list", []) or []:
            triple_id = norm_text(item.get("triple_id", ""))
            key_string = norm_text(item.get("key_string", ""))
            value_string = norm_text(item.get("value_string", ""))
            if not triple_id or not key_string or not value_string:
                continue
            if triple_id in seen_ids:
                continue
            seen_ids.add(triple_id)
            reverse_item = {
                "triple_id": triple_id,
                "key_string": key_string,
                "value_string": value_string,
            }
            if verbose:
                print(f"[stage4] {reverse_item}")
            reverse_list.append(reverse_item)
        out["context"].append(
            {
                "title": norm_text(para.get("title", "")),
                "reverse_list": reverse_list,
            }
        )
    return out


def validate_triple_core(tri: Dict[str, Any]) -> None:
    if not isinstance(tri, dict):
        raise ValueError("triple must be object")
    if tri.get("type") not in {"ATTRIBUTE", "RELATION"}:
        raise ValueError("triple.type invalid")
    for k in ["name", "description_type", "description"]:
        if not isinstance(tri.get(k), str) or not tri[k].strip():
            raise ValueError(f"triple.{k} invalid")


def validate_forward_kv_only(tri: Dict[str, Any]) -> None:
    validate_triple_core(tri)
    kvs = tri.get("kv_lists")
    if not isinstance(kvs, list) or len(kvs) != 1:
        raise ValueError("forward triple.kv_lists must contain exactly 1 item")
    kv = kvs[0]
    key_string = norm_text(kv.get("key_string", ""))
    value_string = norm_text(kv.get("value_string", ""))
    name = norm_text(tri.get("name", ""))
    description = norm_text(tri.get("description", ""))
    if not key_string or not value_string:
        raise ValueError("forward kv missing key/value")
    if not contain_anchor_in_value(value_string, description):
        raise ValueError(
            f"forward kv value_string must description: expected={description!r} got={value_string!r}"
        )
    if not has_anchor_in_key(key_string, name):
        raise ValueError(
            f"forward key_string missing subject anchor: key={key_string!r}, name={name!r}, description={description!r}"
        )


def validate_reverse_item_against_triple(item: Dict[str, Any], tri: Dict[str, Any]) -> None:
    triple_id = norm_text(item.get("triple_id", ""))
    key_string = norm_text(item.get("key_string", ""))
    value_string = norm_text(item.get("value_string", ""))
    name = norm_text(tri.get("name", ""))
    description = norm_text(tri.get("description", ""))
    if not triple_id:
        raise ValueError("reverse item triple_id invalid")
    if not key_string or not value_string:
        raise ValueError("reverse item key/value invalid")
    if not contain_anchor_in_value(value_string, name):
        raise ValueError(
            f"reverse kv value_string must contain name: triple_id={triple_id} expected={name!r} got={value_string!r}"
        )
    if not has_anchor_in_key(key_string, description):
        raise ValueError(
            f"reverse key_string missing object/value anchor: key={key_string!r}, name={name!r}, description={description!r}"
        )



def validate_stage3_forward(stage3: Dict[str, Any], verbose: bool = False) -> None:
    if not isinstance(stage3, dict):
        raise ValueError("stage3 must be a dict")
    if not isinstance(stage3.get("_id"), str):
        raise ValueError("stage3._id missing or invalid")
    if not isinstance(stage3.get("context"), list):
        raise ValueError("stage3.context missing or invalid")

    seen_triple_ids = set()
    for para in stage3["context"]:
        if not isinstance(para, dict):
            raise ValueError("stage3 context item must be object")
        if not isinstance(para.get("title"), str):
            raise ValueError("stage3 context.title invalid")
        if not isinstance(para.get("sentences"), list):
            raise ValueError("stage3 context.sentences invalid")
        if not isinstance(para.get("triple_list"), list):
            raise ValueError("stage3 context.triple_list invalid")
        for tri in para["triple_list"]:
            triple_id = norm_text(tri.get("triple_id", ""))
            if not triple_id:
                raise ValueError("stage3 triple_id invalid")
            if triple_id in seen_triple_ids:
                raise ValueError(f"duplicate stage3 triple_id: {triple_id}")
            seen_triple_ids.add(triple_id)
            validate_forward_kv_only(tri)


def filter_invalid_stage3_forward(
    stage3: Dict[str, Any],
    *,
    sample_id: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    out = {"_id": norm_text(stage3.get("_id", "")), "context": []}
    seen_triple_ids = set()
    dropped = 0

    for para in stage3.get("context", []) or []:
        title = norm_text(para.get("title", ""))
        kept_triples = []
        for tri in para.get("triple_list", []) or []:
            triple_id = norm_text(tri.get("triple_id", ""))
            try:
                if not triple_id:
                    raise ValueError("stage3 triple_id invalid")
                if triple_id in seen_triple_ids:
                    raise ValueError(f"duplicate stage3 triple_id: {triple_id}")
                validate_forward_kv_only(tri)
            except Exception as e:
                dropped += 1
                print(
                    f"[{now_ts()}] WARN sample={sample_id} stage=stage3 "
                    f"title={title!r} triple_id={triple_id or '<missing>'} dropped_invalid_forward {e}",
                    file=sys.stderr,
                )
                continue
            seen_triple_ids.add(triple_id)
            kept_triples.append(tri)

        out["context"].append(
            {
                "title": title,
                "sentences": [norm_text(x) for x in (para.get("sentences", []) or [])],
                "triple_list": kept_triples,
            }
        )

    if verbose and dropped:
        print(f"[stage3_filter] sample={sample_id} dropped_invalid_forward={dropped}")
    return out


def validate_stage4_reverse(stage3: Dict[str, Any], stage4: Dict[str, Any], verbose: bool = False) -> None:
    if not isinstance(stage4, dict):
        raise ValueError("stage4 must be a dict")
    if norm_text(stage4.get("_id", "")) != norm_text(stage3.get("_id", "")):
        raise ValueError("stage4._id does not match stage3._id")
    if not isinstance(stage4.get("context"), list):
        raise ValueError("stage4.context missing or invalid")

    stage3_by_title = {norm_text(p.get("title", "")): p for p in (stage3.get("context", []) or [])}
    stage4_by_title = {norm_text(p.get("title", "")): p for p in (stage4.get("context", []) or [])}

    if set(stage4_by_title.keys()) != set(stage3_by_title.keys()):
        raise ValueError(
            f"stage4 titles do not match stage3 titles: stage3={sorted(stage3_by_title)} stage4={sorted(stage4_by_title)}"
        )

    for title, para3 in stage3_by_title.items():
        para4 = stage4_by_title[title]
        reverse_items = para4.get("reverse_list")
        if not isinstance(reverse_items, list):
            raise ValueError(f"stage4 reverse_list invalid for title={title!r}")
        reverse_by_id = {}
        for item in reverse_items:
            if not isinstance(item, dict):
                raise ValueError("stage4 reverse item must be object")
            triple_id = norm_text(item.get("triple_id", ""))
            if not triple_id:
                raise ValueError("stage4 reverse triple_id invalid")
            if triple_id in reverse_by_id:
                raise ValueError(f"duplicate stage4 triple_id in title={title!r}: {triple_id}")
            reverse_by_id[triple_id] = item

        expected_ids = [norm_text(t.get("triple_id", "")) for t in (para3.get("triple_list", []) or [])]
        if set(reverse_by_id.keys()) != set(expected_ids):
            raise ValueError(
                f"stage4 reverse triple_id set mismatch for title={title!r}: expected={sorted(expected_ids)} got={sorted(reverse_by_id)}"
            )
        for tri in para3.get("triple_list", []) or []:
            validate_reverse_item_against_triple(reverse_by_id[tri["triple_id"]], tri)


def filter_invalid_stage4_reverse(
    stage3: Dict[str, Any],
    stage4: Dict[str, Any],
    *,
    sample_id: str,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    filtered_stage3 = {"_id": norm_text(stage3.get("_id", "")), "context": []}
    filtered_stage4 = {"_id": norm_text(stage4.get("_id", "")), "context": []}
    stage4_by_title = {
        norm_text(para.get("title", "")): para
        for para in (stage4.get("context", []) or [])
    }
    dropped = 0

    for para3 in stage3.get("context", []) or []:
        title = norm_text(para3.get("title", ""))
        para4 = stage4_by_title.get(title, {"title": title, "reverse_list": []})

        reverse_candidates = {}
        for item in para4.get("reverse_list", []) or []:
            triple_id = norm_text(item.get("triple_id", ""))
            if triple_id and triple_id not in reverse_candidates:
                reverse_candidates[triple_id] = item

        kept_triples = []
        kept_reverse_items = []
        for tri in para3.get("triple_list", []) or []:
            triple_id = norm_text(tri.get("triple_id", ""))
            reverse_item = reverse_candidates.get(triple_id)
            if reverse_item is None:
                dropped += 1
                print(
                    f"[{now_ts()}] WARN sample={sample_id} stage=stage4 "
                    f"title={title!r} triple_id={triple_id or '<missing>'} dropped_missing_reverse",
                    file=sys.stderr,
                )
                continue
            try:
                validate_reverse_item_against_triple(reverse_item, tri)
            except Exception as e:
                dropped += 1
                print(
                    f"[{now_ts()}] WARN sample={sample_id} stage=stage4 "
                    f"title={title!r} triple_id={triple_id or '<missing>'} dropped_invalid_reverse {e}",
                    file=sys.stderr,
                )
                continue
            kept_triples.append(tri)
            kept_reverse_items.append(
                {
                    "triple_id": norm_text(reverse_item.get("triple_id", "")),
                    "key_string": norm_text(reverse_item.get("key_string", "")),
                    "value_string": norm_text(reverse_item.get("value_string", "")),
                }
            )

        filtered_stage3["context"].append(
            {
                "title": title,
                "sentences": [norm_text(x) for x in (para3.get("sentences", []) or [])],
                "triple_list": kept_triples,
            }
        )
        filtered_stage4["context"].append(
            {
                "title": title,
                "reverse_list": kept_reverse_items,
            }
        )

    if verbose and dropped:
        print(f"[stage4_filter] sample={sample_id} dropped_invalid_reverse={dropped}")
    return filtered_stage3, filtered_stage4


def validate_final_stage(stage_final: Dict[str, Any], verbose: bool = False) -> None:
    if verbose:
        return
    if not isinstance(stage_final, dict):
        raise ValueError("final stage must be a dict")
    if not isinstance(stage_final.get("_id"), str):
        raise ValueError("final._id missing or invalid")
    if not isinstance(stage_final.get("context"), list):
        raise ValueError("final.context missing or invalid")
    for para in stage_final["context"]:
        if not isinstance(para, dict):
            raise ValueError("final context item must be object")
        if not isinstance(para.get("title"), str):
            raise ValueError("final context.title invalid")
        if not isinstance(para.get("sentences"), list):
            raise ValueError("final context.sentences invalid")
        if not isinstance(para.get("triple_list"), list):
            raise ValueError("final context.triple_list invalid")
        for tri in para["triple_list"]:
            validate_triple(tri)
            validate_kv_anchor_and_value(tri)


def merge_stage3_forward_and_stage4_reverse(stage3: Dict[str, Any], stage4: Dict[str, Any]) -> Dict[str, Any]:
    stage4_by_title = {
        norm_text(para.get("title", "")): {
            norm_text(item.get("triple_id", "")): item
            for item in (para.get("reverse_list", []) or [])
        }
        for para in (stage4.get("context", []) or [])
    }

    out = {"_id": norm_text(stage3.get("_id", "")), "context": []}
    for para in stage3.get("context", []) or []:
        title = norm_text(para.get("title", ""))
        reverse_by_id = stage4_by_title.get(title, {})
        final_triples = []
        for tri in para.get("triple_list", []) or []:
            forward_kv = (tri.get("kv_lists", []) or [{}])[0]
            reverse_item = reverse_by_id.get(norm_text(tri.get("triple_id", "")), {})
            reverse_kv = _repair_single_kv(
                tri,
                {
                    "key_string": norm_text(reverse_item.get("key_string", "")),
                    "value_string": norm_text(reverse_item.get("value_string", "")),
                },
            )
            final_triples.append(
                {
                    "type": norm_text(tri.get("type", "")),
                    "name": norm_text(tri.get("name", "")),
                    "description_type": norm_text(tri.get("description_type", "")),
                    "description": norm_text(tri.get("description", "")),
                    "kv_lists": [
                        {
                            "key_string": norm_text(forward_kv.get("key_string", "")),
                            "value_string": norm_text(forward_kv.get("value_string", "")),
                        },
                        reverse_kv,
                    ],
                }
            )
        out["context"].append(
            {
                "title": title,
                "sentences": [norm_text(x) for x in (para.get("sentences", []) or [])],
                "triple_list": final_triples,
            }
        )
    return out
def merge_stage3_and_stage4(stage3: Dict[str, Any], stage4: Dict[str, Any]) -> Dict[str, Any]:
    return merge_stage3_forward_and_stage4_reverse(stage3, stage4)



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
    stage3_path = cache_path(stage_cache_subdir(cfg.stage_cache_dir, "stage3"), sample_id, "stage3")
    stage4_path = cache_path(stage_cache_subdir(cfg.stage_cache_dir, "stage4"), sample_id, "stage4")

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
        "entities": copy.deepcopy(stage1.get("entities", []) or []),
        "triples": copy.deepcopy(stage1.get("triples", []) or []),
    }
    stage2 = None if cfg.overwrite else load_cached_json(stage2_path)
    if stage2 is None:
        if cfg.answer_aware and cfg.include_answer:
            stage2_input = stage1_input.copy()
            stage2_input["stage1_graph"] = {
                "entities": copy.deepcopy(stage1.get("entities", []) or []),
                "triples": copy.deepcopy(stage1.get("triples", []) or []),
            }

            if verbose:
                print(f"[stage2_input] {stage2_input}")
            try:
                stage2 = await client.create_structured_response(
                    system_prompt=STAGE2_REVISION_SYSTEM_PROMPT_v3,
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

    # ---------------- stage 3: forward only ----------------
    stage3 = None if cfg.overwrite else load_cached_json(stage3_path)
    if stage3 is None:
        stage3_input = build_stage3_forward_input(sample, graph_for_kv)
        try:
            stage3 = await client.create_structured_response(
                system_prompt=STAGE3_FORWARD_SYSTEM_PROMPT,
                user_payload=stage3_input,
                schema_name="stage3_forward_triple_list",
                schema=STAGE3_FORWARD_SCHEMA,
                request_id=make_request_id("stage3", sample_id),
            )
        except Exception as e:
            raise SampleProcessingError("stage3_api", sample_id, e) from e
    try:
        stage3 = normalize_stage3_forward(stage3, verbose=verbose)
        # stage3 = filter_invalid_stage3_forward(stage3, sample_id=sample_id, verbose=verbose)
        validate_stage3_forward(stage3, verbose=verbose)
        save_cached_json(stage3_path, stage3)
    except Exception as e:
        raise SampleProcessingError("stage3_validation", sample_id, e) from e

    # ---------------- stage 4: reverse only ----------------
    stage4 = None if cfg.overwrite else load_cached_json(stage4_path)
    if stage4 is None:
        if sum(len(p.get("triple_list", []) or []) for p in (stage3.get("context", []) or [])) == 0:
            stage4 = build_empty_stage4_from_stage3(stage3)
        else:
            stage4_input = build_stage4_reverse_input(stage3)
            try:
                stage4 = await client.create_structured_response(
                    system_prompt=STAGE4_REVERSE_SYSTEM_PROMPT,
                    user_payload=stage4_input,
                    schema_name="stage4_reverse_kv_list",
                    schema=STAGE4_REVERSE_SCHEMA,
                    request_id=make_request_id("stage4", sample_id),
                )
            except Exception as e:
                raise SampleProcessingError("stage4_api", sample_id, e) from e
    try:
        stage4 = normalize_stage4_reverse(stage4, verbose=verbose)
        save_cached_json(stage4_path, stage4)
        # stage3, stage4 = filter_invalid_stage4_reverse(
        #     stage3,
        #     stage4,
        #     sample_id=sample_id,
        #     verbose=verbose,
        # )
        validate_stage4_reverse(stage3, stage4, verbose=verbose)
    except Exception as e:
        raise SampleProcessingError("stage4_validation", sample_id, e) from e

    try:
        stage_final = merge_stage3_and_stage4(stage3, stage4)
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
        "stage3": stage3,
        "stage4": stage4,
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
    for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
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

    cfg = OpenAIConfig(
        api_key=api_key,
        api_base=args.api_base,
        model=args.model,
        timeout=args.timeout,
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
            for wid in range(max(1, args.concurrency))
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
    ap = argparse.ArgumentParser(description="Four-stage graph-first KG extraction with OpenAI-compatible APIs")
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
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--retry-base", type=float, default=2.0)
    ap.add_argument("--retry-max", type=float, default=30.0)
    ap.add_argument("--verbosity", type=str, default="low", choices=["low", "medium", "high"])
    ap.add_argument("--no-store", action="store_true", help="Set store=false on Responses API")

    ap.add_argument("--resume", action="store_true", help="Skip samples already present in output jsonl")
    ap.add_argument("--stage-cache-dir", type=str, default=None, help="Optional root dir for per-stage caches; uses stage1/stage2/stage3/stage4 subdirs")
    ap.add_argument("--overwrite", action="store_true", help="Ignore caches and recompute")
    ap.add_argument("--error-log", type=str, default="./kg_extract_errors.log")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--sample-retries", type=int, default=2, help="Max times to requeue a failed sample after the initial attempt")

    ap.add_argument("--no-question", action="store_true", help="Do not include question in stage1 prompt")
    ap.add_argument("--no-answer", action="store_true", help="Do not include answer in stage1 prompt")
    ap.add_argument("--answer-aware", action="store_true", help="Enable answer-aware stage2 graph revision before forward/reverse KV generation")
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
