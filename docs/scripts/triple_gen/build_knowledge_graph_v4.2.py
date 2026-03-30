#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_knowledge_graph_v4.py

Two-stage KG extraction pipeline for QA datasets using OpenAI-compatible APIs.

Stage 1:
    Extract an evidence-grounded semantic fact layer from supporting pages.
Stage 2:
    Normalize stage-1 facts into dataset-compatible triple_list JSON.

Key features:
- Async concurrent requests with asyncio + urllib
- Structured Outputs via Responses API or chat.completions JSON schema
- Resume/caching for both stages
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
    python3 build_knowledge_graph_v4.py \
        --input hotpot_train.jsonl \
        --output hotpot_train_triples.jsonl \
        --model openai/gpt-5.2 \
        --api-mode chat \
        --concurrency 8 \
        --stage1-cache-dir ./cache_stage1 \
        --stage2-cache-dir ./cache_stage2
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
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fact_id": {"type": "string"},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "object_kind": {"type": "string", "enum": ["entity", "literal"]},
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
                    "fact_id",
                    "subject",
                    "predicate",
                    "object",
                    "object_kind",
                    "evidence",
                ],
            },
        },
    },
    "required": ["_id", "entities", "facts"],
}

# ============================================================
# Prompts
# ============================================================

STAGE1_SYSTEM_PROMPT = """You are a knowledge graph fact extraction engine.

Your task is to extract a complete, faithful, evidence-grounded semantic fact layer from the provided context.

Primary goal:
1. Extract all explicit facts stated in the provided context.
2. Preserve facts needed for graph connectivity and downstream multi-hop reasoning.
3. Do not hallucinate or infer unstated facts.

Extraction rules:
- Extract facts only from the provided context.
- Cover all explicit facts that may be useful for question answering or graph construction.
- Split multi-fact sentences into multiple facts.
- If one subject links to multiple objects, create multiple facts.
- Use short, stable predicate phrases.
- Keep entities canonical, specific, and retrieval-friendly.
- When entity names are ambiguous or overlapping, prefer a more specific canonical surface form.
- Avoid turning attribute phrases, possessive phrases, or clause fragments into standalone entities unless they are clearly entity-like in the text.
- Every fact must have evidence.
- evidence.title must be the page title.
- evidence.sentence_id must be the integer sentence index in that page.


object_kind rules:
- Use object_kind='entity' for person, organization, work, place, event, group, species, or other concrete graph node.
- Use object_kind='literal' for date, year, profession label, category, nickname, quantity, boolean-like status, short descriptive value, or other attribute-like value.

Deduplication rules:
- Do not output exact duplicate facts.
- If the same fact appears in multiple sentences, keep one fact and attach all evidence items.

Output rules:
- Return valid JSON only.
- Follow the supplied JSON schema exactly.
- Do not include explanations.
- Do not include markdown.
"""

STAGE1_SYSTEM_PROMPT_ANSWER_AWARE = """You are a knowledge graph fact extraction engine.

Your task is to extract a complete, faithful, evidence-grounded semantic fact layer from the provided context.

Primary goal:
1. Extract all explicit facts stated in the provided context.
2. Preserve facts needed for graph connectivity and downstream multi-hop reasoning.
3. If an answer field is provided, ensure facts directly or indirectly connected to that answer are not missed.
4. Do not hallucinate or infer unstated facts.

Extraction rules:
- Extract facts only from the provided context.
- Cover all explicit facts that may be useful for question answering or graph construction.
- Split multi-fact sentences into multiple facts.
- If one subject links to multiple objects, create multiple facts.
- Use short, stable predicate phrases.
- Keep entities canonical, specific, and retrieval-friendly.
- When entity names are ambiguous or overlapping, prefer a more specific canonical surface form.
- Avoid turning attribute phrases, possessive phrases, or clause fragments into standalone entities unless they are clearly entity-like in the text.
- Every fact must have evidence.
- evidence.title must be the page title.
- evidence.sentence_id must be the integer sentence index in that page.

object_kind rules:
- Use object_kind='entity' for person, organization, work, place, event, group, species, or other concrete graph node.
- Use object_kind='literal' for date, year, profession label, category, nickname, quantity, boolean-like status, short descriptive value, or other attribute-like value.

Deduplication rules:
- Do not output exact duplicate facts.
- If the same fact appears in multiple sentences, keep one fact and attach all evidence items.

Output rules:
- Return valid JSON only.
- Follow the supplied JSON schema exactly.
- Do not include explanations.
- Do not include markdown.
"""

def get_stage1_system_prompt(*, answer_aware: bool, include_answer: bool, sample: Optional[Dict[str, Any]] = None) -> str:
    if not answer_aware:
        return STAGE1_SYSTEM_PROMPT
    if not include_answer:
        return STAGE1_SYSTEM_PROMPT
    if sample is not None:
        answer = norm_text((sample or {}).get("answer", ""))
        if not answer:
            return STAGE1_SYSTEM_PROMPT
    return STAGE1_SYSTEM_PROMPT_ANSWER_AWARE


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
        self.config = config
        self._base_url = config.api_base.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    async def aclose(self) -> None:
        return None

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
                data = await asyncio.to_thread(
                    self._post_structured,
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

    def _post_structured(
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
            return self._post_responses(
                request_id=request_id,
                payload=self._build_responses_payload(payload, schema_name, schema),
            )
        if self.config.api_mode == "chat":
            return self._post_chat_completions(
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

    def _post_responses(self, *, request_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            url=f"{self._base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={**self._headers, "X-Client-Request-Id": request_id},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise ResponsesAPIError(f"HTTP {e.code}: {body[:2000]}") from e
        except urllib.error.URLError as e:
            raise ResponsesAPIError(f"Network error: {e}") from e

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ResponsesAPIError(
                f"Response body is not valid JSON for request_id={request_id}: {body[:1000]}"
            ) from e

    def _post_chat_completions(self, *, request_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={**self._headers, "X-Client-Request-Id": request_id},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise ResponsesAPIError(f"HTTP {e.code}: {body[:2000]}") from e
        except urllib.error.URLError as e:
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
        "supporting_facts": [
            [norm_text(x[0]), int(x[1])] for x in (sample.get("supporting_facts", []) or []) if isinstance(x, (list, tuple)) and len(x) >= 2
        ],
        "pages": pages,
    }
    if include_question:
        payload["question"] = norm_text(sample.get("question", ""))
    if include_answer:
        payload["answer"] = norm_text(sample.get("answer", ""))
    return payload


def build_stage2_input(
    sample: Dict[str, Any],
    stage1: Dict[str, Any],
) -> Dict[str, Any]:
    ctx_map = get_context_map(sample)
    supporting_titles = set(get_supporting_titles(sample))

    title_to_sentences: Dict[str, List[str]] = {}
    for title, para in ctx_map.items():
        if title in supporting_titles:
            title_to_sentences[title] = [norm_text(x) for x in (para.get("sentences", []) or [])]

    facts_by_title: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fact in stage1.get("facts", []) or []:
        for ev in fact.get("evidence", []) or []:
            title = norm_text(ev.get("title", ""))
            if title in title_to_sentences:
                facts_by_title[title].append(fact)

    pages: List[Dict[str, Any]] = []
    for title, sentences in title_to_sentences.items():
        pages.append(
            {
                "title": title,
                "sentences": sentences,
                "facts": dedupe_stage2_input_facts(facts_by_title.get(title, [])),
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


# ============================================================
# Validation and cleanup
# ============================================================


def dedupe_stage1(stage1: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "_id": norm_text(stage1.get("_id", "")),
        "entities": [],
        "facts": [],
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

    fact_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for fact in stage1.get("facts", []) or []:
        subject = norm_text(fact.get("subject", ""))
        predicate = norm_text(fact.get("predicate", ""))
        obj = norm_text(fact.get("object", ""))
        kind = norm_text(fact.get("object_kind", ""))
        if not (subject and predicate and obj and kind in {"entity", "literal"}):
            continue
        sig = (subject, predicate, obj, kind)
        evs = []
        ev_seen = set()
        for ev in fact.get("evidence", []) or []:
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

        if sig not in fact_map:
            fact_map[sig] = {
                "fact_id": norm_text(fact.get("fact_id", "")) or f"F{len(fact_map) + 1}",
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "object_kind": kind,
                "evidence": evs,
            }
        else:
            existing_seen = {(x["title"], x["sentence_id"]) for x in fact_map[sig]["evidence"]}
            for ev in evs:
                esig = (ev["title"], ev["sentence_id"])
                if esig not in existing_seen:
                    fact_map[sig]["evidence"].append(ev)
                    existing_seen.add(esig)

    out["facts"] = list(fact_map.values())
    return out


def dedupe_stage2_input_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for fact in facts:
        sig = (
            norm_text(fact.get("subject", "")),
            norm_text(fact.get("predicate", "")),
            norm_text(fact.get("object", "")),
            norm_text(fact.get("object_kind", "")),
        )
        if sig not in seen:
            out.append(
                {
                    "fact_id": norm_text(fact.get("fact_id", "")),
                    "subject": sig[0],
                    "predicate": sig[1],
                    "object": sig[2],
                    "object_kind": sig[3],
                    "evidence": fact.get("evidence", []) or [],
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
    if not isinstance(stage1.get("facts"), list):
        raise ValueError("stage1.facts missing or invalid")
    for ent in stage1["entities"]:
        if not isinstance(ent, dict):
            raise ValueError("stage1 entity must be object")
        if not isinstance(ent.get("entity_id"), str) or not ent["entity_id"].strip():
            raise ValueError("stage1 entity_id missing")
        if not isinstance(ent.get("name"), str) or not ent["name"].strip():
            raise ValueError("stage1 entity name missing")
        if not isinstance(ent.get("aliases"), list):
            raise ValueError("stage1 aliases missing")
    for fact in stage1["facts"]:
        if not isinstance(fact, dict):
            raise ValueError("stage1 fact must be object")
        for k in ["fact_id", "subject", "predicate", "object"]:
            if not isinstance(fact.get(k), str) or not fact[k].strip():
                raise ValueError(f"stage1 fact {k} missing")
        if fact.get("object_kind") not in {"entity", "literal"}:
            raise ValueError("stage1 fact object_kind invalid")
        if not isinstance(fact.get("evidence"), list) or not fact["evidence"]:
            raise ValueError("stage1 fact evidence missing")
        for ev in fact["evidence"]:
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


def has_full_anchor_leak(key_string: str, forbidden_anchor_text: str) -> bool:
    key_norm = normalize_anchor_text(key_string)
    forbidden_norm = normalize_anchor_text(forbidden_anchor_text)
    if not key_norm or not forbidden_norm:
        return False
    return forbidden_norm in key_norm


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

        if expected_forward and value_string != description:
            raise ValueError(
                f"kv_lists must alternate forward/reverse starting with forward: "
                f"index={idx} expected_value={description!r} got={value_string!r}"
            )
        if (not expected_forward) and value_string != name:
            print(tri)
            raise ValueError(
                f"kv_lists must alternate forward/reverse starting with forward: "
                f"index={idx} expected_value={name!r} got={value_string!r}"
            )

        if value_string == description:
            if not has_anchor_in_key(key_string, name):
                raise ValueError(
                    f"forward key_string missing subject anchor: key={key_string!r}, name={name!r}, description={description!r}"
                )
            if normalize_anchor_text(description) != normalize_anchor_text(name) and has_full_anchor_leak(key_string, description):
                raise ValueError(
                    f"forward key_string leaks target description into key: key={key_string!r}, name={name!r}, description={description!r}"
                )
        elif value_string == name:
            if not has_anchor_in_key(key_string, description):
                raise ValueError(
                    f"reverse key_string missing object/value anchor: key={key_string!r}, name={name!r}, description={description!r}"
                )
            if normalize_anchor_text(name) != normalize_anchor_text(description) and has_full_anchor_leak(key_string, name):
                raise ValueError(
                    f"reverse key_string leaks target name into key: key={key_string!r}, name={name!r}, description={description!r}"
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
    for fact in stage1.get("facts", []) or []:
        for ev in fact.get("evidence", []) or []:
            title = norm_text(ev.get("title", ""))
            sid = ev.get("sentence_id")
            if isinstance(sid, int):
                covered.add((title, sid))

    missing = [x for x in required if x not in covered]
    return (len(missing) == 0), missing


# ============================================================
# Cache helpers
# ============================================================


def cache_path(cache_dir: Optional[str], sample_id: str, suffix: str) -> Optional[str]:
    if not cache_dir:
        return None
    return os.path.join(cache_dir, f"{sample_id}.{suffix}.json")


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
# Merge model output back into sample
# ============================================================


def merge_stage2_into_sample(sample: Dict[str, Any], stage2: Dict[str, Any]) -> Dict[str, Any]:
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
                    print(
                        f"[{now_ts()}] progress {done_cnt}/{stats['total']} "
                        f"ok={stats['ok']} err={stats['err']} retried={stats['retried']} "
                        f"dropped={stats['dropped']} last={sample_id} "
                        f"failures={dict(sorted(failure_stats.items()))}"
                    )
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
                        print(
                            f"[{now_ts()}] progress {done_cnt}/{stats['total']} "
                            f"ok={stats['ok']} err={stats['err']} retried={stats['retried']} "
                            f"dropped={stats['dropped']} last={item.sample_id} "
                            f"failures={dict(sorted(failure_stats.items()))}"
                        )
        finally:
            queue.task_done()


# ============================================================
# V4.2 overrides: three-stage workflow
# Stage 1: semantic fact extraction
# Stage 2: triple normalization + forward KV only
# Stage 3: reverse KV only
# ============================================================

STAGE2_SCHEMA: Dict[str, Any] = {
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

STAGE3_SCHEMA: Dict[str, Any] = {
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

STAGE2_SYSTEM_PROMPT = """You are a normalization engine for DAG-KV / KBLaM triple construction.

Your task is to convert the provided stage-1 semantic facts into dataset-compatible triples and generate ONLY the FORWARD KV for each triple.

Primary goals:
1. Preserve all valid facts from stage 1.
2. Convert each fact into exactly one triple.
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
- Do not drop facts.
- Do not merge different facts.
- Do not generate reverse KV in this stage.
"""

STAGE3_SYSTEM_PROMPT = """You are a reverse-KV generation engine for DAG-KV / KBLaM triple construction.

Your task is to generate ONLY the REVERSE KV for each provided triple.

You are given:
- triple_id
- type
- name
- description_type
- description
- the already generated forward KV

Your output must contain exactly one reverse KV per input triple.

============================================================
I. OUTPUT FORMAT
============================================================

For each page, output:
{
  "title": "<page title>",
  "reverse_list": [
    {
      "triple_id": "<copy exactly from input>",
      "key_string": "...",
      "value_string": "..."
    }
  ]
}

============================================================
II. REVERSE KV RULES
============================================================

For each triple:
- key_string MUST be a natural semantic query or statement fragment anchored on description
- value_string MUST equal name exactly

The reverse key_string MUST:
- preserve the object/value anchor (the description, or a faithful surface form of it)
- preserve the meaning of description_type
- point from the object/value toward the missing subject phrase
- be logically equivalent to the provided triple
- be concise, stable, and retrieval-friendly
- not invent information
- not contain the full subject phrase when avoidable

CRITICAL HARD CONSTRAINTS:
- Copy triple_id exactly.
- value_string MUST equal name exactly.
- Generate reverse KV only. Do not output forward KV.

BAD EXAMPLE:
name="Norway", description_type="population date", description="August 2018"
{"triple_id": "T00001", "key_string": "the population date of Norway is", "value_string": "August 2018"}
This is wrong because it is forward.

GOOD EXAMPLE:
{"triple_id": "T00001", "key_string": "the country whose population date is August 2018 is", "value_string": "Norway"}

============================================================
III. OUTPUT RULES
============================================================

- Return valid JSON only.
- Follow the supplied schema exactly.
- Do not include explanations.
- Do not include markdown.
- Do not drop items.
- Do not reorder or change triple_id.
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

    preferred = [kv for kv in cleaned if kv["value_string"] == description and has_anchor_in_key(kv["key_string"], name)]
    if preferred:
        return _repair_single_kv(tri, preferred[0])

    preferred = [kv for kv in cleaned if kv["value_string"] == description]
    if preferred:
        return _repair_single_kv(tri, preferred[0])

    return _repair_single_kv(tri, cleaned[0])


def normalize_stage2_forward(stage2: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    out = {"_id": norm_text(stage2.get("_id", "")), "context": []}
    triple_counter = 1

    for para in stage2.get("context", []) or []:
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
                print(f"[stage2] {tri2}")
            triples.append(tri2)

        out["context"].append(
            {
                "title": norm_text(para.get("title", "")),
                "sentences": [norm_text(x) for x in (para.get("sentences", []) or [])],
                "triple_list": triples,
            }
        )
    return out


def build_stage3_input(stage2: Dict[str, Any]) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    for para in stage2.get("context", []) or []:
        triples = []
        for tri in para.get("triple_list", []) or []:
            kvs = tri.get("kv_lists", []) or []
            forward_kv = kvs[0] if kvs else {"key_string": "", "value_string": ""}
            triples.append(
                {
                    "triple_id": norm_text(tri.get("triple_id", "")),
                    "type": norm_text(tri.get("type", "")),
                    "name": norm_text(tri.get("name", "")),
                    "description_type": norm_text(tri.get("description_type", "")),
                    "description": norm_text(tri.get("description", "")),
                    "forward_kv": {
                        "key_string": norm_text(forward_kv.get("key_string", "")),
                        "value_string": norm_text(forward_kv.get("value_string", "")),
                    },
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
        "_id": norm_text(stage2.get("_id", "")),
        "pages": pages,
    }


def build_empty_stage3_from_stage2(stage2: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": norm_text(stage2.get("_id", "")),
        "context": [
            {
                "title": norm_text(para.get("title", "")),
                "reverse_list": [],
            }
            for para in (stage2.get("context", []) or [])
        ],
    }


def normalize_stage3_reverse(stage3: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    out = {"_id": norm_text(stage3.get("_id", "")), "context": []}
    for para in stage3.get("context", []) or []:
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
                print(f"[stage3] {reverse_item}")
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
    if value_string != description:
        raise ValueError(
            f"forward kv value_string must equal description: expected={description!r} got={value_string!r}"
        )
    if not has_anchor_in_key(key_string, name):
        raise ValueError(
            f"forward key_string missing subject anchor: key={key_string!r}, name={name!r}, description={description!r}"
        )
    if normalize_anchor_text(name) != normalize_anchor_text(description) and has_full_anchor_leak(key_string, description):
        raise ValueError(
            f"forward key_string leaks target description into key: key={key_string!r}, name={name!r}, description={description!r}"
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
    if value_string != name:
        raise ValueError(
            f"reverse kv value_string must equal name: triple_id={triple_id} expected={name!r} got={value_string!r}"
        )
    if not has_anchor_in_key(key_string, description):
        raise ValueError(
            f"reverse key_string missing object/value anchor: key={key_string!r}, name={name!r}, description={description!r}"
        )
    if normalize_anchor_text(name) != normalize_anchor_text(description) and has_full_anchor_leak(key_string, name):
        raise ValueError(
            f"reverse key_string leaks target name into key: key={key_string!r}, name={name!r}, description={description!r}"
        )


def validate_stage2(stage2: Dict[str, Any], verbose: bool = False) -> None:
    if not isinstance(stage2, dict):
        raise ValueError("stage2 must be a dict")
    if not isinstance(stage2.get("_id"), str):
        raise ValueError("stage2._id missing or invalid")
    if not isinstance(stage2.get("context"), list):
        raise ValueError("stage2.context missing or invalid")

    seen_triple_ids = set()
    for para in stage2["context"]:
        if not isinstance(para, dict):
            raise ValueError("stage2 context item must be object")
        if not isinstance(para.get("title"), str):
            raise ValueError("stage2 context.title invalid")
        if not isinstance(para.get("sentences"), list):
            raise ValueError("stage2 context.sentences invalid")
        if not isinstance(para.get("triple_list"), list):
            raise ValueError("stage2 context.triple_list invalid")
        for tri in para["triple_list"]:
            triple_id = norm_text(tri.get("triple_id", ""))
            if not triple_id:
                raise ValueError("stage2 triple_id invalid")
            if triple_id in seen_triple_ids:
                raise ValueError(f"duplicate stage2 triple_id: {triple_id}")
            seen_triple_ids.add(triple_id)
            validate_forward_kv_only(tri)


def validate_stage3(stage2: Dict[str, Any], stage3: Dict[str, Any], verbose: bool = False) -> None:
    if not isinstance(stage3, dict):
        raise ValueError("stage3 must be a dict")
    if norm_text(stage3.get("_id", "")) != norm_text(stage2.get("_id", "")):
        raise ValueError("stage3._id does not match stage2._id")
    if not isinstance(stage3.get("context"), list):
        raise ValueError("stage3.context missing or invalid")

    stage2_by_title = {norm_text(p.get("title", "")): p for p in (stage2.get("context", []) or [])}
    stage3_by_title = {norm_text(p.get("title", "")): p for p in (stage3.get("context", []) or [])}

    if set(stage3_by_title.keys()) != set(stage2_by_title.keys()):
        raise ValueError(
            f"stage3 titles do not match stage2 titles: stage2={sorted(stage2_by_title)} stage3={sorted(stage3_by_title)}"
        )

    for title, para2 in stage2_by_title.items():
        para3 = stage3_by_title[title]
        reverse_items = para3.get("reverse_list")
        if not isinstance(reverse_items, list):
            raise ValueError(f"stage3 reverse_list invalid for title={title!r}")
        reverse_by_id = {}
        for item in reverse_items:
            if not isinstance(item, dict):
                raise ValueError("stage3 reverse item must be object")
            triple_id = norm_text(item.get("triple_id", ""))
            if not triple_id:
                raise ValueError("stage3 reverse triple_id invalid")
            if triple_id in reverse_by_id:
                raise ValueError(f"duplicate stage3 triple_id in title={title!r}: {triple_id}")
            reverse_by_id[triple_id] = item

        expected_ids = [norm_text(t.get("triple_id", "")) for t in (para2.get("triple_list", []) or [])]
        if set(reverse_by_id.keys()) != set(expected_ids):
            raise ValueError(
                f"stage3 reverse triple_id set mismatch for title={title!r}: expected={sorted(expected_ids)} got={sorted(reverse_by_id)}"
            )
        for tri in para2.get("triple_list", []) or []:
            validate_reverse_item_against_triple(reverse_by_id[tri["triple_id"]], tri)


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


def merge_stage2_and_stage3(stage2: Dict[str, Any], stage3: Dict[str, Any]) -> Dict[str, Any]:
    stage3_by_title = {
        norm_text(para.get("title", "")): {
            norm_text(item.get("triple_id", "")): item
            for item in (para.get("reverse_list", []) or [])
        }
        for para in (stage3.get("context", []) or [])
    }

    out = {"_id": norm_text(stage2.get("_id", "")), "context": []}
    for para in stage2.get("context", []) or []:
        title = norm_text(para.get("title", ""))
        reverse_by_id = stage3_by_title.get(title, {})
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


@dataclass
class WorkerConfig:
    include_question: bool
    include_answer: bool
    answer_aware: bool
    supporting_pages_only: bool
    stage1_cache_dir: Optional[str]
    stage2_cache_dir: Optional[str]
    stage3_cache_dir: Optional[str]
    verify_supporting_coverage: bool
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
    if verbose:
        print(f"[sample] {sample}")

    stage1_path = cache_path(cfg.stage1_cache_dir, sample_id, "stage1")
    stage2_path = cache_path(cfg.stage2_cache_dir, sample_id, "stage2")
    stage3_path = cache_path(cfg.stage3_cache_dir, sample_id, "stage3")

    # ---------------- stage 1 ----------------
    try:
        stage1 = None if cfg.overwrite else load_cached_json(stage1_path)
        if stage1 is None:
            stage1_input = build_stage1_input(
                sample,
                include_question=cfg.include_question,
                include_answer=cfg.include_answer,
                supporting_pages_only=cfg.supporting_pages_only,
            )
            stage1_system_prompt = get_stage1_system_prompt(
                answer_aware=cfg.answer_aware,
                include_answer=cfg.include_answer,
                sample=sample,
            )
            try:
                stage1 = await client.create_structured_response(
                    system_prompt=stage1_system_prompt,
                    user_payload=stage1_input,
                    schema_name="stage1_semantic_facts",
                    schema=STAGE1_SCHEMA,
                    request_id=make_request_id("stage1", sample_id),
                )
            except Exception as e:
                raise SampleProcessingError("stage1_api", sample_id, e) from e
            try:
                stage1 = dedupe_stage1(stage1)
                if verbose:
                    print(f"[stage1] {stage1}")
                validate_stage1(stage1)
                if cfg.verify_supporting_coverage:
                    ok, missing = check_supporting_sentence_coverage(sample, stage1)
                    if not ok:
                        raise ValueError(
                            f"Stage1 coverage check failed for {sample_id}, missing supporting sentences: {missing}"
                        )
            except Exception as e:
                raise SampleProcessingError("stage1_validation", sample_id, e) from e
            save_cached_json(stage1_path, stage1)
        else:
            try:
                stage1 = dedupe_stage1(stage1)
                validate_stage1(stage1)
            except Exception as e:
                raise SampleProcessingError("stage1_validation", sample_id, e) from e
    except SampleProcessingError:
        raise

    # ---------------- stage 2: forward only ----------------
    try:
        stage2 = None if cfg.overwrite else load_cached_json(stage2_path)
        if stage2 is None:
            stage2_input = build_stage2_input(sample, stage1)
            try:
                stage2 = await client.create_structured_response(
                    system_prompt=STAGE2_SYSTEM_PROMPT,
                    user_payload=stage2_input,
                    schema_name="stage2_forward_triple_list",
                    schema=STAGE2_SCHEMA,
                    request_id=make_request_id("stage2", sample_id),
                )
            except Exception as e:
                raise SampleProcessingError("stage2_api", sample_id, e) from e
            try:
                stage2 = normalize_stage2_forward(stage2, verbose=verbose)
                validate_stage2(stage2, verbose=verbose)
            except Exception as e:
                raise SampleProcessingError("stage2_validation", sample_id, e) from e
            save_cached_json(stage2_path, stage2)
        else:
            try:
                stage2 = normalize_stage2_forward(stage2, verbose=verbose)
                validate_stage2(stage2, verbose=verbose)
            except Exception as e:
                raise SampleProcessingError("stage2_validation", sample_id, e) from e
    except SampleProcessingError:
        raise

    # ---------------- stage 3: reverse only ----------------
    try:
        stage3 = None if cfg.overwrite else load_cached_json(stage3_path)
        if stage3 is None:
            if sum(len(p.get("triple_list", []) or []) for p in (stage2.get("context", []) or [])) == 0:
                stage3 = build_empty_stage3_from_stage2(stage2)
            else:
                stage3_input = build_stage3_input(stage2)
                try:
                    stage3 = await client.create_structured_response(
                        system_prompt=STAGE3_SYSTEM_PROMPT,
                        user_payload=stage3_input,
                        schema_name="stage3_reverse_kv_list",
                        schema=STAGE3_SCHEMA,
                        request_id=make_request_id("stage3", sample_id),
                    )
                except Exception as e:
                    raise SampleProcessingError("stage3_api", sample_id, e) from e
            try:
                stage3 = normalize_stage3_reverse(stage3, verbose=verbose)
                validate_stage3(stage2, stage3, verbose=verbose)
            except Exception as e:
                raise SampleProcessingError("stage3_validation", sample_id, e) from e
            save_cached_json(stage3_path, stage3)
        else:
            try:
                stage3 = normalize_stage3_reverse(stage3, verbose=verbose)
                validate_stage3(stage2, stage3, verbose=verbose)
            except Exception as e:
                raise SampleProcessingError("stage3_validation", sample_id, e) from e

        try:
            stage_final = merge_stage2_and_stage3(stage2, stage3)
            validate_final_stage(stage_final, verbose=verbose)
            final_sample = merge_stage2_into_sample(sample, stage_final)
            final_sample = normalize_final_sample_output(final_sample)
        except Exception as e:
            raise SampleProcessingError("finalization", sample_id, e) from e
    except SampleProcessingError:
        raise

    return sample_id, final_sample, {"stage1": stage1, "stage2": stage2, "stage3": stage3}


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and args.api_key_env != "OFOXAI_API_KEY":
        api_key = os.environ.get("OFOXAI_API_KEY", "").strip()
    if not api_key and not args.allow_empty_api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is empty."
            " Set it directly, or export OFOXAI_API_KEY for OfoxAI."
        )

    ensure_dir(args.stage1_cache_dir)
    ensure_dir(args.stage2_cache_dir)
    ensure_dir(args.stage3_cache_dir)
    ensure_dir(os.path.dirname(args.output) or ".")
    if args.error_log:
        ensure_dir(os.path.dirname(args.error_log) or ".")

    samples = read_json_or_jsonl(args.input)
    print(f"Load {len(samples)} samples from {args.input}")
    # 过滤掉无法answer的样本
    samples = [s for s in samples if s.get("answerable", True) is not False]
    print(f"Filter {len(samples)} samples after answerable check")


    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(samples)

    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    if args.skip_comparison:
        new_samples = []
        for s in samples:
            if s.get("type") == "comparison":
                continue
            new_samples.append(s)
        samples = new_samples

    print(f"Process loaded {len(samples)} samples from {args.input}")
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
        stage1_cache_dir=args.stage1_cache_dir,
        stage2_cache_dir=args.stage2_cache_dir,
        stage3_cache_dir=args.stage3_cache_dir,
        verify_supporting_coverage=not args.skip_supporting_coverage_check,
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

        print(
            f"[{now_ts()}] DONE total={total} ok={stats['ok']} err={stats['err']} "
            f"retried={stats['retried']} dropped={stats['dropped']} failures={dict(sorted(failure_stats.items()))}"
        )
    finally:
        await client.aclose()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Three-stage async KG extraction with OpenAI-compatible APIs")
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

    ap.add_argument("--stage1-cache-dir", type=str, default="./cache_stage1")
    ap.add_argument("--stage2-cache-dir", type=str, default="./cache_stage2")
    ap.add_argument("--stage3-cache-dir", type=str, default="./cache_stage3")
    ap.add_argument("--resume", action="store_true", help="Skip samples already present in output jsonl")
    ap.add_argument("--overwrite", action="store_true", help="Ignore caches and recompute")
    ap.add_argument("--error-log", type=str, default="./kg_extract_errors.log")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--sample-retries", type=int, default=2, help="Max times to requeue a failed sample after the initial attempt")

    ap.add_argument("--no-question", action="store_true", help="Do not include question in stage1 prompt")
    ap.add_argument("--no-answer", action="store_true", help="Do not include answer in stage1 prompt")
    ap.add_argument("--answer-aware", action="store_true", help="Use an answer-aware stage1 prompt to improve recall of answer-linked supporting facts")
    ap.add_argument("--use-all-context-pages", action="store_true", help="Use all context/pages end-to-end; default is supporting pages only")
    ap.add_argument("--skip-supporting-coverage-check", action="store_true", help="Skip local check that each supporting sentence is covered by stage1 evidence")
    ap.add_argument("--skip-comparison", action="store_true", help="Add SKTP comparison description type to stage2 style hints")
    ap.add_argument("--verbose", action="store_true", help="Print verbose output")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for shuffling samples")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
