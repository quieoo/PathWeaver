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
    sid = norm_text(sample.get("_id", ""))
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
                                "attribute_desc_alias": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                                "kv_lists": {
                                    "type": "array",
                                    "minItems": 2,
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
                                "attribute_desc_alias",
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


# ============================================================
# Prompts
# ============================================================

STAGE1_SYSTEM_PROMPT = """You are a knowledge graph fact extraction engine.

Your task is to extract a complete, faithful, evidence-grounded semantic fact layer from the provided QA sample.

Primary goal:
1. Cover all supporting facts completely.
2. Preserve all bridge facts needed to connect supporting facts into one knowledge graph.
3. Extract only facts explicitly stated in the provided text.
4. Do not hallucinate.

Important:
- Do NOT generate triple_list.
- Do NOT generate kv_lists.
- Do NOT classify into ATTRIBUTE or RELATION yet.
- Instead, extract a semantic fact layer with entities, facts, object_kind, and evidence.

Extraction rules:
- Focus first on supporting fact sentences.
- Also include other explicitly stated bridge facts from the same supporting pages if they help graph connectivity.
- Split multi-fact sentences into multiple facts.
- If one subject links to multiple objects, create multiple facts.
- Use short, stable predicate phrases.
- Keep entities canonical and retrieval-friendly.
- Add aliases when they are explicitly stated or directly obvious within the page wording (e.g. surname reference like 'Corelli' for 'Franco Corelli').
- Every fact must have evidence.
- evidence.title must be the page title.
- evidence.sentence_id must be the integer sentence index in that page.

object_kind rules:
- Use object_kind='entity' for person, organization, work, place, event, group, or other concrete graph node.
- Use object_kind='literal' for date, year, profession label, category, nickname, feature, quantity, short descriptive value, or other attribute-like value.

Deduplication rules:
- Do not output exact duplicate facts.
- If the same fact appears in multiple sentences, keep one fact and attach all evidence items.

Output rules:
- Return valid JSON only.
- Follow the supplied JSON schema exactly.
- Do not include explanations.
- Do not include markdown.
"""

STAGE2_SYSTEM_PROMPT = """You are a normalization engine for DAG-KV / KBLaM style triple_list construction.

Your task is to convert the provided stage-1 semantic facts into the final triple_list format used by the dataset.

Primary goal:
1. Preserve all facts from stage 1.
2. Convert each fact into exactly one triple in dataset-compatible format.
3. Classify each triple as ATTRIBUTE or RELATION.
4. Generate bidirectional kv_lists.
5. Keep the output 100% compatible with the target triple_list schema.

Important constraints:
- Do not drop any fact from stage 1.
- Do not add unsupported facts.
- Do not hallucinate.
- Keep wording concise, stable, and retrieval-friendly.
- Mimic an existing dataset style rather than inventing a new schema.

Schema rules for each triple:
{
  "type": "ATTRIBUTE" or "RELATION",
  "name": "<subject entity>",
  "description_type": "<relation or attribute name>",
  "description": "<object entity or attribute value>",
  "attribute_desc_alias": null or "<alias>",
  "kv_lists": [
    {"key_string": "...", "value_string": "..."},
    {"key_string": "...", "value_string": "..."}
  ]
}

ATTRIBUTE vs RELATION rules:
- Use ATTRIBUTE when the object is literal-like: date, year, category, profession label, nickname, feature, time span, short descriptive value, etc.
- Use RELATION when the object is entity-like: person, organization, work, place, event, group, or other graph node.
- Use stage1.object_kind as the primary signal, but keep compatibility with dataset style.

Naming rules:
- Prefer short, stable description_type values such as:
  birth date, death date, release year, production location, nickname, occupation, nationality,
  starring, directed by, based on, composed by, partnered with, located in, voice type, type.
- ATTRIBUTE triples should usually use attribute_desc_alias = null.
- RELATION triples may use a short alias when helpful, such as director, star, partnership.

kv_lists rules:
- ATTRIBUTE triples:
  1. "the <description_type> of <name>" -> "<description>"
  2. "<description> is the <description_type> of" -> "<name>"
- RELATION triples:
  generate natural bidirectional pairs.
  Two pairs are sufficient, but four pairs are allowed if they stay concise and faithful.
- All kv pairs must remain semantically faithful.

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
        return {
            "title": norm_text(entry.get("title", "")),
            "sentences": [norm_text(s) for s in (entry.get("sentences", []) or [])],
            "triple_list": entry.get("triple_list", []) or [],
        }

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        title = norm_text(entry[0])
        raw_sentences = entry[1] if isinstance(entry[1], list) else []
        triple_list = entry[2] if len(entry) >= 3 and isinstance(entry[2], list) else []
        return {
            "title": title,
            "sentences": [norm_text(s) for s in raw_sentences],
            "triple_list": triple_list,
        }

    return None


def get_context_map(sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_para in sample.get("context", []) or []:
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
    return titles


def build_stage1_input(
    sample: Dict[str, Any],
    *,
    include_question: bool,
    include_answer: bool,
    supporting_pages_only: bool,
    include_non_supporting_titles: bool,
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
        for raw_para in sample.get("context", []) or []:
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
        "_id": norm_text(sample.get("_id", "")),
        "supporting_facts": [
            [norm_text(x[0]), int(x[1])] for x in (sample.get("supporting_facts", []) or []) if isinstance(x, (list, tuple)) and len(x) >= 2
        ],
        "pages": pages,
    }
    if include_question:
        payload["question"] = norm_text(sample.get("question", ""))
    if include_answer:
        payload["answer"] = norm_text(sample.get("answer", ""))
    if include_non_supporting_titles:
        payload["other_context_titles"] = [
            para["title"]
            for para in (
                normalize_context_entry(raw_para)
                for raw_para in (sample.get("context", []) or [])
            )
            if para is not None and para["title"] not in set(supporting_titles)
        ]
    return payload


def build_stage2_input(
    sample: Dict[str, Any],
    stage1: Dict[str, Any],
    *,
    relation_alias_map: Dict[str, Optional[str]],
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
        "_id": norm_text(sample.get("_id", "")),
        "pages": pages,
        "style_hints": {
            "relation_alias_map": relation_alias_map,
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


def validate_stage2(stage2: Dict[str, Any]) -> None:
    if not isinstance(stage2, dict):
        raise ValueError("stage2 must be a dict")
    if not isinstance(stage2.get("_id"), str):
        raise ValueError("stage2._id missing or invalid")
    if not isinstance(stage2.get("context"), list):
        raise ValueError("stage2.context missing or invalid")

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
            validate_triple(tri)


def validate_triple(tri: Dict[str, Any]) -> None:
    if not isinstance(tri, dict):
        raise ValueError("triple must be object")
    if tri.get("type") not in {"ATTRIBUTE", "RELATION"}:
        raise ValueError("triple.type invalid")
    for k in ["name", "description_type", "description"]:
        if not isinstance(tri.get(k), str) or not tri[k].strip():
            raise ValueError(f"triple.{k} invalid")
    alias = tri.get("attribute_desc_alias")
    if alias is not None and not isinstance(alias, str):
        raise ValueError("triple.attribute_desc_alias invalid")
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


RELATION_ALIAS_MAP: Dict[str, Optional[str]] = {
    "directed by": "director",
    "starring": "star",
    "based on": "based on",
    "composed by": "composer",
    "partnered with": "partnership",
    "located in": "location",
    "appeared with": "appeared with",
    "appeared on stages of": "appeared on stages of",
}


def normalize_stage2(stage2: Dict[str, Any]) -> Dict[str, Any]:
    out = {"_id": norm_text(stage2.get("_id", "")), "context": []}
    for para in stage2.get("context", []) or []:
        triples = []
        seen = set()
        for tri in para.get("triple_list", []) or []:
            tri2 = {
                "type": norm_text(tri.get("type", "")),
                "name": norm_text(tri.get("name", "")),
                "description_type": norm_text(tri.get("description_type", "")),
                "description": norm_text(tri.get("description", "")),
                "attribute_desc_alias": None if tri.get("attribute_desc_alias") is None else norm_text(tri.get("attribute_desc_alias", "")) or None,
                "kv_lists": [],
            }
            for kv in tri.get("kv_lists", []) or []:
                key_string = norm_text(kv.get("key_string", ""))
                value_string = norm_text(kv.get("value_string", ""))
                if key_string and value_string:
                    tri2["kv_lists"].append({
                        "key_string": key_string,
                        "value_string": value_string,
                    })
            sig = (
                tri2["type"],
                tri2["name"],
                tri2["description_type"],
                tri2["description"],
            )
            if sig in seen:
                continue
            seen.add(sig)
            triples.append(tri2)
        out["context"].append(
            {
                "title": norm_text(para.get("title", "")),
                "sentences": [norm_text(x) for x in (para.get("sentences", []) or [])],
                "triple_list": triples,
            }
        )
    return out


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


def merge_stage2_into_sample(sample: Dict[str, Any], stage2: Dict[str, Any], keep_non_supporting_empty: bool) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    title_to_triples = {
        norm_text(para.get("title", "")): para.get("triple_list", []) or []
        for para in (stage2.get("context", []) or [])
    }

    for idx, para in enumerate(out.get("context", []) or []):
        norm_para = normalize_context_entry(para)
        if norm_para is None:
            continue

        title = norm_para["title"]
        if title in title_to_triples:
            triple_list = title_to_triples[title]
        elif keep_non_supporting_empty:
            triple_list = norm_para["triple_list"]
        else:
            continue

        if isinstance(para, dict):
            para["triple_list"] = triple_list
        elif isinstance(para, list):
            if len(para) >= 3:
                para[2] = triple_list
            else:
                out["context"][idx] = list(para) + [triple_list]
        elif isinstance(para, tuple):
            items = list(para[:2])
            items.append(triple_list)
            out["context"][idx] = items
    return out


# ============================================================
# Worker
# ============================================================


@dataclass
class WorkerConfig:
    include_question: bool
    include_answer: bool
    supporting_pages_only: bool
    include_non_supporting_titles: bool
    keep_non_supporting_empty: bool
    stage1_cache_dir: Optional[str]
    stage2_cache_dir: Optional[str]
    verify_supporting_coverage: bool
    overwrite: bool


async def process_one_sample(
    *,
    idx: int,
    sample: Dict[str, Any],
    client: OpenAIResponsesClient,
    cfg: WorkerConfig,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    sample_id = safe_sample_id(sample, idx)
    sample["_id"] = sample_id

    stage1_path = cache_path(cfg.stage1_cache_dir, sample_id, "stage1")
    stage2_path = cache_path(cfg.stage2_cache_dir, sample_id, "stage2")

    # ---------------- stage 1 ----------------
    stage1 = None if cfg.overwrite else load_cached_json(stage1_path)
    if stage1 is None:
        stage1_input = build_stage1_input(
            sample,
            include_question=cfg.include_question,
            include_answer=cfg.include_answer,
            supporting_pages_only=cfg.supporting_pages_only,
            include_non_supporting_titles=cfg.include_non_supporting_titles,
        )
        stage1 = await client.create_structured_response(
            system_prompt=STAGE1_SYSTEM_PROMPT,
            user_payload=stage1_input,
            schema_name="stage1_semantic_facts",
            schema=STAGE1_SCHEMA,
            request_id=make_request_id("stage1", sample_id),
        )
        stage1 = dedupe_stage1(stage1)
        validate_stage1(stage1)
        if cfg.verify_supporting_coverage:
            ok, missing = check_supporting_sentence_coverage(sample, stage1)
            if not ok:
                raise ValueError(
                    f"Stage1 coverage check failed for {sample_id}, missing supporting sentences: {missing}"
                )
        save_cached_json(stage1_path, stage1)
    else:
        stage1 = dedupe_stage1(stage1)
        validate_stage1(stage1)

    # ---------------- stage 2 ----------------
    stage2 = None if cfg.overwrite else load_cached_json(stage2_path)
    if stage2 is None:
        stage2_input = build_stage2_input(
            sample,
            stage1,
            relation_alias_map=RELATION_ALIAS_MAP,
        )
        stage2 = await client.create_structured_response(
            system_prompt=STAGE2_SYSTEM_PROMPT,
            user_payload=stage2_input,
            schema_name="stage2_triple_list",
            schema=STAGE2_SCHEMA,
            request_id=make_request_id("stage2", sample_id),
        )
        stage2 = normalize_stage2(stage2)
        validate_stage2(stage2)
        save_cached_json(stage2_path, stage2)
    else:
        stage2 = normalize_stage2(stage2)
        validate_stage2(stage2)

    final_sample = merge_stage2_into_sample(sample, stage2, cfg.keep_non_supporting_empty)
    return sample_id, final_sample, {"stage1": stage1, "stage2": stage2}


# ============================================================
# Runner
# ============================================================


async def bounded_process(
    *,
    sem: asyncio.Semaphore,
    idx: int,
    sample: Dict[str, Any],
    client: OpenAIResponsesClient,
    cfg: WorkerConfig,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    async with sem:
        return await process_one_sample(idx=idx, sample=sample, client=client, cfg=cfg)


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
    ensure_dir(os.path.dirname(args.output) or ".")
    if args.error_log:
        ensure_dir(os.path.dirname(args.error_log) or ".")

    samples = read_json_or_jsonl(args.input)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    if args.skip_comparison:
        new_samples = []
        for s in samples:
            if s.get("type")=="comparison":
                continue
            else:
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
        supporting_pages_only=not args.use_all_context_pages,
        include_non_supporting_titles=args.include_non_supporting_titles,
        keep_non_supporting_empty=not args.drop_non_supporting_context,
        stage1_cache_dir=args.stage1_cache_dir,
        stage2_cache_dir=args.stage2_cache_dir,
        verify_supporting_coverage=not args.skip_supporting_coverage_check,
        overwrite=args.overwrite,
    )

    client = OpenAIResponsesClient(cfg)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    try:
        tasks = []
        total = 0
        skipped = 0
        for idx, sample in enumerate(samples):
            sample_id = safe_sample_id(sample, idx)
            if sample_id in existing_ids:
                skipped += 1
                continue
            total += 1
            tasks.append(
                asyncio.create_task(
                    bounded_process(
                        sem=sem,
                        idx=idx,
                        sample=sample,
                        client=client,
                        cfg=worker_cfg,
                    )
                )
            )

        print(f"[{now_ts()}] Loaded {len(samples)} samples; pending={total}; skipped_existing={skipped}")

        done_cnt = 0
        ok_cnt = 0
        err_cnt = 0

        for coro in asyncio.as_completed(tasks):
            done_cnt += 1
            try:
                sample_id, final_sample, _meta = await coro
                append_jsonl(args.output, final_sample)
                ok_cnt += 1
                if done_cnt % max(1, args.progress_every) == 0 or done_cnt == total:
                    print(f"[{now_ts()}] progress {done_cnt}/{total} ok={ok_cnt} err={err_cnt} last={sample_id}")
            except Exception as e:
                err_cnt += 1
                tb = traceback.format_exc()
                msg = f"[{now_ts()}] ERROR {e}\n{tb}"
                print(msg, file=sys.stderr)
                if args.error_log:
                    with open(args.error_log, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")

        print(f"[{now_ts()}] DONE total={total} ok={ok_cnt} err={err_cnt} skipped={skipped}")
    finally:
        await client.aclose()


# ============================================================
# CLI
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Two-stage async KG extraction with OpenAI-compatible APIs")
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
    ap.add_argument("--resume", action="store_true", help="Skip samples already present in output jsonl")
    ap.add_argument("--overwrite", action="store_true", help="Ignore caches and recompute")
    ap.add_argument("--error-log", type=str, default="./kg_extract_errors.log")
    ap.add_argument("--progress-every", type=int, default=10)

    ap.add_argument("--no-question", action="store_true", help="Do not include question in stage1 prompt")
    ap.add_argument("--no-answer", action="store_true", help="Do not include answer in stage1 prompt")
    ap.add_argument("--use-all-context-pages", action="store_true", help="Use all context pages instead of supporting pages only")
    ap.add_argument("--include-non-supporting-titles", action="store_true", help="Include other context titles as weak hints in stage1 input")
    ap.add_argument("--drop-non-supporting-context", action="store_true", help="Do not preserve non-supporting pages in final output")
    ap.add_argument("--skip-supporting-coverage-check", action="store_true", help="Skip local check that each supporting sentence is covered by stage1 evidence")
    ap.add_argument("--skip-comparison", action="store_true", help="Add SKTP comparison description type to stage2 style hints")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
