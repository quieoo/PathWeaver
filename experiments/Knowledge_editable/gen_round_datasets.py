#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multi-round knowledge-edit datasets for DAG-KV editable evaluation.

The script builds a round-0 subset from an existing tripled dataset, then
iteratively edits a sampled subset of examples each round by calling an
OpenAI-compatible model. Each edited example returns:

- a minimally modified triple_list
- a new question / answer pair grounded in the edited triples
- metadata describing the edit

This is the "minimal viable simulation" version: we do not maintain a shared
global graph online. Instead, we replay multi-round sample-local graph edits
offline and rebuild DAGs later from each round's updated triple dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_KG_PATH = ROOT / "docs" / "scripts" / "triple_gen" / "build_knowledge_graph_v5.py"


def load_build_kg_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("build_knowledge_graph_v5", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kg = load_build_kg_module(DEFAULT_BUILD_KG_PATH)
OpenAIConfig = kg.OpenAIConfig
OpenAIResponsesClient = kg.OpenAIResponsesClient
ResponsesAPIError = kg.ResponsesAPIError
norm_text = kg.norm_text
safe_sample_id = kg.safe_sample_id
read_json_or_jsonl = kg.read_json_or_jsonl
append_jsonl = kg.append_jsonl
write_json = kg.write_json
programmatic_forward_kv = kg.programmatic_forward_kv
programmatic_reverse_kv = kg.programmatic_reverse_kv


EDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "edit_type": {
            "type": "string",
            "enum": [
                "attribute_update",
                "attribute_insert",
                "relation_update",
                "relation_insert",
            ],
        },
        "target_entity": {"type": "string"},
        "target_relation": {"type": "string"},
        "new_question": {"type": "string"},
        "new_answer": {"type": "string"},
        "edit_summary": {"type": "string"},
        "changed_triple": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["ATTRIBUTE", "RELATION"]},
                "name": {"type": "string"},
                "description_type": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["type", "name", "description_type", "description"],
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
                },
                "required": ["type", "name", "description_type", "description"],
            },
        },
    },
    "required": [
        "edit_type",
        "target_entity",
        "target_relation",
        "new_question",
        "new_answer",
        "edit_summary",
        "changed_triple",
        "triple_list",
    ],
}


EDIT_SYSTEM_PROMPT = """You are editing a QA-oriented local knowledge graph sample for a DAG-KV editable-memory benchmark.

You will receive:
- one existing QA sample
- its current triple_list
- a requested edit type
- a suggested target triple or anchor entity

Your job is to create ONE minimally edited successor sample.

Hard requirements:
1. Preserve the original topic and keep most triples unchanged.
2. Apply exactly one primary knowledge edit of the requested type:
   - attribute_update: replace the value of one existing ATTRIBUTE triple
   - attribute_insert: add one new ATTRIBUTE triple
   - relation_update: replace the tail entity of one existing RELATION triple
   - relation_insert: add one new RELATION triple
3. Generate a NEW question and NEW answer that depend on the edited knowledge.
4. The new answer must be different from the old answer string.
5. Return the FULL edited triple_list after modification, not only the diff.
6. Keep triples self-consistent. Avoid introducing many extra facts.
7. Use concise, dataset-style triples:
   - type in {ATTRIBUTE, RELATION}
   - name = head entity
   - description_type = relation or attribute label
   - description = tail entity or attribute value
8. Do not output kv_lists. They will be generated programmatically later.
9. Do not mention uncertainty, fictionality, or that this is an edit.
10. Produce a natural single-hop or multi-hop QA item that is answerable from the edited triple_list.

Quality preference:
- For updates, prefer editing an answer-relevant triple if one is suggested.
- For inserts, keep the new triple strongly connected to the existing entities.
- The new question should be specific and should resolve to the new answer, not the old one.

Return valid JSON only that matches the schema exactly.
"""


REPAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "edit_type": {
            "type": "string",
            "enum": [
                "attribute_update",
                "attribute_insert",
                "relation_update",
                "relation_insert",
            ],
        },
        "target_entity": {"type": "string"},
        "target_relation": {"type": "string"},
        "new_question": {"type": "string"},
        "new_answer": {"type": "string"},
        "edit_summary": {"type": "string"},
        "answer_sufficient": {"type": "boolean"},
        "missing_links": {
            "type": "array",
            "items": {"type": "string"},
        },
        "revision_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "changed_triple": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["ATTRIBUTE", "RELATION"]},
                "name": {"type": "string"},
                "description_type": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["type", "name", "description_type", "description"],
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
                },
                "required": ["type", "name", "description_type", "description"],
            },
        },
    },
    "required": [
        "edit_type",
        "target_entity",
        "target_relation",
        "new_question",
        "new_answer",
        "edit_summary",
        "answer_sufficient",
        "missing_links",
        "revision_notes",
        "changed_triple",
        "triple_list",
    ],
}


REPAIR_SYSTEM_PROMPT = """You are a post-edit QA graph review and repair system for a DAG-KV editable-memory benchmark.

You will receive:
- the original sample before editing
- a candidate edited sample produced by a previous generation step
- the requested edit type and target hint
- lightweight topic context from the original sample

Your job is to minimally repair the candidate edited sample so that it is structurally cleaner and more answerable from its own triple_list.

Primary goals:
1. Preserve the intended topic and preserve most triples unchanged.
2. Preserve exactly one primary edit of the requested type.
3. Make the edited triple_list internally self-consistent.
4. Make sure the new question is answerable from the edited triple_list.
5. Make sure the new answer is supported by the edited triple_list and still differs from the original pre-edit answer.
6. Keep the graph compact and avoid unrelated extra facts.
7. Prefer minimal repair over rewriting everything.

What to check and repair:
- whether the new question can be answered from the edited triple_list
- whether the changed_triple is actually reflected in triple_list
- whether relation / attribute wording is concise and natural
- whether the graph remains connected around the edited entity or answer
- whether extra noisy or duplicate triples can be removed without losing the intended edit

Output rules:
- Return the FULL repaired triple_list, not just a diff.
- Keep type in {ATTRIBUTE, RELATION}.
- Use dataset-style triples only.
- Do not output kv_lists.
- answer_sufficient should be true if the repaired triple_list explicitly supports the new answer for the new question.
- If still insufficient, set answer_sufficient to false and summarize the remaining issues in missing_links.
- revision_notes should briefly describe what you repaired.

Return valid JSON only that matches the schema exactly.
"""


@dataclass
class EditRequest:
    round_id: int
    sample_index: int
    edit_type: str
    target_hint: Dict[str, Any]
    request_seed: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_triple(triple: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    triple_type = norm_text(triple.get("type") or triple.get("triple_type")).upper()
    if triple_type not in {"ATTRIBUTE", "RELATION"}:
        return None
    name = norm_text(triple.get("name") or triple.get("head"))
    relation = norm_text(triple.get("description_type") or triple.get("relation"))
    description = norm_text(triple.get("description") or triple.get("tail"))
    if not name or not relation or not description:
        return None
    clean = {
        "type": triple_type,
        "name": name,
        "description_type": relation,
        "description": description,
    }
    clean["kv_lists"] = [
        programmatic_forward_kv(clean),
        programmatic_reverse_kv(clean),
    ]
    return clean


def normalize_triple_list(triples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for triple in triples:
        clean = normalize_triple(triple)
        if clean is None:
            continue
        key = (
            clean["type"],
            clean["name"].casefold(),
            clean["description_type"].casefold(),
            clean["description"].casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def strip_stale_graph_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    for key in ("dag", "answer_sufficient", "missing_links", "revision_notes"):
        out.pop(key, None)
    return out


def slim_sample_for_prompt(sample: Dict[str, Any]) -> Dict[str, Any]:
    triples = []
    for idx, tri in enumerate(sample.get("triple_list", [])):
        triples.append(
            {
                "idx": idx,
                "type": tri.get("type"),
                "name": tri.get("name"),
                "description_type": tri.get("description_type"),
                "description": tri.get("description"),
            }
        )
    return {
        "_id": sample.get("_id", ""),
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "type": sample.get("type", ""),
        "supporting_facts": sample.get("supporting_facts", []),
        "triples": triples,
    }


def slim_context_for_repair(sample: Dict[str, Any], *, max_pages: int = 4, max_sentences: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for page in sample.get("context", [])[:max_pages]:
        if not isinstance(page, dict):
            continue
        title = norm_text(page.get("title"))
        sentences = page.get("sentences", []) or []
        clean_sentences = [norm_text(x) for x in sentences[:max_sentences] if norm_text(x)]
        if title or clean_sentences:
            out.append({"title": title, "sentences": clean_sentences})
    return out


def find_answer_relevant_triples(sample: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    answer = norm_text(sample.get("answer"))
    if not answer:
        return []
    out: List[Tuple[int, Dict[str, Any]]] = []
    for idx, tri in enumerate(sample.get("triple_list", [])):
        hay = " ".join(
            [
                norm_text(tri.get("name")),
                norm_text(tri.get("description_type")),
                norm_text(tri.get("description")),
            ]
        )
        if answer.casefold() in hay.casefold():
            out.append((idx, tri))
    return out


def choose_or_first(items: Sequence[Any], rng: Optional[random.Random]) -> Any:
    if not items:
        return None
    if rng is None:
        return items[0]
    return rng.choice(list(items))


def choose_target_hint(sample: Dict[str, Any], edit_type: str, rng: Optional[random.Random]) -> Dict[str, Any]:
    triples = sample.get("triple_list", [])
    by_type = {
        "ATTRIBUTE": [(i, t) for i, t in enumerate(triples) if norm_text(t.get("type")).upper() == "ATTRIBUTE"],
        "RELATION": [(i, t) for i, t in enumerate(triples) if norm_text(t.get("type")).upper() == "RELATION"],
    }
    answer_related = find_answer_relevant_triples(sample)

    if edit_type == "attribute_update":
        candidates = [x for x in answer_related if norm_text(x[1].get("type")).upper() == "ATTRIBUTE"] or by_type["ATTRIBUTE"]
        target_idx, target = choose_or_first(candidates, rng) if candidates else (None, None)
        return {
            "target_triple_idx": target_idx,
            "target_triple": target,
            "anchor_entity": norm_text((target or {}).get("name")),
        }

    if edit_type == "relation_update":
        candidates = [x for x in answer_related if norm_text(x[1].get("type")).upper() == "RELATION"] or by_type["RELATION"]
        target_idx, target = choose_or_first(candidates, rng) if candidates else (None, None)
        return {
            "target_triple_idx": target_idx,
            "target_triple": target,
            "anchor_entity": norm_text((target or {}).get("name")),
        }

    preferred = answer_related or by_type["RELATION"] or by_type["ATTRIBUTE"]
    target_idx, target = choose_or_first(preferred, rng) if preferred else (None, None)
    return {
        "target_triple_idx": target_idx,
        "target_triple": target,
        "anchor_entity": norm_text((target or {}).get("name")),
    }


def build_round0_dataset(
    source_rows: Sequence[Dict[str, Any]],
    *,
    initial_size: int,
    rng: Optional[random.Random],
) -> List[Dict[str, Any]]:
    eligible_rows = [
        row for row in source_rows
        if normalize_triple_list(row.get("triple_list", []))
    ]
    if len(eligible_rows) < initial_size:
        raise ValueError(
            f"Eligible rows with non-empty triple_list = {len(eligible_rows)}, "
            f"smaller than initial_size={initial_size}"
        )
    if rng is None:
        chosen = list(eligible_rows)[:initial_size]
    else:
        chosen = rng.sample(list(eligible_rows), initial_size)
    round0: List[Dict[str, Any]] = []
    for idx, row in enumerate(chosen):
        clean = strip_stale_graph_fields(row)
        clean["triple_list"] = normalize_triple_list(clean.get("triple_list", []))
        clean["edit_meta"] = {
            "round": 0,
            "edited_in_round": False,
            "edit_type": "none",
            "source_sample_id": safe_sample_id(clean, idx),
            "old_answer": clean.get("answer", ""),
            "new_answer": clean.get("answer", ""),
            "target_triple_idx": None,
            "edit_summary": "initial sample",
        }
        round0.append(clean)
    return round0


def validate_edited_payload(
    sample: Dict[str, Any],
    request: EditRequest,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    triples = normalize_triple_list(payload.get("triple_list", []))
    if not triples:
        raise ValueError("Model returned empty triple_list after normalization")

    new_question = norm_text(payload.get("new_question"))
    new_answer = norm_text(payload.get("new_answer"))
    edit_summary = norm_text(payload.get("edit_summary"))
    target_entity = norm_text(payload.get("target_entity"))
    target_relation = norm_text(payload.get("target_relation"))
    if not new_question or not new_answer:
        raise ValueError("Missing new_question or new_answer")
    if new_answer.casefold() == norm_text(sample.get("answer")).casefold():
        raise ValueError("new_answer is identical to old answer")

    changed_triple = normalize_triple(payload.get("changed_triple", {}))
    if changed_triple is None:
        raise ValueError("Invalid changed_triple")

    updated = strip_stale_graph_fields(sample)
    updated["question"] = new_question
    updated["answer"] = new_answer
    updated["triple_list"] = triples
    updated["edit_meta"] = {
        "round": request.round_id,
        "edited_in_round": True,
        "edit_type": request.edit_type,
        "source_sample_id": safe_sample_id(sample, request.sample_index),
        "old_answer": sample.get("answer", ""),
        "new_answer": new_answer,
        "target_triple_idx": request.target_hint.get("target_triple_idx"),
        "target_entity": target_entity,
        "target_relation": target_relation,
        "edit_summary": edit_summary,
        "changed_triple": changed_triple,
        "request_seed": request.request_seed,
    }
    return updated


def validate_repair_payload(
    sample: Dict[str, Any],
    request: EditRequest,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    updated = validate_edited_payload(sample, request, payload)
    missing_links = [norm_text(x) for x in (payload.get("missing_links", []) or []) if norm_text(x)]
    revision_notes = [norm_text(x) for x in (payload.get("revision_notes", []) or []) if norm_text(x)]
    updated.setdefault("edit_meta", {})
    updated["edit_meta"]["repair_pipeline"] = {
        "applied": True,
        "answer_sufficient": bool(payload.get("answer_sufficient", False)),
        "missing_links": missing_links,
        "revision_notes": revision_notes,
    }
    return updated


def compute_repair_signals(original_sample: Dict[str, Any], edited_sample: Dict[str, Any], request: EditRequest) -> Dict[str, Any]:
    new_answer = norm_text(edited_sample.get("answer"))
    answer_in_triples = False
    for tri in edited_sample.get("triple_list", []):
        hay = " ".join(
            [
                norm_text(tri.get("name")),
                norm_text(tri.get("description_type")),
                norm_text(tri.get("description")),
            ]
        )
        if new_answer and new_answer.casefold() in hay.casefold():
            answer_in_triples = True
            break
    target_idx = request.target_hint.get("target_triple_idx")
    target_present = isinstance(target_idx, int) and 0 <= target_idx < len(edited_sample.get("triple_list", []))
    return {
        "new_answer_in_triples": answer_in_triples,
        "target_hint_index_present": target_present,
        "triple_count_before": len(original_sample.get("triple_list", [])),
        "triple_count_after": len(edited_sample.get("triple_list", [])),
    }


def _fallback_question_for_triple(triple: Dict[str, Any]) -> str:
    relation = norm_text(triple.get("description_type"))
    name = norm_text(triple.get("name"))
    if norm_text(triple.get("type")).upper() == "ATTRIBUTE":
        return f"What is the {relation} of {name}?"
    return f"What is the {relation} of {name}?"


def _fallback_new_tail(old_tail: str, *, round_id: int, sample_index: int, request_seed: int) -> str:
    base = norm_text(old_tail) or "unknown"
    if any(ch.isdigit() for ch in base):
        return f"{base} updated r{round_id} s{request_seed % 1000}"
    return f"{base} revised r{round_id} s{request_seed % 1000}"


def build_programmatic_fallback_edit(
    sample: Dict[str, Any],
    request: EditRequest,
) -> Dict[str, Any]:
    triples = [copy.deepcopy(t) for t in normalize_triple_list(sample.get("triple_list", []))]
    if not triples:
        raise ValueError("Cannot build fallback edit from empty triple_list")

    target_idx = request.target_hint.get("target_triple_idx")
    target = None
    if isinstance(target_idx, int) and 0 <= target_idx < len(triples):
        target = triples[target_idx]
    if target is None:
        rng = random.Random(request.request_seed + request.round_id * 131)
        target = rng.choice(triples)
        target_idx = triples.index(target)

    anchor_entity = norm_text(request.target_hint.get("anchor_entity")) or norm_text(target.get("name"))

    if request.edit_type == "attribute_insert":
        changed = normalize_triple(
            {
                "type": "ATTRIBUTE",
                "name": anchor_entity or norm_text(target.get("name")) or "Unknown Entity",
                "description_type": f"editable attribute round {request.round_id}",
                "description": f"value r{request.round_id} s{request.request_seed % 1000}",
            }
        )
        assert changed is not None
        triples.append(changed)
    elif request.edit_type == "relation_insert":
        changed = normalize_triple(
            {
                "type": "RELATION",
                "name": anchor_entity or norm_text(target.get("name")) or "Unknown Entity",
                "description_type": f"linked to round {request.round_id}",
                "description": f"Entity R{request.round_id} S{request.request_seed % 1000}",
            }
        )
        assert changed is not None
        triples.append(changed)
    else:
        changed = copy.deepcopy(target)
        changed["description"] = _fallback_new_tail(
            norm_text(target.get("description")),
            round_id=request.round_id,
            sample_index=request.sample_index,
            request_seed=request.request_seed,
        )
        changed = normalize_triple(changed)
        assert changed is not None
        triples[target_idx] = changed

    triples = normalize_triple_list(triples)
    payload = {
        "edit_type": request.edit_type,
        "target_entity": norm_text(changed.get("name")),
        "target_relation": norm_text(changed.get("description_type")),
        "new_question": _fallback_question_for_triple(changed),
        "new_answer": norm_text(changed.get("description")),
        "edit_summary": f"Programmatic fallback edit for {request.edit_type}.",
        "changed_triple": {
            "type": changed["type"],
            "name": changed["name"],
            "description_type": changed["description_type"],
            "description": changed["description"],
        },
        "triple_list": [
            {
                "type": tri["type"],
                "name": tri["name"],
                "description_type": tri["description_type"],
                "description": tri["description"],
            }
            for tri in triples
        ],
    }
    return validate_edited_payload(sample, request, payload)


def make_cache_file_name(sample_id: str, request: EditRequest) -> str:
    target_idx = request.target_hint.get("target_triple_idx")
    if target_idx is None:
        target_idx = "na"
    safe_edit = request.edit_type.replace("/", "_")
    return f"{sample_id}__{safe_edit}__t{target_idx}__s{request.request_seed}.json"


def make_repair_cache_file_name(sample_id: str, request: EditRequest) -> str:
    safe_edit = request.edit_type.replace("/", "_")
    return f"{sample_id}__{safe_edit}__repair__s{request.request_seed}.json"


async def edit_one_sample(
    *,
    client: Any,
    sample: Dict[str, Any],
    request: EditRequest,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    sample_id = safe_sample_id(sample, request.sample_index)
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        ensure_dir(cache_dir)
        cache_path = cache_dir / make_cache_file_name(sample_id, request)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            try:
                return validate_edited_payload(sample, request, payload)
            except Exception:
                # Ignore stale/bad cache and re-query.
                try:
                    bad_path = cache_path.with_suffix(".invalid.json")
                    cache_path.rename(bad_path)
                except Exception:
                    pass

    user_payload = {
        "round_id": request.round_id,
        "request_seed": request.request_seed,
        "requested_edit_type": request.edit_type,
        "target_hint": request.target_hint,
        "sample": slim_sample_for_prompt(sample),
        "editing_goal": {
            "keep_most_triples_unchanged": True,
            "generate_full_edited_triple_list": True,
            "new_answer_must_differ_from_old_answer": True,
        },
    }
    raw_payload = await client.create_structured_response(
        system_prompt=EDIT_SYSTEM_PROMPT,
        user_payload=user_payload,
        schema_name="knowledge_edit_round_sample",
        schema=EDIT_SCHEMA,
        request_id=f"knowledge-edit-r{request.round_id}-{sample_id}-{request.request_seed}",
    )
    if cache_path is not None:
        write_json(str(cache_path), raw_payload)
    return validate_edited_payload(sample, request, raw_payload)


async def repair_one_sample(
    *,
    client: Any,
    original_sample: Dict[str, Any],
    edited_sample: Dict[str, Any],
    request: EditRequest,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    sample_id = safe_sample_id(original_sample, request.sample_index)
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        ensure_dir(cache_dir)
        cache_path = cache_dir / make_repair_cache_file_name(sample_id, request)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            try:
                return validate_repair_payload(original_sample, request, payload)
            except Exception:
                try:
                    bad_path = cache_path.with_suffix(".invalid.json")
                    cache_path.rename(bad_path)
                except Exception:
                    pass

    user_payload = {
        "round_id": request.round_id,
        "request_seed": request.request_seed,
        "requested_edit_type": request.edit_type,
        "target_hint": request.target_hint,
        "quality_signals": compute_repair_signals(original_sample, edited_sample, request),
        "topic_context": slim_context_for_repair(original_sample),
        "original_sample": slim_sample_for_prompt(original_sample),
        "candidate_edited_sample": slim_sample_for_prompt(edited_sample),
        "repair_goal": {
            "preserve_primary_edit": True,
            "keep_most_triples_unchanged": True,
            "return_full_repaired_triple_list": True,
        },
    }
    raw_payload = await client.create_structured_response(
        system_prompt=REPAIR_SYSTEM_PROMPT,
        user_payload=user_payload,
        schema_name="knowledge_edit_round_repair",
        schema=REPAIR_SCHEMA,
        request_id=f"knowledge-edit-repair-r{request.round_id}-{sample_id}-{request.request_seed}",
    )
    if cache_path is not None:
        write_json(str(cache_path), raw_payload)
    return validate_repair_payload(original_sample, request, raw_payload)


async def run_round(
    *,
    client: Any,
    current_rows: Sequence[Dict[str, Any]],
    round_id: int,
    edits_per_round: int,
    seed: Optional[int],
    cache_dir: Optional[Path],
    concurrency: int,
    sample_retries: int,
    enable_repair_pipeline: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed + round_id * 10007) if seed is not None else random.Random()
    next_rows = [copy.deepcopy(row) for row in current_rows]
    chosen_indices = rng.sample(range(len(next_rows)), edits_per_round)

    requests: List[Tuple[int, EditRequest]] = []
    edit_types = [
        "attribute_update",
        "attribute_insert",
        "relation_update",
        "relation_insert",
    ]
    for idx in chosen_indices:
        sample = next_rows[idx]
        edit_type = rng.choice(edit_types)
        target_hint = choose_target_hint(sample, edit_type, rng)
        requests.append(
            (
                idx,
                EditRequest(
                    round_id=round_id,
                    sample_index=idx,
                    edit_type=edit_type,
                    target_hint=target_hint,
                    request_seed=rng.randint(0, 10**9),
                ),
            )
        )

    sem = asyncio.Semaphore(max(1, concurrency))
    edit_logs: List[Dict[str, Any]] = []

    async def _worker(sample_idx: int, request: EditRequest) -> None:
        sample = next_rows[sample_idx]
        last_err: Optional[Exception] = None
        updated: Optional[Dict[str, Any]] = None
        for retry_idx in range(sample_retries + 1):
            retry_request = request
            if retry_idx > 0:
                retry_rng = random.Random(request.request_seed + round_id * 9973 + retry_idx * 7919)
                retry_edit_type = retry_rng.choice(edit_types)
                retry_request = EditRequest(
                    round_id=request.round_id,
                    sample_index=request.sample_index,
                    edit_type=retry_edit_type,
                    target_hint=choose_target_hint(sample, retry_edit_type, retry_rng),
                    request_seed=request.request_seed + retry_idx,
                )
            try:
                async with sem:
                    updated = await edit_one_sample(
                        client=client,
                        sample=sample,
                        request=retry_request,
                        cache_dir=(cache_dir / f"round_{round_id}") if cache_dir is not None else None,
                    )
                    if enable_repair_pipeline:
                        updated = await repair_one_sample(
                            client=client,
                            original_sample=sample,
                            edited_sample=updated,
                            request=retry_request,
                            cache_dir=(cache_dir / f"round_{round_id}_repair") if cache_dir is not None else None,
                        )
                request = retry_request
                break
            except Exception as exc:
                last_err = exc
        if updated is None:
            updated = build_programmatic_fallback_edit(sample, request)
            updated.setdefault("edit_meta", {})
            updated["edit_meta"]["fallback_used"] = True
            updated["edit_meta"]["fallback_reason"] = str(last_err) if last_err is not None else "unknown"
        elif enable_repair_pipeline and "repair_pipeline" not in updated.get("edit_meta", {}):
            updated.setdefault("edit_meta", {})
            updated["edit_meta"]["repair_pipeline"] = {
                "applied": False,
                "answer_sufficient": False,
                "missing_links": [],
                "revision_notes": [],
            }
        next_rows[sample_idx] = updated
        edit_logs.append(
            {
                "round": round_id,
                "sample_index": sample_idx,
                "sample_id": safe_sample_id(sample, sample_idx),
                "edit_type": request.edit_type,
                "old_question": sample.get("question", ""),
                "new_question": updated.get("question", ""),
                "old_answer": sample.get("answer", ""),
                "new_answer": updated.get("answer", ""),
                "target_hint": request.target_hint,
                "edit_meta": updated.get("edit_meta", {}),
            }
        )

    tasks = [_worker(idx, req) for idx, req in requests]
    await asyncio.gather(*tasks)

    for idx, row in enumerate(next_rows):
        meta = row.get("edit_meta", {})
        if int(meta.get("round", -1)) != round_id:
            row["edit_meta"] = {
                "round": round_id,
                "edited_in_round": False,
                "edit_type": "carry_over",
                "source_sample_id": safe_sample_id(row, idx),
                "old_answer": row.get("answer", ""),
                "new_answer": row.get("answer", ""),
                "target_triple_idx": None,
                "edit_summary": f"carried over from round {round_id - 1}",
            }

    edit_logs.sort(key=lambda x: x["sample_index"])
    return next_rows, edit_logs


def build_client(args) -> Any:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and args.api_key_env != "OFOXAI_API_KEY":
        api_key = os.environ.get("OFOXAI_API_KEY", "").strip()
    if not api_key and not args.allow_empty_api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is empty. "
            "Set it directly, or use --allow-empty-api-key for local servers."
        )

    cfg = OpenAIConfig(
        api_key=api_key,
        api_base=args.api_base,
        model=args.model,
        timeout=args.timeout,
        max_connections=args.max_connections or args.concurrency,
        max_keepalive_connections=args.max_keepalive_connections or args.concurrency,
        max_retries=args.max_retries,
        retry_base=args.retry_base,
        retry_max=args.retry_max,
        store=not args.no_store,
        verbosity=args.verbosity,
        api_mode=args.api_mode,
    )
    return OpenAIResponsesClient(cfg)


def build_parser() -> argparse.ArgumentParser:
    def optional_int(value: str) -> Optional[int]:
        if value.strip().lower() == "none":
            return None
        return int(value)

    ap = argparse.ArgumentParser(description="Generate multi-round editable datasets with an OpenAI-compatible LLM")
    ap.add_argument("--input", type=str, required=True, help="Input tripled dataset (.json or .jsonl)")
    ap.add_argument("--output-dir", type=str, required=True, help="Directory to store round jsonl files")
    ap.add_argument("--initial-size", type=int, default=100, help="Number of samples in round0")
    ap.add_argument("--num-rounds", type=int, default=5, help="How many update rounds to generate after round0")
    ap.add_argument("--edits-per-round", type=int, default=20, help="How many samples to edit per round")
    ap.add_argument(
        "--seed",
        type=optional_int,
        default=None,
        help="Random seed for reproducible sampling; use None to keep round0 as ordered truncation",
    )
    ap.add_argument("--resume", action="store_true", help="Reuse existing round files if present")
    ap.add_argument("--cache-dir", type=str, default=None, help="Optional cache directory for raw model outputs")
    ap.add_argument("--sample-retries", type=int, default=2, help="Extra retries for one sample if validation fails")
    ap.add_argument("--enable-repair-pipeline", action="store_true", help="Run a post-edit QA graph review / repair pass after generation")

    ap.add_argument("--model", type=str, default="Qwen3.5-27B", help="OpenAI-compatible served model name")
    ap.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", os.environ.get("OFOXAI_BASE_URL", "http://localhost:8000/v1")),
        help="OpenAI-compatible API base URL",
    )
    ap.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--api-mode", type=str, default="chat", choices=["responses", "chat"])
    ap.add_argument("--allow-empty-api-key", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-connections", type=int, default=None)
    ap.add_argument("--max-keepalive-connections", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--retry-base", type=float, default=2.0)
    ap.add_argument("--retry-max", type=float, default=30.0)
    ap.add_argument("--verbosity", type=str, default="low", choices=["low", "medium", "high"])
    ap.add_argument("--no-store", action="store_true")
    return ap


async def async_main(args) -> None:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        ensure_dir(cache_dir)

    manifest_path = output_dir / "manifest.json"
    round0_path = output_dir / "round0.jsonl"
    source_rows = read_json_or_jsonl(args.input)
    if args.edits_per_round > args.initial_size:
        raise ValueError("--edits-per-round cannot exceed --initial-size")

    rng = random.Random(args.seed) if args.seed is not None else None
    if args.resume and round0_path.exists():
        current_rows = read_jsonl(round0_path)
    else:
        current_rows = build_round0_dataset(
            source_rows,
            initial_size=args.initial_size,
            rng=rng,
        )
        write_jsonl(round0_path, current_rows)

    client = build_client(args)
    started = time.time()
    manifest: Dict[str, Any] = {
        "input": args.input,
        "output_dir": str(output_dir),
        "initial_size": args.initial_size,
        "num_rounds": args.num_rounds,
        "edits_per_round": args.edits_per_round,
        "seed": args.seed,
        "model": args.model,
        "api_base": args.api_base,
        "api_mode": args.api_mode,
        "enable_repair_pipeline": args.enable_repair_pipeline,
        "round_files": {"round0": str(round0_path)},
        "started_at": started,
    }

    try:
        for round_id in range(1, args.num_rounds + 1):
            round_path = output_dir / f"round{round_id}.jsonl"
            edit_log_path = output_dir / f"round{round_id}_edits.json"

            if args.resume and round_path.exists() and edit_log_path.exists():
                current_rows = read_jsonl(round_path)
                manifest["round_files"][f"round{round_id}"] = str(round_path)
                continue

            print(f"[round {round_id}] editing {args.edits_per_round} / {len(current_rows)} samples ...", flush=True)
            current_rows, edit_logs = await run_round(
                client=client,
                current_rows=current_rows,
                round_id=round_id,
                edits_per_round=args.edits_per_round,
                seed=args.seed,
                cache_dir=cache_dir,
                concurrency=args.concurrency,
                sample_retries=args.sample_retries,
                enable_repair_pipeline=args.enable_repair_pipeline,
            )
            write_jsonl(round_path, current_rows)
            write_json(str(edit_log_path), edit_logs)
            manifest["round_files"][f"round{round_id}"] = str(round_path)
            print(f"[round {round_id}] done -> {round_path}", flush=True)
    finally:
        await client.aclose()

    manifest["finished_at"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_at"] - started
    write_json(str(manifest_path), manifest)
    print(f"Saved manifest to {manifest_path}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
