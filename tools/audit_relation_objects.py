#!/usr/bin/env python3
"""Audit suspicious RELATION triples whose object looks like a generic role/type term."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kblam.stores.common import canonical_entity_key, normalize_text


DEFAULT_GENERIC_TERMS = {
    "actor",
    "actress",
    "administrator",
    "advisor",
    "ambassador",
    "architect",
    "artist",
    "athlete",
    "author",
    "award",
    "band",
    "brother",
    "capital",
    "ceo",
    "chairman",
    "child",
    "coach",
    "college",
    "company",
    "composer",
    "country",
    "county",
    "daughter",
    "director",
    "doctor",
    "drummer",
    "dynasty",
    "editor",
    "emperor",
    "engineer",
    "father",
    "film",
    "founder",
    "friend",
    "government",
    "guitarist",
    "headquarters",
    "hero",
    "historian",
    "home town",
    "hospital",
    "husband",
    "king",
    "label",
    "language",
    "lawyer",
    "leader",
    "league",
    "lieutenant",
    "location",
    "magazine",
    "manager",
    "mayor",
    "minister",
    "model",
    "mother",
    "movie",
    "musician",
    "nationality",
    "newspaper",
    "novelist",
    "organization",
    "owner",
    "parent",
    "performer",
    "person",
    "philosopher",
    "photographer",
    "place of birth",
    "place of death",
    "player",
    "poet",
    "politician",
    "president",
    "prince",
    "producer",
    "professor",
    "province",
    "queen",
    "record label",
    "religion",
    "researcher",
    "river",
    "school",
    "scientist",
    "screenwriter",
    "secretary",
    "senator",
    "singer",
    "sister",
    "songwriter",
    "son",
    "spouse",
    "state",
    "student",
    "teacher",
    "team",
    "television",
    "town",
    "university",
    "village",
    "vocalist",
    "wife",
    "writer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--store-version", choices=["auto", "v1", "v2"], default="auto")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sample-per-object", type=int, default=3)
    parser.add_argument("--generic-terms-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def detect_store_version(store_dir: Path) -> str:
    if (store_dir / "graph_v2").is_dir():
        return "v2"
    if (store_dir / "graph").is_dir():
        return "v1"
    raise FileNotFoundError(f"Cannot detect store version under {store_dir}")


def load_generic_terms(path: Path | None) -> set[str]:
    terms = set(DEFAULT_GENERIC_TERMS)
    if path is None:
        return terms
    for line in path.read_text(encoding="utf-8").splitlines():
        item = canonical_entity_key(line)
        if item:
            terms.add(item)
    return terms


def percentile_summary(values: list[int | float]) -> dict[str, int | float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "min": float(data.min()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


def is_suspicious_object(object_name: str, generic_terms: set[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    normalized = normalize_text(object_name)
    key = canonical_entity_key(normalized)
    if key in generic_terms:
        reasons.append("generic_term_lexicon")

    tokens = [token for token in key.split() if token]
    if tokens and len(tokens) <= 3 and all(token.isalpha() for token in tokens):
        if normalized == normalized.lower():
            reasons.append("lowercase_common_phrase")

    if len(tokens) == 1 and tokens[0] in {"director", "mother", "father", "husband", "wife", "composer", "performer"}:
        reasons.append("singleton_role_word")

    return bool(reasons), reasons


def audit_v1(store_dir: Path, generic_terms: set[str], sample_per_object: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    db_path = store_dir / "graph" / "graph_store.sqlite3"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                t.triple_id,
                s.canonical_name AS subject_name,
                t.predicate,
                o.canonical_name AS object_name,
                t.object_kind,
                COALESCE(
                    (
                        SELECT ts.title FROM triple_sources ts
                        WHERE ts.triple_id = t.triple_id AND ts.title != ''
                        ORDER BY ts.dataset_id, ts.source_index, ts.triple_index
                        LIMIT 1
                    ),
                    ''
                ) AS source_title
            FROM triples t
            JOIN nodes s ON s.node_id = t.subject_id
            JOIN nodes o ON o.node_id = t.object_id
            WHERE t.triple_type = 'RELATION' AND t.object_kind = 'entity'
            ORDER BY t.triple_id
            """
        ).fetchall()
    finally:
        conn.close()

    return summarize_audit(rows, generic_terms, sample_per_object)


def audit_v2(store_dir: Path, generic_terms: set[str], sample_per_object: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = store_dir / "graph_v2"
    entity_node_ids = np.load(root / "entity_node_ids.npy", mmap_mode="r")
    entity_names = json.loads((root / "entity_names.json").read_text(encoding="utf-8"))
    triple_subject_pos = np.load(root / "triple_subject_pos.npy", mmap_mode="r")
    triple_predicates = json.loads((root / "triple_predicates.json").read_text(encoding="utf-8"))
    triple_object_names = json.loads((root / "triple_object_names.json").read_text(encoding="utf-8"))
    triple_object_kind = np.load(root / "triple_object_kind.npy", mmap_mode="r")
    triple_type = np.load(root / "triple_type.npy", mmap_mode="r")
    triple_titles = json.loads((root / "triple_titles.json").read_text(encoding="utf-8"))
    triple_ids = np.load(root / "triple_ids.npy", mmap_mode="r")

    rows: list[dict[str, Any]] = []
    for triple_idx in range(int(triple_ids.shape[0])):
        if str(triple_type[triple_idx]) != "RELATION" or str(triple_object_kind[triple_idx]) != "entity":
            continue
        subject_name = entity_names[int(triple_subject_pos[triple_idx])]
        rows.append(
            {
                "triple_id": int(triple_ids[triple_idx]),
                "subject_name": subject_name,
                "predicate": str(triple_predicates[triple_idx]),
                "object_name": str(triple_object_names[triple_idx]),
                "object_kind": "entity",
                "source_title": str(triple_titles[triple_idx]),
            }
        )
    return summarize_audit(rows, generic_terms, sample_per_object)


def summarize_audit(
    rows: Any,
    generic_terms: set[str],
    sample_per_object: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = 0
    suspicious = 0
    object_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    samples_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    suspicious_rows: list[dict[str, Any]] = []

    for row in rows:
        total += 1
        object_name = str(row["object_name"])
        flagged, reasons = is_suspicious_object(object_name, generic_terms)
        if not flagged:
            continue
        suspicious += 1
        object_key = canonical_entity_key(object_name)
        object_counter[object_key] += 1
        reason_counter.update(reasons)
        record = {
            "triple_id": int(row["triple_id"]),
            "subject": str(row["subject_name"]),
            "predicate": str(row["predicate"]),
            "object": object_name,
            "object_key": object_key,
            "title": str(row["source_title"]),
            "reasons": reasons,
        }
        suspicious_rows.append(record)
        if len(samples_by_object[object_key]) < sample_per_object:
            samples_by_object[object_key].append(record)

    top_objects = []
    for object_key, count in object_counter.most_common():
        top_objects.append(
            {
                "object_key": object_key,
                "count": count,
                "samples": samples_by_object[object_key],
            }
        )

    summary = {
        "relation_entity_triples": total,
        "suspicious_relation_entity_triples": suspicious,
        "suspicious_ratio": (suspicious / total) if total else 0.0,
        "unique_suspicious_objects": len(object_counter),
        "top_suspicious_object_frequency": percentile_summary(list(object_counter.values())),
        "reason_breakdown": dict(reason_counter.most_common()),
    }
    return summary, top_objects


def main() -> None:
    args = parse_args()
    version = detect_store_version(args.store_dir) if args.store_version == "auto" else args.store_version
    generic_terms = load_generic_terms(args.generic_terms_file)

    if version == "v1":
        summary, top_objects = audit_v1(args.store_dir, generic_terms, args.sample_per_object)
    else:
        summary, top_objects = audit_v2(args.store_dir, generic_terms, args.sample_per_object)

    payload = {
        "config": {
            "store_dir": str(args.store_dir),
            "store_version": version,
            "top_k": args.top_k,
            "sample_per_object": args.sample_per_object,
            "generic_terms_count": len(generic_terms),
        },
        "summary": summary,
        "top_suspicious_objects": top_objects[: args.top_k],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
