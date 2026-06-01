#!/usr/bin/env python3
"""Merge structural reasoning datasets into 1-hop / 2-hop / 3-hop+ splits.

Default inputs match the six datasets listed in `docs/EXPs/2.structual_reasoning.md`.

Hop inference rules:
* `popqa`, `squad_v4`: single-hop datasets.
* `2wiki`: use `len(evidences)` when available; this file is expected to be 2-hop.
* `hotpot`: use the number of unique supporting-fact titles.
* `musique`: parse the `id` prefix like `2hop__...`, fall back to the
  `question_decomposition` length.
* `mintqa`: use `metadata.support_hops`, fall back to supporting-fact titles.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [
    Path("/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl"),
    Path("/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl"),
    Path(
        "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/"
        "2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl"
    ),
    Path(
        "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/"
        "hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl"
    ),
    Path(
        "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/"
        "musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl"
    ),
    Path(
        "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/"
        "mintqa_pruned64_hop2_dag_aa.jsonl"
    ),
]

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "hop_datasets"
HOP_PREFIX_RE = re.compile(r"^(?P<hops>\d+)hop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input JSONL files to merge. Defaults to the six structural reasoning datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Do not add `_source_dataset`, `_hop_count`, `_hop_bucket` fields to each sample.",
    )
    return parser.parse_args()


def detect_source(path: Path) -> str:
    name = path.name.lower()
    if "popqa" in name:
        return "popqa"
    if "squad" in name:
        return "squad_v4"
    if "2wiki" in name:
        return "2wiki"
    if "hotpot" in name:
        return "hotpot"
    if "musique" in name:
        return "musique"
    if "mintqa" in name:
        return "mintqa"
    raise ValueError(f"Cannot infer dataset source from filename: {path}")


def infer_popqa_hops(_: dict[str, Any]) -> int:
    return 1


def infer_squad_hops(_: dict[str, Any]) -> int:
    return 1


def infer_2wiki_hops(sample: dict[str, Any]) -> int:
    evidences = sample.get("evidences")
    if isinstance(evidences, list) and evidences:
        return len(evidences)
    supporting_facts = sample.get("supporting_facts")
    if isinstance(supporting_facts, list) and supporting_facts:
        return len(supporting_facts)
    raise ValueError(f"Cannot infer hops for 2wiki sample: {sample.get('_id')}")


def infer_hotpot_hops(sample: dict[str, Any]) -> int:
    supporting_facts = sample.get("supporting_facts")
    if not isinstance(supporting_facts, list) or not supporting_facts:
        raise ValueError(f"Missing supporting facts for hotpot sample: {sample.get('_id')}")

    titles = []
    for fact in supporting_facts:
        if isinstance(fact, list) and fact:
            titles.append(fact[0])
    if not titles:
        raise ValueError(f"Malformed supporting facts for hotpot sample: {sample.get('_id')}")
    return len(set(titles))


def infer_musique_hops(sample: dict[str, Any]) -> int:
    sample_id = sample.get("id") or sample.get("_id") or ""
    match = HOP_PREFIX_RE.match(sample_id)
    if match:
        return int(match.group("hops"))

    decomposition = sample.get("question_decomposition")
    if isinstance(decomposition, list) and decomposition:
        return len(decomposition)
    raise ValueError(f"Cannot infer hops for musique sample: {sample_id}")


def infer_mintqa_hops(sample: dict[str, Any]) -> int:
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        support_hops = metadata.get("support_hops")
        if isinstance(support_hops, int) and support_hops > 0:
            return support_hops

    supporting_facts = sample.get("supporting_facts")
    if isinstance(supporting_facts, list) and supporting_facts:
        titles = [fact[0] for fact in supporting_facts if isinstance(fact, list) and fact]
        if titles:
            return len(set(titles))
    raise ValueError(f"Cannot infer hops for mintqa sample: {sample.get('id')}")


INFER_HOPS = {
    "popqa": infer_popqa_hops,
    "squad_v4": infer_squad_hops,
    "2wiki": infer_2wiki_hops,
    "hotpot": infer_hotpot_hops,
    "musique": infer_musique_hops,
    "mintqa": infer_mintqa_hops,
}


def hop_bucket(hops: int) -> str:
    if hops == 1:
        return "1hop"
    if hops == 2:
        return "2hop"
    return "3hop_plus"


def annotate_sample(
    sample: dict[str, Any],
    source: str,
    hops: int,
    keep_original: bool,
) -> dict[str, Any]:
    if keep_original:
        return sample

    annotated = copy.deepcopy(sample)
    annotated["_source_dataset"] = source
    annotated["_hop_count"] = hops
    annotated["_hop_bucket"] = hop_bucket(hops)
    return annotated


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def merge_datasets(inputs: list[Path], out_dir: Path, keep_original: bool) -> dict[str, Counter]:
    outputs = {
        "1hop": out_dir / "merged_1hop.jsonl",
        "2hop": out_dir / "merged_2hop.jsonl",
        "3hop_plus": out_dir / "merged_3hop_plus.jsonl",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in outputs.values():
        ensure_parent(path)

    stats_by_source: dict[str, Counter] = defaultdict(Counter)
    total_stats = Counter()

    with (
        outputs["1hop"].open("w", encoding="utf-8") as f_1hop,
        outputs["2hop"].open("w", encoding="utf-8") as f_2hop,
        outputs["3hop_plus"].open("w", encoding="utf-8") as f_3hop,
    ):
        writers = {
            "1hop": f_1hop,
            "2hop": f_2hop,
            "3hop_plus": f_3hop,
        }

        for input_path in inputs:
            source = detect_source(input_path)
            infer_fn = INFER_HOPS[source]

            with input_path.open("r", encoding="utf-8") as infile:
                for line_no, line in enumerate(infile, start=1):
                    sample = json.loads(line)
                    try:
                        hops = infer_fn(sample)
                    except Exception as exc:
                        raise ValueError(
                            f"Failed to infer hops for {input_path}:{line_no}"
                        ) from exc

                    bucket = hop_bucket(hops)
                    annotated = annotate_sample(sample, source, hops, keep_original)
                    writers[bucket].write(json.dumps(annotated, ensure_ascii=False) + "\n")

                    stats_by_source[source][bucket] += 1
                    total_stats[bucket] += 1

    stats_by_source["TOTAL"] = total_stats
    return stats_by_source


def print_summary(stats_by_source: dict[str, Counter], out_dir: Path) -> None:
    print(f"Output directory: {out_dir}")
    print("Merged dataset summary:")
    for source, counter in stats_by_source.items():
        summary = ", ".join(
            f"{bucket}={counter.get(bucket, 0)}"
            for bucket in ("1hop", "2hop", "3hop_plus")
        )
        print(f"  {source}: {summary}")


def main() -> None:
    args = parse_args()
    stats = merge_datasets(args.inputs, args.out_dir, args.keep_original)
    print_summary(stats, args.out_dir)


if __name__ == "__main__":
    main()
