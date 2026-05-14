import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json_objects(path: str) -> List[Dict[str, Any]]:
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
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unsupported JSON root format: {path}")


def read_concatenated_json_objects(path: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    rows: List[Dict[str, Any]] = []
    buf = ""

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while True:
                s = buf.lstrip()
                if not s:
                    buf = ""
                    break
                try:
                    obj, idx = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    break
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
                rows.append(obj)
                buf = s[idx:]

    s = buf.lstrip()
    while s:
        obj, idx = decoder.raw_decode(s)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
        rows.append(obj)
        s = s[idx:].lstrip()

    return rows


def load_samples(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        # Prefer true JSONL first for speed; fall back to concatenated objects.
        try:
            return read_json_objects(path)
        except json.JSONDecodeError:
            return read_concatenated_json_objects(path)
    return read_json_objects(path)


def normalized_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_fingerprint(sample: Dict[str, Any]) -> str:
    base = copy.deepcopy(sample)
    for key in [
        "id",
        "_id",
        "source_id",
        "dataset",
        "dataset_name",
        "merged_from",
        "merge_meta",
    ]:
        base.pop(key, None)
    return hashlib.md5(normalized_json(base).encode("utf-8")).hexdigest()


def sample_quality(sample: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    answer_sufficient = 1 if sample.get("answer_sufficient") is True else 0
    triple_count = len(sample.get("triple_list") or [])
    context_count = len(sample.get("context") or [])
    revision_count = len(sample.get("revision_notes") or [])
    payload_size = len(normalized_json(sample))
    return (
        answer_sufficient,
        triple_count,
        context_count,
        revision_count,
        payload_size,
    )


def canonical_source_id(sample: Dict[str, Any], fallback_index: int) -> str:
    raw = sample.get("id", sample.get("_id"))
    if raw is None or str(raw).strip() == "":
        return f"missingid_{fallback_index:08d}"
    return str(raw).strip()


def prepare_sample(
    sample: Dict[str, Any],
    dataset_name: str,
    fallback_index: int,
) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    source_id = canonical_source_id(out, fallback_index)
    merged_id = f"{dataset_name}_{source_id}"

    out["dataset"] = dataset_name
    out["source_id"] = source_id
    out["id"] = merged_id
    out["_id"] = merged_id
    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_dataset_arg(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --dataset value {raw!r}. Expected format like hotpot=/path/to/file.jsonl"
        )
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(
            f"Invalid --dataset value {raw!r}. Expected non-empty dataset name and path."
        )
    return name, path


def merge_datasets(
    dataset_specs: List[Tuple[str, str]],
    require_answer_sufficient: bool,
    dedupe_by_content: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    content_seen: Dict[str, str] = {}
    stats: Dict[str, Any] = {
        "input_datasets": [],
        "input_samples": 0,
        "kept_after_answer_sufficient": 0,
        "dropped_answer_sufficient": 0,
        "duplicate_ids_dropped": 0,
        "duplicate_contents_dropped": 0,
        "final_samples": 0,
    }

    for dataset_name, path in dataset_specs:
        samples = load_samples(path)
        dataset_stat = {
            "dataset": dataset_name,
            "path": path,
            "loaded": len(samples),
            "kept_after_answer_sufficient": 0,
            "dropped_answer_sufficient": 0,
            "duplicate_ids_dropped": 0,
            "duplicate_contents_dropped": 0,
            "final_kept": 0,
        }
        stats["input_datasets"].append(dataset_stat)
        stats["input_samples"] += len(samples)

        for idx, sample in enumerate(samples):
            if require_answer_sufficient and sample.get("answer_sufficient") is not True:
                dataset_stat["dropped_answer_sufficient"] += 1
                stats["dropped_answer_sufficient"] += 1
                continue

            prepared = prepare_sample(sample, dataset_name, idx)
            merged_id = prepared["id"]
            dataset_stat["kept_after_answer_sufficient"] += 1
            stats["kept_after_answer_sufficient"] += 1

            prev = merged_by_id.get(merged_id)
            if prev is not None:
                if sample_quality(prepared) > sample_quality(prev):
                    merged_by_id[merged_id] = prepared
                dataset_stat["duplicate_ids_dropped"] += 1
                stats["duplicate_ids_dropped"] += 1
                continue

            if dedupe_by_content:
                fp = content_fingerprint(prepared)
                prev_id = content_seen.get(fp)
                if prev_id is not None:
                    prev_sample = merged_by_id[prev_id]
                    if sample_quality(prepared) > sample_quality(prev_sample):
                        merged_by_id.pop(prev_id)
                        merged_by_id[merged_id] = prepared
                        content_seen[fp] = merged_id
                    dataset_stat["duplicate_contents_dropped"] += 1
                    stats["duplicate_contents_dropped"] += 1
                    continue
                content_seen[fp] = merged_id

            merged_by_id[merged_id] = prepared

    merged = list(merged_by_id.values())
    merged.sort(key=lambda x: x["id"])

    final_counts = Counter(sample["dataset"] for sample in merged)
    for dataset_stat in stats["input_datasets"]:
        dataset_stat["final_kept"] = int(final_counts.get(dataset_stat["dataset"], 0))

    stats["final_samples"] = len(merged)
    stats["final_dataset_counts"] = dict(sorted(final_counts.items()))
    return merged, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec in the form dataset_name=/path/to/file.jsonl. Repeat for multiple datasets.",
    )
    ap.add_argument("--output", required=True, help="Merged output JSONL path.")
    ap.add_argument(
        "--stats-output",
        default="",
        help="Optional JSON path for merge statistics.",
    )
    ap.add_argument(
        "--allow-answer-insufficient",
        action="store_true",
        help="Keep samples whose answer_sufficient is not True.",
    )
    ap.add_argument(
        "--disable-content-dedupe",
        action="store_true",
        help="Only dedupe by final merged id, not by sample content.",
    )
    args = ap.parse_args()

    dataset_specs = [parse_dataset_arg(x) for x in args.dataset]
    merged, stats = merge_datasets(
        dataset_specs=dataset_specs,
        require_answer_sufficient=not args.allow_answer_insufficient,
        dedupe_by_content=not args.disable_content_dedupe,
    )

    write_jsonl(args.output, merged)
    if args.stats_output:
        os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[DONE] wrote {len(merged)} samples to {args.output}")


if __name__ == "__main__":
    main()
