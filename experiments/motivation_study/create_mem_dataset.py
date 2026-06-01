import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kblam.utils.dataset import iter_row_docs, load_dataset  # noqa: E402


DEFAULT_DATASETS = [
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_train.json",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_train.json",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_dev.jsonl",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_dataset.json",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_dev.json",
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_train.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple source datasets into one large MSA memory-doc dataset."
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Source dataset paths. Defaults to the motivation-study dataset list.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save merged memory docs as a JSON file.",
    )
    return parser.parse_args()


def collect_unique_docs(dataset_path: str) -> Tuple[List[str], Dict[str, int]]:
    rows = load_dataset(dataset_path)
    docs: List[str] = []
    seen = set()
    raw_doc_count = 0

    for row in rows:
        for doc in iter_row_docs(row):
            raw_doc_count += 1
            if doc and doc not in seen:
                seen.add(doc)
                docs.append(doc)

    stats = {
        "rows": len(rows),
        "raw_docs": raw_doc_count,
        "unique_docs": len(docs),
    }
    return docs, stats


def main() -> None:
    args = parse_args()

    merged_docs: List[str] = []
    global_seen = set()
    total_raw_docs = 0
    total_unique_before_merge = 0

    print("Merging datasets into one memory-doc collection:\n")

    for dataset_path in args.dataset_paths:
        docs, stats = collect_unique_docs(dataset_path)
        total_raw_docs += stats["raw_docs"]
        total_unique_before_merge += stats["unique_docs"]

        new_docs = 0
        for doc in docs:
            if doc in global_seen:
                continue
            global_seen.add(doc)
            merged_docs.append(doc)
            new_docs += 1

        print(f"[{Path(dataset_path).name}]")
        print(f"  Rows: {stats['rows']}")
        print(f"  Raw docs: {stats['raw_docs']}")
        print(f"  Unique docs in dataset: {stats['unique_docs']}")
        print(f"  New docs added to merged set: {new_docs}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_docs, f, ensure_ascii=False, indent=2)

    print("\n[Summary]")
    print(f"Datasets merged: {len(args.dataset_paths)}")
    print(f"Total raw docs across datasets: {total_raw_docs}")
    print(f"Total per-dataset unique docs (before global dedup): {total_unique_before_merge}")
    print(f"Final merged unique docs: {len(merged_docs)}")
    print(f"Saved merged memory docs to: {output_path}")


if __name__ == "__main__":
    main()
