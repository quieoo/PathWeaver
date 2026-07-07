#!/usr/bin/env python3
"""Export a PathWeaver V1 store into the current Store V2 layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kblam.stores import GraphStore, GraphStoreV2, KVStore, KVStoreV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store-dir", type=Path, required=True)
    parser.add_argument("--output-store-dir", type=Path, required=True)
    parser.add_argument(
        "--segment-rows",
        type=int,
        default=131072,
        help="Maximum rows per key/value tensor segment.",
    )
    parser.add_argument(
        "--copy-canonical",
        action="store_true",
        help="Also copy the V1 graph/kv directories under canonical/ for debugging and provenance.",
    )
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    src = args.source_store_dir
    dst = args.output_store_dir
    dst.mkdir(parents=True, exist_ok=True)

    with KVStore(src / "kv", create=False) as kv_store:
        KVStoreV2.export_from_v1(dst / "kv_v2", kv_store, segment_rows=args.segment_rows).close()
    with GraphStore(src / "graph", create=False) as graph_store:
        GraphStoreV2.export_from_v1(dst / "graph_v2", graph_store).close()

    if args.copy_canonical:
        copy_tree(src / "kv", dst / "canonical" / "kv")
        copy_tree(src / "graph", dst / "canonical" / "graph")

    manifest = {
        "store_version": "v2-alpha",
        "source_store_dir": str(src.resolve()),
        "output_store_dir": str(dst.resolve()),
        "segment_rows": args.segment_rows,
        "copy_canonical": args.copy_canonical,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
