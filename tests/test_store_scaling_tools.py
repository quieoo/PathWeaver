from __future__ import annotations

import json

from kblam.stores import GraphStore, KVStore
from tools.build_pathweaver_store_scale_sweep import parse_append_tier
from tools.build_pathweaver_stores import ingest_dataset


def _sample(sample_id: str, subject: str) -> dict:
    return {
        "_id": sample_id,
        "triple_list": [
            {
                "type": "ATTRIBUTE",
                "name": subject,
                "description_type": "occupation",
                "description": "writer",
            }
        ],
    }


def test_ingest_dataset_respects_source_row_slice(tmp_path):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _sample("zero", "Alice"),
                _sample("one", "Bob"),
                _sample("two", "Carol"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with KVStore(tmp_path / "store" / "kv") as kv_store, GraphStore(
        tmp_path / "store" / "graph"
    ) as graph_store:
        stats = ingest_dataset(
            dataset,
            "slice",
            kv_store,
            graph_store,
            sample_start=1,
            sample_limit=1,
        )
        assert stats == {
            "samples": 1,
            "triples_seen": 1,
            "triples_added": 1,
            "kv_pairs_seen": 1,
        }
        assert [name for _, name in graph_store.entity_nodes()] == ["Bob"]
        source = graph_store._conn.execute(
            "SELECT sample_id, source_index FROM triple_sources"
        ).fetchone()
        assert tuple(source) == ("one", 1)


def test_append_tier_parser_accepts_reproducible_slice(tmp_path):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    tier = parse_append_tier(f"000200::train::{dataset}::100::50")
    assert tier.label == "000200"
    assert tier.dataset_id == "train"
    assert tier.sample_start == 100
    assert tier.sample_limit == 50
