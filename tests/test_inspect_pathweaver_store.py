import numpy as np

from kblam.stores import GraphStore, KVStore
from tools.inspect_pathweaver_store import inspect_store


def test_inspect_store_reports_graph_and_kv_integrity(tmp_path):
    root = tmp_path / "store"
    with KVStore(root / "kv") as kv_store, GraphStore(root / "graph") as graph_store:
        offsets = []
        for index, (key, value) in enumerate(
            [
                ("Alice knows", "Bob"),
                ("the entity that knows Bob is", "Alice"),
                ("Bob works at", "Acme"),
                ("the entity that works at Acme is", "Bob"),
                ("the age of Alice is", "30"),
            ]
        ):
            offsets.append(
                kv_store.add(
                    key,
                    value,
                    dataset_id="dataset-a",
                    sample_id="sample-1",
                    triple_index=index // 2,
                    kv_index=index % 2,
                )
            )
        graph_store.add_triple(
            triple_type="RELATION",
            subject="Alice",
            predicate="knows",
            object_value="Bob",
            kv_offsets=offsets[0:2],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=0,
        )
        graph_store.add_triple(
            triple_type="RELATION",
            subject="Bob",
            predicate="works at",
            object_value="Acme",
            kv_offsets=offsets[2:4],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=1,
        )
        graph_store.add_triple(
            triple_type="ATTRIBUTE",
            subject="Alice",
            predicate="age",
            object_value="30",
            kv_offsets=offsets[4:5],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=2,
        )
        entities = graph_store.entity_nodes()
        graph_store.write_entity_embeddings(
            [node_id for node_id, _ in entities],
            np.eye(len(entities), dtype=np.float32),
        )

    report = inspect_store(root, max_hops=2, max_seeds=0)

    assert report["graph"]["entities"] == 3
    assert report["graph"]["literals"] == 1
    assert report["graph"]["triple_types"] == {"ATTRIBUTE": 1, "RELATION": 2}
    assert report["topology"]["weakly_connected_components"] == 1
    assert report["topology"]["largest_component_entities"] == 3
    assert report["kv_integrity"]["records"] == 5
    assert report["kv_integrity"]["invalid_references"] == 0
    assert report["kv_integrity"]["orphan_records"] == 0
    assert report["local_expansion"]["by_hop"]["2"]["triples"]["max"] == 3
    assert report["entity_index"]["coverage_ratio"] == 1.0
