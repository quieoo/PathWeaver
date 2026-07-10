import numpy as np
import json
import pytest

from kblam.stores import EntityResolver, GraphStore, GraphStoreV2


def test_graph_store_deduplicates_triples_and_recovers_local_graph(tmp_path):
    resolver = EntityResolver({"Wunna": "Maung Wunna"})
    with GraphStore(tmp_path / "graph", resolver=resolver) as store:
        directed_by = store.add_triple(
            triple_type="RELATION",
            subject="Wearing Velvet Slippers under a Golden Umbrella",
            predicate="directed by",
            object_value="Wunna",
            kv_offsets=[0, 1],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=0,
            title="Film facts",
        )
        duplicate = store.add_triple(
            triple_type="RELATION",
            subject="Wearing Velvet Slippers under a Golden Umbrella",
            predicate="directed by",
            object_value="Maung Wunna",
            kv_offsets=[0, 1],
            dataset_id="dataset-b",
            sample_id="sample-2",
            source_index=0,
            triple_index=4,
        )
        won = store.add_triple(
            triple_type="RELATION",
            subject="Maung Wunna",
            predicate="won",
            object_value="Myanmar Motion Picture Academy Awards",
            kv_offsets=[2, 3],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=1,
        )
        store.add_triple(
            triple_type="ATTRIBUTE",
            subject="Maung Wunna",
            predicate="death year",
            object_value="2011",
            kv_offsets=[4, 5],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=2,
        )

        assert duplicate == directed_by
        film_id = store.resolve_node_id("Wearing Velvet Slippers under a Golden Umbrella")
        wunna_id = store.resolve_node_id("Wunna")
        assert film_id is not None
        assert wunna_id == store.resolve_node_id("Maung Wunna")
        assert store.get_node_name(wunna_id) == "Maung Wunna"

        node_ids, triples = store.get_local_subgraph([film_id], hops=2)
        assert wunna_id in node_ids
        assert {triple.triple_id for triple in triples} >= {directed_by, won}
        assert store.get_triple(directed_by).title == "Film facts"
        assert store.get_triple(won).kv_offsets == (2, 3)
        assert store.stats()["triples"] == 3
        assert store.stats()["triple_sources"] == 4


def test_graph_store_entity_vector_search(tmp_path):
    with GraphStore(tmp_path / "graph") as store:
        store.add_triple(
            triple_type="RELATION",
            subject="Alice",
            predicate="knows",
            object_value="Bob",
            kv_offsets=[0, 1],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=0,
        )
        alice_id = store.resolve_node_id("Alice")
        bob_id = store.resolve_node_id("Bob")
        assert alice_id is not None and bob_id is not None
        store.write_entity_embeddings(
            [alice_id, bob_id],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            metadata={"model_path": "/models/hnsw"},
        )
        assert json.loads(store.entity_vector_meta_path.read_text())["model_path"] == "/models/hnsw"

        result = store.search_entities(np.asarray([0.9, 0.1], dtype=np.float32), top_k=1, backend="exact")
        assert result[0][0] == alice_id
        assert result[0][1] > 0.9

        store.add_triple(
            triple_type="RELATION",
            subject="Carol",
            predicate="knows",
            object_value="Alice",
            kv_offsets=[2, 3],
            dataset_id="dataset-b",
            sample_id="sample-2",
            source_index=0,
            triple_index=0,
        )
        assert not store.entity_vectors_path.exists()
        assert not store.entity_vector_meta_path.exists()


def test_graph_store_limits_incident_triples_per_node(tmp_path):
    with GraphStore(tmp_path / "graph") as store:
        for index in range(3):
            store.add_triple(
                triple_type="RELATION",
                subject="Hub",
                predicate=f"connects to {index}",
                object_value=f"Leaf {index}",
                kv_offsets=[index * 2, index * 2 + 1],
                dataset_id="dataset-a",
                sample_id=f"sample-{index}",
                source_index=0,
                triple_index=index,
            )
        hub_id = store.resolve_node_id("Hub")
        assert hub_id is not None

        _, triples = store.get_local_subgraph([hub_id], hops=1, max_incident_triples_per_node=2)
        assert len(triples) == 2


def test_graph_store_v2_limits_incident_triples_per_node(tmp_path):
    with GraphStore(tmp_path / "graph") as source:
        for index in range(3):
            source.add_triple(
                triple_type="RELATION",
                subject="Hub",
                predicate=f"connects to {index}",
                object_value=f"Leaf {index}",
                kv_offsets=[index * 2, index * 2 + 1],
                dataset_id="dataset-a",
                sample_id=f"sample-{index}",
                source_index=0,
                triple_index=index,
            )
        hub_id = source.resolve_node_id("Hub")
        assert hub_id is not None
        GraphStoreV2.export_from_v1(tmp_path / "graph_v2", source)

    with GraphStoreV2(tmp_path / "graph_v2", create=False) as store:
        _, triples = store.get_local_subgraph([hub_id], hops=1, max_incident_triples_per_node=2)
        assert len(triples) == 2


def test_graph_store_keeps_unicode_and_symbol_only_nodes(tmp_path):
    with GraphStore(tmp_path / "graph") as store:
        store.add_triple(
            triple_type="ATTRIBUTE",
            subject="Greek letters",
            predicate="symbol",
            object_value="ΣΞ",
            kv_offsets=[0],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=0,
        )
        store.add_triple(
            triple_type="RELATION",
            subject="?",
            predicate="certified by",
            object_value="Authority",
            kv_offsets=[1],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=1,
        )

        assert store.resolve_node_id("ΣΞ") is not None
        assert store.resolve_node_id("?") is not None


def test_graph_store_reuses_loaded_hnsw_index(tmp_path):
    pytest.importorskip("hnswlib")
    graph_dir = tmp_path / "graph"
    with GraphStore(graph_dir) as store:
        store.add_triple(
            triple_type="RELATION",
            subject="Alice",
            predicate="knows",
            object_value="Bob",
            kv_offsets=[0, 1],
            dataset_id="dataset-a",
            sample_id="sample-1",
            source_index=0,
            triple_index=0,
        )
        nodes = store.entity_nodes()
        store.write_entity_embeddings(
            [node_id for node_id, _ in nodes],
            np.eye(len(nodes), dtype=np.float32),
        )
        store.build_hnsw_index()

    with GraphStore(graph_dir, create=False) as store:
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        store.search_entities(query, top_k=1, backend="hnsw")
        loaded_index = store._hnsw_index
        store.search_entities(query, top_k=1, backend="hnsw")
        assert loaded_index is not None
        assert store._hnsw_index is loaded_index
