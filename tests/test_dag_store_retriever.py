import importlib.util
from pathlib import Path

import numpy as np

from kblam.dag_store_retriever import DAGKVStoreRetriever
from kblam.stores import GraphStore, KVStore


class FakeEmbedder:
    def encode(self, sentences, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return np.asarray([[1.0, 0.0] for _ in sentences], dtype=np.float32)


class FakeExtractor:
    def extract(self, samples):
        output = []
        for sample in samples:
            kv_nodes = []
            for triple in sample["triple_list"]:
                kv = triple["kv_lists"][0]
                kv_nodes.append({"key": kv["key_string"], "value": kv["value_string"]})
            sample["dag"] = {"kv_nodes": kv_nodes, "adj": [], "meta": {"scorer": "fake"}}
            output.append(sample)
        return output


def _add_relation(kv_store, graph_store, subject, predicate, object_value, start_offset):
    offsets = [
        kv_store.add(f"{subject} {predicate}", object_value),
        kv_store.add(f"the entity that {predicate} {object_value} is", subject),
    ]
    assert offsets == [start_offset, start_offset + 1]
    return graph_store.add_triple(
        triple_type="RELATION",
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        kv_offsets=offsets,
        dataset_id="test",
        sample_id=f"sample-{start_offset}",
        source_index=start_offset,
        triple_index=0,
    )


def _build_store(root):
    with KVStore(root / "kv") as kv_store, GraphStore(root / "graph") as graph_store:
        alice_triple = _add_relation(kv_store, graph_store, "Alice", "knows", "Bob", 0)
        _add_relation(kv_store, graph_store, "Bob", "knows", "Carol", 2)
        dave_triple = _add_relation(kv_store, graph_store, "Dave", "knows", "Eve", 4)

        entities = dict(graph_store.entity_nodes())
        vectors_by_name = {
            "Alice": [1.0, 0.0],
            "Bob": [-1.0, 0.0],
            "Carol": [-1.0, 0.0],
            "Dave": [0.99, 0.01],
            "Eve": [-1.0, 0.0],
        }
        node_ids = sorted(entities)
        graph_store.write_entity_embeddings(
            node_ids,
            np.asarray([vectors_by_name[entities[node_id]] for node_id in node_ids], dtype=np.float32),
        )
    return alice_triple, dave_triple


def test_retriever_expands_each_entity_and_recovers_kv_offsets(tmp_path):
    alice_triple, dave_triple = _build_store(tmp_path)
    with DAGKVStoreRetriever(
        tmp_path,
        FakeEmbedder(),
        entity_top_k=2,
        subgraph_hops=1,
        max_triples_per_seed=1,
        search_backend="exact",
    ) as retriever:
        candidate = retriever.retrieve("Who knows whom?")
        prepared = retriever.build_candidate_sample({"question": "Who knows whom?"}, candidate)

    assert {triple.triple_id for triple in candidate.triples} == {alice_triple, dave_triple}
    assert len(candidate.entity_hits) == 2
    assert [hit.name for hit in candidate.entity_hits] == ["Alice", "Dave"]
    assert [triple["kv_lists"][0]["key_string"] for triple in prepared["triple_list"]] == [
        "Alice knows",
        "Dave knows",
    ]


def test_batch_wrapper_restores_original_triples_and_adds_retrieval_metadata(tmp_path):
    _build_store(tmp_path)
    rows = [{"_id": "q1", "question": "Who knows whom?", "triple_list": [{"original": True}]}]
    with DAGKVStoreRetriever(
        tmp_path,
        FakeEmbedder(),
        entity_top_k=1,
        subgraph_hops=1,
        search_backend="exact",
    ) as retriever:
        tool = _load_tool_module()
        output = tool.retrieve_rows(rows, retriever, FakeExtractor())

    assert output[0]["triple_list"] == [{"original": True}]
    assert output[0]["dag"]["kv_nodes"][0] == {"key": "Alice knows", "value": "Bob"}
    assert output[0]["dag"]["meta"]["retrieval"]["candidate_triples"] == 1
    assert "__pathweaver_input_index" not in output[0]


def test_compare_outputs_ignores_retrieval_metadata():
    tool = _load_tool_module()
    reference = [
        {
            "_id": "q1",
            "answer": "b",
            "dag": {"kv_nodes": [{"key": "a", "value": "b"}], "adj": [[0]], "meta": {}},
        }
    ]
    generated = [
        {
            "_id": "q1",
            "answer": "b",
            "dag": {
                "kv_nodes": [{"key": "a", "value": "b"}],
                "adj": [[0]],
                "meta": {"retrieval": {"candidate_triples": 1}},
            },
        }
    ]

    report = tool.compare_outputs(generated, reference)
    assert report["exact_dag_ratio"] == 1.0
    assert report["macro_kv_f1"] == 1.0
    assert report["generated_answer_coverage"] == 1.0


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools/retrieve_pathweaver_dags.py"
    spec = importlib.util.spec_from_file_location("retrieve_pathweaver_dags", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
