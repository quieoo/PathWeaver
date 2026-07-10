import importlib.util
from pathlib import Path

import numpy as np
import torch

from kblam.dag_store_retriever import DAGExtractionConfig, DAGKVStoreRetriever, TrainableDAGExtractor
from kblam.online_dag_kv_retriever import OnlineDAGKBRetriever
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


class OffsetExtractor:
    backend = "v8-answer-blind"

    def extract(self, samples):
        output = []
        for sample in samples:
            assert "answer" not in sample
            nodes = [
                {
                    "key": kv["key_string"],
                    "value": kv["value_string"],
                    "kv_offset": kv["kv_offset"],
                }
                for triple in sample["triple_list"]
                for kv in triple["kv_lists"]
            ]
            adj = [[0] * len(nodes) for _ in nodes]
            for index in range(len(nodes) - 1):
                adj[index][index + 1] = 1
            row = dict(sample)
            row["dag"] = {"kv_nodes": nodes, "adj": adj, "meta": {"answer_free_inference": True}}
            output.append(row)
        return output


class ProfiledExtractor:
    backend = "v8-answer-blind"

    def __init__(self):
        self.last_profile = {
            "build_graph": 0.01,
            "encode": 0.02,
            "feature_prepare": 0.03,
            "model_score": 0.04,
            "select_export": 0.05,
            "total": 0.15,
        }

    def extract(self, samples):
        output = []
        for sample in samples:
            row = dict(sample)
            row["dag"] = {"kv_nodes": [], "adj": [], "meta": {"answer_free_inference": True}}
            output.append(row)
        return output


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 2
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode_key(self, base_emb):
        return torch.from_numpy(base_emb) + 1

    def encode_val(self, base_emb):
        return torch.from_numpy(base_emb) + 2


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
        rows = len(kv_store)
        key_tensors = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        value_tensors = key_tensors + 100
        kv_store.write_tensors(key_tensors, value_tensors)
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
        prepared = retriever.build_candidate_sample(
            {"question": "Who knows whom?", "context": [{"triple_list": [{"stale": True}]}]},
            candidate,
        )

    assert {triple.triple_id for triple in candidate.triples} == {alice_triple, dave_triple}
    assert len(candidate.entity_hits) == 2
    assert prepared["context"] == []
    assert [hit.name for hit in candidate.entity_hits] == ["Alice", "Dave"]
    assert [triple["kv_lists"][0]["key_string"] for triple in prepared["triple_list"]] == [
        "Alice knows",
        "Dave knows",
    ]


def test_hybrid_seed_prefers_long_exact_entity_mention(tmp_path):
    _build_store(tmp_path)
    with DAGKVStoreRetriever(
        tmp_path,
        FakeEmbedder(),
        entity_top_k=1,
        subgraph_hops=1,
        search_backend="exact",
        seed_strategy="hybrid",
        mention_min_chars=4,
    ) as retriever:
        candidate = retriever.retrieve("What does Dave know?")

    assert [(hit.name, hit.source) for hit in candidate.entity_hits] == [("Dave", "mention")]
    assert {(triple.subject, triple.object) for triple in candidate.triples} == {("Dave", "Eve")}


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


def test_trainable_extractor_adapts_v8_load_and_infer_api(tmp_path):
    script = tmp_path / "fake_v8.py"
    script.write_text(
        """
def load_models(path, cpu):
    return 'edge', 'node', {'is_joint': False}, 'cpu'

def infer(args, samples, embedder, edge_model, node_model, checkpoint, device):
    output = []
    for sample in samples:
        row = dict(sample)
        row['dag'] = {
            'kv_nodes': [],
            'adj': [],
            'meta': {
                'answer_free_inference': True,
                'selection_mode': args.selection_mode,
                'terminal_reranker': args.terminal_reranker,
            },
        }
        output.append(row)
    return output
""",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    extractor = TrainableDAGExtractor(
        script,
        checkpoint,
        FakeEmbedder(),
        config=DAGExtractionConfig(selection_mode="legacy", terminal_reranker="heuristic"),
        cpu=True,
    )

    output = extractor.extract([{"_id": "q1", "question": "Question"}])

    assert extractor.backend == "v8-answer-blind"
    assert output[0]["dag"]["meta"] == {
        "answer_free_inference": True,
        "selection_mode": "legacy",
        "terminal_reranker": "heuristic",
    }


def test_online_retriever_aligns_offsets_tensors_and_adjacency(tmp_path):
    _build_store(tmp_path)
    online = OnlineDAGKBRetriever(
        encoder=FakeEncoder(),
        store_dir=str(tmp_path),
        entity_embedder=FakeEmbedder(),
        dag_extractor=OffsetExtractor(),
        entity_top_k=1,
        subgraph_hops=1,
        search_backend="exact",
        use_multihop_adj=True,
        max_hops=10,
        hop_decay=1.0,
        dynamic_hops_by_longest_path=True,
    )
    try:
        result = online.get_kb_for_queries(["Who knows whom?"], device="cpu")[0]
    finally:
        online.close()

    assert [node["kv_offset"] for node in result.dag["kv_nodes"]] == [0, 1]
    np.testing.assert_allclose(result.kb_keys.numpy(), [[1, 2], [3, 4]])
    np.testing.assert_allclose(result.kb_values.numpy(), [[102, 103], [104, 105]])
    np.testing.assert_allclose(result.kb_adj.to_dense().numpy(), [[0, 1], [0, 0]])
    assert result.dag["meta"]["answer_free_inference"] is True


def test_online_retriever_forwards_incident_triple_cap(tmp_path):
    _build_store(tmp_path)
    online = OnlineDAGKBRetriever(
        encoder=FakeEncoder(),
        store_dir=str(tmp_path),
        entity_embedder=FakeEmbedder(),
        dag_extractor=OffsetExtractor(),
        entity_top_k=1,
        subgraph_hops=1,
        max_incident_triples_per_node=1,
        search_backend="exact",
    )
    try:
        result = online.get_kb_for_queries(["Who knows whom?"], device="cpu")[0]
    finally:
        online.close()

    assert len(result.dag["kv_nodes"]) == 2
    assert result.dag["meta"]["retrieval"]["candidate_triples"] == 1


def test_online_retriever_surfaces_dag_substage_profile(tmp_path):
    _build_store(tmp_path)
    online = OnlineDAGKBRetriever(
        encoder=FakeEncoder(),
        store_dir=str(tmp_path),
        entity_embedder=FakeEmbedder(),
        dag_extractor=ProfiledExtractor(),
        entity_top_k=1,
        subgraph_hops=1,
        search_backend="exact",
    )
    try:
        online.get_kb_for_queries(["Who knows whom?"], device="cpu")
        stats = online.stats()
    finally:
        online.close()

    assert stats["dag_average_seconds"]["build_graph"] == 0.01
    assert stats["dag_average_seconds"]["encode"] == 0.02
    assert stats["dag_average_seconds"]["feature_prepare"] == 0.03
    assert stats["dag_average_seconds"]["model_score"] == 0.04
    assert stats["dag_average_seconds"]["select_export"] == 0.05
    assert stats["dag_average_seconds"]["total"] == 0.15


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools/retrieve_pathweaver_dags.py"
    spec = importlib.util.spec_from_file_location("retrieve_pathweaver_dags", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
