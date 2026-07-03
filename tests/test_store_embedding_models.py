import json
import sys
from types import SimpleNamespace

import numpy as np

from kblam.stores import GraphStore, KVStore
from tools.build_pathweaver_stores import build_entity_embeddings, build_kv_embeddings


def test_hnsw_and_kv_embeddings_use_independent_models(tmp_path, monkeypatch):
    loaded_models = []
    kv_load_config = {}

    class FakeFirstModule:
        def half(self):
            kv_load_config["half_called"] = True

    class FakeSentenceTransformer:
        def __init__(self, model_path, model_kwargs=None, tokenizer_kwargs=None):
            self.model_path = str(model_path)
            loaded_models.append(self.model_path)
            if self.model_path.endswith("kv-model"):
                kv_load_config["model_kwargs"] = model_kwargs
                kv_load_config["tokenizer_kwargs"] = tokenizer_kwargs

        def to(self, device):
            kv_load_config["device"] = device
            return self

        def _first_module(self):
            return FakeFirstModule()

        def encode(self, texts, **kwargs):
            marker = 1.0 if self.model_path.endswith("hnsw-model") else 2.0
            return np.asarray([[marker, float(len(text))] for text in texts], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    kv_model = tmp_path / "kv-model"
    hnsw_model = tmp_path / "hnsw-model"
    with KVStore(tmp_path / "store" / "kv") as kv_store, GraphStore(tmp_path / "store" / "graph") as graph_store:
        kv_store.add("Alice knows", "Bob")
        kv_store.add("the entity that knows Bob is", "Alice")
        graph_store.add_triple(
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

        build_entity_embeddings(graph_store, str(hnsw_model), batch_size=8)
        build_kv_embeddings(kv_store, str(kv_model), batch_size=4, prompt_name="query")

        entity_meta = json.loads(graph_store.entity_vector_meta_path.read_text())
        kv_meta = json.loads(kv_store.tensor_meta_path.read_text())
        assert entity_meta["model_path"] == str(hnsw_model.resolve())
        assert kv_meta["model_path"] == str(kv_model.resolve())
        assert kv_meta["prompt_name"] == "query"
        assert kv_meta["encoding_profile"] == "qwen3-embedding-v2"
        assert kv_meta["padding_side"] == "left"
        assert graph_store.entity_vectors_path.is_file()
        assert kv_store.tensor_count == 2

    assert loaded_models == [str(hnsw_model), str(kv_model)]
    assert kv_load_config["model_kwargs"] == {"device_map": "auto"}
    assert kv_load_config["tokenizer_kwargs"] == {"padding_side": "left"}
    assert kv_load_config["half_called"] is True
