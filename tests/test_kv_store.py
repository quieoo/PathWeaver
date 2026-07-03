import numpy as np
import pytest
import json

from kblam.stores import KVStore


def test_kv_store_offsets_deduplicate_and_persist(tmp_path):
    root = tmp_path / "kv"
    with KVStore(root) as store:
        first = store.add(
            "Alice parent",
            "Bob",
            dataset_id="dataset-a",
            sample_id="sample-1",
            triple_index=0,
            kv_index=0,
        )
        duplicate = store.add(
            "Alice parent",
            "Bob",
            dataset_id="dataset-b",
            sample_id="sample-2",
            triple_index=3,
            kv_index=0,
        )
        second = store.add("Bob occupation", "Engineer")

        assert (first, duplicate, second) == (0, 0, 1)
        assert len(store) == 2
        assert store.get(0).value_text == "Bob"

    with KVStore(root, create=False) as reopened:
        assert [record.offset for record in reopened.iter_records()] == [0, 1]


def test_kv_store_tensor_arrays_follow_offsets(tmp_path):
    with KVStore(tmp_path / "kv") as store:
        store.add("k0", "v0")
        store.add("k1", "v1")
        store.write_tensors(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            metadata={"model_path": "/models/kv"},
        )
        assert json.loads(store.tensor_meta_path.read_text())["model_path"] == "/models/kv"
        keys, values = store.get_tensors([1, 0])
        np.testing.assert_array_equal(keys, [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(values, [[0.0, 2.0], [2.0, 0.0]])

        store.add("k2", "v2")
        assert store.append_tensors(
            np.asarray([[3.0, 3.0]], dtype=np.float32),
            np.asarray([[4.0, 4.0]], dtype=np.float32),
        ) == (2, 3)
        keys, _ = store.get_tensors([2])
        np.testing.assert_array_equal(keys, [[3.0, 3.0]])

        store.add("k3", "v3")
        with pytest.raises(ValueError, match="shapes must match"):
            store.append_tensors(
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[1.0, 2.0]], dtype=np.float32),
            )
