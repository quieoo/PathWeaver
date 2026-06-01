import argparse
import json
import random
import statistics
import time
from typing import Any, Iterable, List, Sequence

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Please install the experiment dependencies "
        "in your runtime environment before running this script."
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: sentence-transformers. Please install the experiment "
        "dependencies in your runtime environment before running this script."
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: tqdm. Please install the experiment dependencies "
        "in your runtime environment before running this script."
    ) from exc

try:
    import hnswlib
except ImportError:
    hnswlib = None


DEFAULT_DATASET_PATH = (
    "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/"
    "msa_merged_memory_docs.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an ANNS index from entity names extracted from dataset triple_list "
            "entries, then measure query latency."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the JSON dataset.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Truncate the dataset to the first N samples before extraction.",
    )
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        required=True,
        help="Local SentenceTransformer embedding model path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k nearest neighbors to retrieve for each query.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=1000,
        help="Number of queries used for latency measurement.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used when encoding entity names.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=1,
        help="Batch size used when encoding query texts for online latency tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="SentenceTransformer device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=None,
        help="Optional cap on the number of unique entity names added to the index.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=100,
        help="HNSW ef_search value.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW ef_construction value.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=32,
        help="HNSW M value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling.",
    )
    parser.add_argument(
        "--show-results",
        type=int,
        default=5,
        help="Print retrieval examples for the first N measured queries.",
    )
    return parser.parse_args()


def load_dataset(path: str) -> list[Any]:
    if path.endswith(".jsonl"):
        rows: list[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_no} in {path}: {exc}"
                    ) from exc
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level JSON dataset to be a list, got {type(data).__name__}"
        )
    return data


def iter_triples(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        triple_list = obj.get("triple_list")
        if isinstance(triple_list, list):
            for item in triple_list:
                if isinstance(item, dict):
                    yield item
        triple_lists = obj.get("triple_lists")
        if isinstance(triple_lists, list):
            for entry in triple_lists:
                if isinstance(entry, dict):
                    yield entry
                elif isinstance(entry, list):
                    for item in entry:
                        if isinstance(item, dict):
                            yield item
        for value in obj.values():
            yield from iter_triples(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_triples(item)


def extract_title_from_doc(doc: str) -> str | None:
    if not isinstance(doc, str):
        return None
    first_line = doc.splitlines()[0].strip() if doc.splitlines() else ""
    if first_line.lower().startswith("title:"):
        title = first_line.split(":", 1)[1].strip()
        return title or None
    return None


def deduplicate_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        norm = item.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def extract_entity_names(samples: Sequence[Any]) -> tuple[List[str], int, int]:
    names: List[str] = []
    triple_name_count = 0
    title_fallback_count = 0

    for sample in samples:
        sample_names: List[str] = []
        for triple in iter_triples(sample):
            name = triple.get("name")
            if isinstance(name, str) and name.strip():
                sample_names.append(name.strip())

        if sample_names:
            triple_name_count += len(sample_names)
            names.extend(sample_names)
            continue

        title = extract_title_from_doc(sample)
        if title:
            title_fallback_count += 1
            names.append(title)

    return deduplicate_keep_order(names), triple_name_count, title_fallback_count


def compute_stats_ms(values_seconds: Sequence[float]) -> dict[str, float]:
    if not values_seconds:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    values_ms = [v * 1000.0 for v in values_seconds]
    return {
        "count": float(len(values_ms)),
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": float(np.percentile(values_ms, 50)),
        "p95_ms": float(np.percentile(values_ms, 95)),
    }


def format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name}: count={int(stats['count'])}, "
        f"mean={stats['mean_ms']:.3f} ms, "
        f"p50={stats['p50_ms']:.3f} ms, "
        f"p95={stats['p95_ms']:.3f} ms"
    )


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=np.float32)


class NumpyExactIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def knn_query(self, query_vec: np.ndarray, k: int):
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        scores = q @ self.vectors.T
        k = min(k, scores.shape[1])
        top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        labels = np.take_along_axis(top_idx, order, axis=1)
        sorted_scores = np.take_along_axis(top_scores, order, axis=1)
        distances = 1.0 - sorted_scores
        return labels, distances


def build_index(
    vectors: np.ndarray,
    *,
    ef_construction: int,
    M: int,
    ef_search: int,
) -> tuple[Any, str]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape={vectors.shape}")
    if hnswlib is None:
        print("hnswlib is not available, using numpy_exact index")
        return NumpyExactIndex(vectors), "numpy_exact"

    num_elements, dim = vectors.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=num_elements,
        ef_construction=ef_construction,
        M=M,
    )
    index.add_items(vectors, np.arange(num_elements))
    index.set_ef(max(ef_search, 1))
    return index, "hnswlib"


def measure_query_latency(
    model: SentenceTransformer,
    index: Any,
    query_texts: Sequence[str],
    entity_names: Sequence[str],
    *,
    query_batch_size: int,
    top_k: int,
    show_results: int,
) -> tuple[list[float], list[float], list[float]]:
    encode_times: List[float] = []
    ann_times: List[float] = []
    total_times: List[float] = []

    for idx, query_text in enumerate(tqdm(query_texts, desc="Measuring query latency")):
        start_total = time.perf_counter()

        start_encode = time.perf_counter()
        query_vec = model.encode(
            [query_text],
            batch_size=query_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        encode_elapsed = time.perf_counter() - start_encode

        start_ann = time.perf_counter()
        labels, distances = index.knn_query(query_vec, k=top_k)
        ann_elapsed = time.perf_counter() - start_ann
        total_elapsed = time.perf_counter() - start_total

        encode_times.append(encode_elapsed)
        ann_times.append(ann_elapsed)
        total_times.append(total_elapsed)

        if idx < show_results:
            retrieved = [
                entity_names[label]
                for label in labels[0].tolist()
            ]
            print(
                f"[Query {idx}] {query_text}\n"
                f"  Retrieved: {retrieved}\n"
                f"  Distances: {distances[0].tolist()}"
            )

    return encode_times, ann_times, total_times


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_dataset(args.dataset_path)

    if args.dataset_limit is not None:
        dataset = dataset[: args.dataset_limit]

    entity_names, triple_name_count, title_fallback_count = extract_entity_names(dataset)
    if not entity_names:
        raise ValueError(
            "No entity names were extracted. Expected triple_list/triple_lists.name, "
            "or a string document beginning with 'Title: ...'."
        )

    if args.max_elements is not None:
        entity_names = entity_names[: args.max_elements]

    top_k = min(args.top_k, len(entity_names))
    query_limit = min(args.query_limit, len(entity_names))
    query_texts = random.sample(entity_names, k=query_limit)

    print("[Config]")
    print(f"dataset_path={args.dataset_path}")
    print(f"dataset_samples={len(dataset)}")
    print(f"triple_name_count={triple_name_count}")
    print(f"title_fallback_count={title_fallback_count}")
    print(f"unique_entity_names={len(entity_names)}")
    print(f"embedding_model_path={args.embedding_model_path}")
    print(f"top_k={top_k}")
    print(f"query_limit={query_limit}")
    print()

    model = SentenceTransformer(args.embedding_model_path, device=args.device)

    print("Encoding entity names...")
    encode_start = time.perf_counter()
    entity_vectors = encode_texts(
        model,
        entity_names,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    encode_elapsed = time.perf_counter() - encode_start
    print(
        f"Entity encoding finished: shape={entity_vectors.shape}, "
        f"total_time={encode_elapsed:.3f} s, "
        f"avg_per_entity={encode_elapsed / len(entity_names) * 1000.0:.3f} ms"
    )

    print("Building index...")
    index_start = time.perf_counter()
    index, backend = build_index(
        entity_vectors,
        ef_construction=args.ef_construction,
        M=args.M,
        ef_search=args.ef_search,
    )
    index_elapsed = time.perf_counter() - index_start
    print(
        f"Index build finished: backend={backend}, total_time={index_elapsed:.3f} s"
    )
    print()

    encode_times, ann_times, total_times = measure_query_latency(
        model,
        index,
        query_texts,
        entity_names,
        query_batch_size=args.query_batch_size,
        top_k=top_k,
        show_results=args.show_results,
    )

    print()
    print("[Latency Summary]")
    print(format_stats("query_encode", compute_stats_ms(encode_times)))
    print(format_stats("ann_search", compute_stats_ms(ann_times)))
    print(format_stats("end_to_end", compute_stats_ms(total_times)))


if __name__ == "__main__":
    main()
class NumpyExactIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def knn_query(self, query_vec: np.ndarray, k: int):
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores = q @ self.vectors.T
        top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        labels = np.take_along_axis(top_idx, order, axis=1)
        sorted_scores = np.take_along_axis(top_scores, order, axis=1)
        distances = 1.0 - sorted_scores
        return labels, distances
