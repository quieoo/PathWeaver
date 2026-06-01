import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import statistics
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from kblam.metrics_evaluator import evaluate_model_outputs, full_evaluation
from kblam.utils.dataset import (
    build_memory_docs,
    build_query_samples,
    compute_retrieval_recall_stats,
    extract_doc_title,
    get_question,
    get_sample_id,
    load_dataset,
    load_queryset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MSA_ROOT = REPO_ROOT / "MSA"
if str(MSA_ROOT) not in sys.path:
    sys.path.append(str(MSA_ROOT))

from src.msa_service import MSAEngine  # noqa: E402
from src.config.memory_config import GenerateConfig, MemoryConfig, ModelConfig  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run MSA on a single unified dataset file.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Unified dataset json path. Compatible with vector_rag.py inputs.",
    )
    parser.add_argument("--dataset-type", type=str, default="2wiki", help="Dataset type for logging.")
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optionally truncate the loaded dataset to the first N rows before building memory/query data.",
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of leading rows used as queries.")
    parser.add_argument("--mintqa_min_hop", type=int, default=None, help="MintQA minimum support hops for query filtering.")
    parser.add_argument(
        "--queryset-path",
        type=str,
        default=None,
        help="Optional query set path. Supports json/jsonl samples.",
    )
    parser.add_argument(
        "--memory-docs",
        type=int,
        default=None,
        help="Number of leading rows used to build the memory bank. Defaults to the full dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/n0/models/MSA-4B",
        help="MSA model path.",
    )
    parser.add_argument("--max-batch-size", type=int, default=16, help="Per-GPU query batch size.")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2048, help="Max generated tokens.")
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--max-query-seq-len", type=int, default=0)
    parser.add_argument("--template", type=str, default="QWEN3_INSTRUCT_TEMPLATE")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--max-chunk-per-block", type=int, default=16 * 1024)
    parser.add_argument(
        "--pooling-kernel-size",
        type=int,
        default=None,
        help="Override pooling_kernel_size from the model msa_config.",
    )
    parser.add_argument(
        "--doc-top-k",
        type=int,
        default=None,
        help="Override doc_top_k from the model msa_config.",
    )
    parser.add_argument(
        "--memory-cache-dir",
        type=str,
        default=None,
        help="Directory for serialized MSA memory cache. Reuses prefills across runs when possible.",
    )
    parser.add_argument(
        "--disable-memory-cache",
        action="store_true",
        help="Disable serialized memory cache reuse even if --memory-cache-dir is set.",
    )
    parser.add_argument("--dis_out_path", type=str, default=None, help="Output file path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def sort_requests(data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items_with_length = [(item, len(item["question"])) for item in data_items]
    items_with_length.sort(key=lambda x: x[1])
    return [item[0] for item in items_with_length]


def should_regenerate(request: Dict[str, Any], response: str):
    suffix = "<|object_ref_end|>"
    if response.endswith(suffix) and "The answer to the question is:" not in response:
        if not request.get("regenerated", False):
            request["regenerated"] = True
        split_token = "\nPlease answer the question based"
        if split_token in response:
            response = split_token + response.split(split_token, 1)[1]
        response = response.replace("<|endoftext|>", "")
        return "<regenerate>" + response
    return None


def parse_pred_answer(generated_text: str) -> str:
    cleaned = generated_text.replace("<|endoftext|>", "").strip()
    marker = "The answer to the question is:"
    if marker in cleaned:
        answer = cleaned.split(marker, 1)[-1]
        answer = answer.split("<|im_end|>", 1)[0]
        answer = answer.strip()
        if answer:
            return answer
    if "<|im_end|>" in cleaned:
        cleaned = cleaned.split("<|im_end|>", 1)[0].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def read_msa_config(model_path: str) -> Dict[str, Any]:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f).get("msa_config", {})


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[idx]


def summarize_latency(name: str, values: List[float], unit: str = "s"):
    if not values:
        print(f"[WARN] No values for {name}")
        return
    mean_v = statistics.fmean(values)
    p50_v = _percentile(values, 50)
    p95_v = _percentile(values, 95)
    print(f"{name}: mean={mean_v:.4f}{unit}, p50={p50_v:.4f}{unit}, p95={p95_v:.4f}{unit}")


def _load_prompt_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )


def _count_text_tokens(tokenizer, text: str) -> int:
    if tokenizer is None or not text:
        return 0
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return len(token_ids)


def _collect_retrieved_doc_ids(recall_topk: Dict[Any, Any]) -> List[int]:
    doc_ids = set()
    for _, per_layer_entries in (recall_topk or {}).items():
        if not isinstance(per_layer_entries, list):
            continue
        for entry in per_layer_entries:
            if isinstance(entry, int):
                if entry >= 0:
                    doc_ids.add(entry)
                continue
            if not isinstance(entry, dict):
                continue
            topk_doc_ids = entry.get("topk_doc_ids", [])
            if not isinstance(topk_doc_ids, list):
                continue
            for doc_id in topk_doc_ids:
                if isinstance(doc_id, int) and doc_id >= 0:
                    doc_ids.add(doc_id)
    return sorted(doc_ids)


def build_memory_cache_root(args, dataset_path: str, model_config: ModelConfig) -> str | None:
    if args.disable_memory_cache or not args.memory_cache_dir:
        return None

    dataset_abs = os.path.abspath(dataset_path)
    cache_descriptor = {
        "dataset_path": dataset_abs,
        "model_path": os.path.abspath(args.model_path),
        "block_size": args.block_size,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_descriptor, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = os.path.join(os.path.abspath(args.memory_cache_dir), f"msa_mem_{cache_key}")
    os.makedirs(cache_root, exist_ok=True)
    print(f"Memory cache key descriptor: {cache_descriptor}")
    return cache_root


def _build_memory_docs_from_dataset_input(
    dataset: List[Any],
    memory_rows: Optional[int],
) -> Tuple[List[str], int, str]:
    rows = dataset if memory_rows is None else dataset[:memory_rows]
    rows_used = len(rows)

    if not rows:
        return [], 0, "empty"

    first_item = rows[0]
    if isinstance(first_item, str):
        docs = []
        seen = set()
        for doc in rows:
            text = str(doc).strip()
            if text and text not in seen:
                seen.add(text)
                docs.append(text)
        return docs, rows_used, "memory_docs"

    docs = build_memory_docs(rows, None)
    return docs, rows_used, "samples"


def run_msa(
    args,
    dataset: List[Any],
    queryset: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    memory_docs, memory_rows_used, dataset_mode = _build_memory_docs_from_dataset_input(
        dataset, args.memory_docs
    )
    memory_doc_titles = [extract_doc_title(doc) for doc in memory_docs]
    query_source = queryset if queryset is not None else dataset
    query_samples = build_query_samples(query_source, args.n_samples, seed=args.seed)
    # query_samples = build_query_samples(dataset, args.n_samples, start_idx=args.memory_docs)
    # print(f"Memory docs from {0} to {args.memory_docs - 1}, Query samples from {args.memory_docs} to {args.memory_docs + args.n_samples - 1}")

    
    if not memory_docs:
        raise ValueError("No memory docs were built from the dataset.")
    if not query_samples:
        raise ValueError("No query samples were built from the dataset.")

    print(f"Dataset type: {args.dataset_type}")
    print(f"Dataset mode: {dataset_mode}")
    print(f"Loaded {len(dataset)} rows from dataset")
    if queryset is not None:
        print(f"Loaded {len(queryset)} rows from queryset")
    print(f"Memory rows used: {memory_rows_used}")
    print(f"Query rows used: {len(query_samples)}")
    print(f"Built {len(memory_docs)} unique memory docs")

    msa_config = read_msa_config(args.model_path)
    doc_top_k = args.doc_top_k if args.doc_top_k is not None else msa_config.get("doc_top_k", 16)
    pooling_kernel_size = (
        args.pooling_kernel_size
        if args.pooling_kernel_size is not None
        else msa_config.get("pooling_kernel_size", 64)
    )
    router_layer_idx = msa_config.get("router_layer_idx", "all")
    print(f"Pooling kernel size: {pooling_kernel_size}")

    model_config = ModelConfig(
        model_path=args.model_path,
        doc_top_k=doc_top_k,
        pooling_kernel_size=pooling_kernel_size,
        router_layer_idx=router_layer_idx,
    )
    generate_config = GenerateConfig(
        devices=list(range(torch.cuda.device_count())),
        template=args.template,
        max_generate_tokens=args.max_length,
        max_seq_len=args.max_seq_len,
        max_query_seq_len=args.max_query_seq_len,
        max_batch_size=args.max_batch_size,
        top_p=args.top_p,
        temperature=args.temperature,
        qa_mode=True,
    )
    memory_config = MemoryConfig(
        block_size=args.block_size,
        slice_chunk_size=args.max_chunk_per_block,
        pooling_kernel_size=pooling_kernel_size,
        memory_file_path="",
    )

    predictions: List[str] = []
    answers: List[str] = []
    stat_question_latency: List[float] = []
    stat_answer_ttft: List[float] = []
    stat_answer_decode_time: List[float] = []
    stat_answer_token_count: List[int] = []
    stat_answer_tpot: List[float] = []
    stat_retrieval_round_total: List[float] = []
    stat_prefill_before_doc_query: List[float] = []
    stat_doc_query_time: List[float] = []
    stat_retrieval_tail_time: List[float] = []
    stat_answer_prefill: List[float] = []
    stat_final_round_prefill: List[float] = []
    stat_final_round_answer_ttft: List[float] = []
    stat_final_round_routing: List[float] = []
    stat_generate_count: List[int] = []
    stat_total_input_tokens: List[int] = []
    stat_total_output_tokens: List[int] = []
    stat_retrieval_recall: List[float] = []
    stat_retrieval_hit: List[float] = []
    stat_retrieval_all_hit: List[float] = []
    prompt_tokenizer = _load_prompt_tokenizer(args.model_path)

    sorted_requests = sort_requests(query_samples)
    requests = OrderedDict((idx, item) for idx, item in enumerate(sorted_requests))
    world = generate_config.world
    if world <= 0:
        raise RuntimeError("MSA requires at least one visible CUDA device.")
    total_batch = max(1, args.max_batch_size * max(1, world))
    for req_idx in requests:
        requests[req_idx]["idx"] = req_idx
        requests[req_idx]["latency"] = 0.0
        requests[req_idx]["generate_count"] = 0
        requests[req_idx]["input_tokens_total"] = 0
        requests[req_idx]["output_tokens_total"] = 0
        requests[req_idx]["elapsed_before_answer_round"] = 0.0
        requests[req_idx]["answer_ttft"] = None
        requests[req_idx]["answer_decode_time"] = 0.0
        requests[req_idx]["answer_token_count"] = 0
        requests[req_idx]["retrieval_round_total"] = 0.0
        requests[req_idx]["prefill_before_doc_query"] = None
        requests[req_idx]["doc_query_time"] = 0.0
        requests[req_idx]["answer_prefill"] = None
        requests[req_idx]["final_round_prefill"] = None
        requests[req_idx]["final_round_answer_ttft"] = None
        requests[req_idx]["final_round_prefill_before_doc_query"] = None
        requests[req_idx]["final_round_doc_query_time"] = 0.0

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as fp:
        pickle.dump(memory_docs, fp)
        memory_path = fp.name

    try:
        memory_config.memory_file_path = memory_path
        memory_cache_root = build_memory_cache_root(args, args.dataset_path, model_config)
        previous_memory_cache = os.environ.get("MEMORY_DATA_PATH")
        if memory_cache_root:
            print(f"Memory cache directory: {memory_cache_root}")
            os.environ["MEMORY_DATA_PATH"] = memory_cache_root

        with MSAEngine(generate_config, model_config, memory_config) as engine:
            results: List[Dict[str, Any]] = []
            to_send: Dict[int, Dict[str, Any]] = {}
            pbar = tqdm(total=len(requests), desc="MSA inference")

            while len(results) < len(query_samples):
                num = min(total_batch - len(to_send), len(requests))
                for _ in range(num):
                    idx, request = requests.popitem(last=False)
                    to_send[idx] = request

                prompts, indices = [], []
                for idx, request in to_send.items():
                    prompt_text = request.get("new_question", request["question"])
                    request["input_tokens_total"] = request.get("input_tokens_total", 0) + _count_text_tokens(
                        prompt_tokenizer, prompt_text
                    )
                    prompts.append(prompt_text)
                    indices.append(idx)

                batch_start = time.perf_counter()
                texts, recall_topks, round_metrics, _ = engine.generate(prompts, require_recall_topk=True)
                # print(f"first generate\n input: {prompts}\n output: {texts}")
                batch_elapsed = time.perf_counter() - batch_start

                first_answer_latencies = (round_metrics or {}).get("first_answer_token_latency_s", [])
                answer_token_counts = (round_metrics or {}).get("answer_token_count", [])
                prefill_before_doc_query = (round_metrics or {}).get("prefill_before_doc_query_s", [])
                doc_query_times = (round_metrics or {}).get("doc_query_time_s", [])
                is_regenerate_rounds = (round_metrics or {}).get("is_regenerate_round", [])
                for req_idx in indices:
                    to_send[req_idx]["generate_count"] = to_send[req_idx].get("generate_count", 0) + 1
                finished_this_round = 0
                for idx, response in enumerate(texts):
                    req_idx = indices[idx]
                    request = to_send[req_idx]
                    request["latency"] = request.get("latency", 0.0) + batch_elapsed
                    request["output_tokens_total"] = request.get("output_tokens_total", 0) + _count_text_tokens(
                        prompt_tokenizer, response
                    )
                    answer_first_token_latency = (
                        first_answer_latencies[idx] if idx < len(first_answer_latencies) else None
                    )
                    if answer_first_token_latency is not None and request.get("answer_ttft") is None:
                        request["answer_prefill"] = answer_first_token_latency
                    answer_token_count = answer_token_counts[idx] if idx < len(answer_token_counts) else None
                    if answer_token_count is not None:
                        request["answer_token_count"] = request.get("answer_token_count", 0) + int(answer_token_count)
                    router_prefill_time = (
                        prefill_before_doc_query[idx] if idx < len(prefill_before_doc_query) else None
                    )
                    if router_prefill_time is not None and request.get("prefill_before_doc_query") is None:
                        request["prefill_before_doc_query"] = router_prefill_time
                    doc_query_time = doc_query_times[idx] if idx < len(doc_query_times) else None
                    if doc_query_time is not None:
                        request["doc_query_time"] = request.get("doc_query_time", 0.0) + float(doc_query_time)
                    is_regenerate_round = (
                        is_regenerate_rounds[idx] if idx < len(is_regenerate_rounds) else False
                    )
                    cleaned = response.replace("<|endoftext|>", "")
                    split_token = "\nPlease answer the question based"
                    if split_token in cleaned:
                        cleaned = split_token + cleaned.split(split_token, 1)[1]
                    new_prompt = should_regenerate(request, cleaned)
                    # print(f"following generate\n input: {request}\n output: {new_prompt}")
                    if new_prompt is not None:
                        request["retrieval_round_total"] = request.get("retrieval_round_total", 0.0) + batch_elapsed
                        request["elapsed_before_answer_round"] = (
                            request.get("elapsed_before_answer_round", 0.0) + batch_elapsed
                        )
                        request["new_question"] = new_prompt
                    else:
                        recall_topk = {}
                        if recall_topks:
                            recall_topk = {layer: v[idx] for layer, v in recall_topks.items()}
                        request = to_send.pop(req_idx)
                        request["recall_topk"] = recall_topk
                        request["response"] = cleaned
                        request["pred_answer"] = parse_pred_answer(cleaned)
                        retrieval_round_total = request.get("retrieval_round_total", 0.0)
                        answer_prefill = request.get("answer_prefill")
                        final_round_answer_ttft = None
                        final_round_routing = 0.0
                        if is_regenerate_round and answer_first_token_latency is not None:
                            request["final_round_answer_ttft"] = answer_first_token_latency
                            request["final_round_prefill_before_doc_query"] = (
                                float(router_prefill_time) if router_prefill_time is not None else None
                            )
                            request["final_round_doc_query_time"] = (
                                float(doc_query_time) if doc_query_time is not None else 0.0
                            )
                            final_round_answer_ttft = request["final_round_answer_ttft"]
                            final_round_routing = (
                                (request["final_round_prefill_before_doc_query"] or 0.0)
                                + request.get("final_round_doc_query_time", 0.0)
                            )
                            request["final_round_prefill"] = max(
                                0.0,
                                final_round_answer_ttft - final_round_routing,
                            )
                        if answer_prefill is not None:
                            request["answer_ttft"] = retrieval_round_total + answer_prefill
                            request["answer_decode_time"] = max(
                                0.0,
                                request.get("latency", 0.0) - request["answer_ttft"],
                            )
                        else:
                            request["answer_ttft"] = None
                            request["answer_decode_time"] = 0.0
                        results.append(request)
                        stat_question_latency.append(request.get("latency", 0.0))
                        if retrieval_round_total > 0.0:
                            stat_retrieval_round_total.append(request["retrieval_round_total"])
                        if request.get("prefill_before_doc_query") is not None:
                            stat_prefill_before_doc_query.append(request["prefill_before_doc_query"])
                        if request.get("doc_query_time", 0.0) > 0.0:
                            stat_doc_query_time.append(request["doc_query_time"])
                        if (
                            request.get("retrieval_round_total", 0.0) > 0.0
                            and request.get("prefill_before_doc_query") is not None
                        ):
                            retrieval_tail_time = (
                                request["retrieval_round_total"]
                                - request["prefill_before_doc_query"]
                                - request.get("doc_query_time", 0.0)
                            )
                            stat_retrieval_tail_time.append(max(0.0, retrieval_tail_time))
                        if request.get("answer_ttft") is not None:
                            stat_answer_ttft.append(request["answer_ttft"])
                        if request.get("answer_prefill") is not None:
                            stat_answer_prefill.append(request["answer_prefill"])
                        if request.get("final_round_answer_ttft") is not None:
                            stat_final_round_answer_ttft.append(request["final_round_answer_ttft"])
                        if request.get("final_round_prefill_before_doc_query") is not None or request.get(
                            "final_round_doc_query_time", 0.0
                        ) > 0.0:
                            stat_final_round_routing.append(
                                (request.get("final_round_prefill_before_doc_query") or 0.0)
                                + request.get("final_round_doc_query_time", 0.0)
                            )
                        if request.get("final_round_prefill") is not None:
                            stat_final_round_prefill.append(request["final_round_prefill"])
                        if request.get("answer_decode_time", 0.0) > 0.0:
                            stat_answer_decode_time.append(request["answer_decode_time"])
                        if request.get("answer_token_count", 0) > 0:
                            stat_answer_token_count.append(request["answer_token_count"])
                            if request.get("answer_decode_time", 0.0) > 0.0:
                                stat_answer_tpot.append(
                                    request["answer_decode_time"] / max(1, request["answer_token_count"])
                                )
                        stat_generate_count.append(request.get("generate_count", 0))
                        stat_total_input_tokens.append(request.get("input_tokens_total", 0))
                        stat_total_output_tokens.append(request.get("output_tokens_total", 0))
                        finished_this_round += 1

                pbar.update(finished_this_round)

            pbar.close()

        # Restore the original query order for evaluation/export.
        results.sort(key=lambda item: item["idx"])
        predictions = [item["pred_answer"] for item in results]
        answers = [item["answer"] for item in results]
        for item in results:
            retrieved_doc_ids = _collect_retrieved_doc_ids(item.get("recall_topk", {}))
            retrieved_titles = [
                memory_doc_titles[doc_id]
                for doc_id in retrieved_doc_ids
                if 0 <= doc_id < len(memory_doc_titles)
            ]
            recall_stats = compute_retrieval_recall_stats(
                item.get("supporting_titles", []),
                retrieved_titles,
            )
            stat_retrieval_recall.append(recall_stats["recall"])
            stat_retrieval_hit.append(recall_stats["hit"])
            stat_retrieval_all_hit.append(recall_stats["all_hit"])
        sample_ids = []
        for item in results:
            sample_id = item.get("id")
            if sample_id in (None, ""):
                sample_id = str(item["idx"])
            sample_ids.append(str(sample_id))

    finally:
        if previous_memory_cache is None:
            os.environ.pop("MEMORY_DATA_PATH", None)
        else:
            os.environ["MEMORY_DATA_PATH"] = previous_memory_cache
        if os.path.exists(memory_path):
            os.remove(memory_path)

    print("\n========== MSA Metrics ==========")
    print("[Overall]")
    summarize_latency("Per-question answer time", stat_question_latency, "s")
    if stat_question_latency and sum(stat_question_latency) > 0:
        print(f"Throughput (QPS): {len(stat_question_latency) / sum(stat_question_latency):.2f}")

    print("\n[Retrieval Phase]")
    if stat_retrieval_round_total:
        summarize_latency("Retrieval round total", stat_retrieval_round_total, "s")
    else:
        print("[WARN] No retrieval-round totals were captured.")
    if stat_prefill_before_doc_query:
        summarize_latency("Router prefill before doc_query", stat_prefill_before_doc_query, "s")
    else:
        print("[WARN] No router-prefill timings were captured.")
    if stat_doc_query_time:
        summarize_latency("doc_query retrieval core", stat_doc_query_time, "s")
    else:
        print("[WARN] No doc_query timings were captured.")
    if stat_retrieval_tail_time:
        summarize_latency("Retrieval round tail (ID generation/context copy)", stat_retrieval_tail_time, "s")
    else:
        print("[WARN] No retrieval-tail timings were captured.")

    print("\n[Answer Phase]")
    if stat_answer_ttft:
        summarize_latency("Answer TTFT (= retrieval round total + answer prefill)", stat_answer_ttft, "s")
    else:
        print("[WARN] No Answer TTFT values were captured.")
    if stat_answer_prefill:
        summarize_latency("Answer prefill to first token", stat_answer_prefill, "s")
    else:
        print("[WARN] No answer-prefill timings were captured.")
    if stat_final_round_answer_ttft:
        summarize_latency("Final-round answer TTFT", stat_final_round_answer_ttft, "s")
    else:
        print("[WARN] No final-round answer TTFT values were captured.")
    if stat_final_round_routing:
        summarize_latency("Final-round routing", stat_final_round_routing, "s")
    else:
        print("[WARN] No final-round routing timings were captured.")
    if stat_final_round_prefill:
        summarize_latency("Final-round prefill (= final-round answer TTFT - final-round routing)", stat_final_round_prefill, "s")
    else:
        print("[WARN] No final-round-prefill timings were captured.")
    if stat_answer_decode_time:
        summarize_latency("Answer decode time", stat_answer_decode_time, "s")
    else:
        print("[WARN] No answer decode time values were captured.")
    summarize_latency("Answer token count", stat_answer_token_count, "")
    if stat_answer_tpot:
        summarize_latency("Answer TPOT", stat_answer_tpot, "s/token")
    else:
        print("[WARN] No answer TPOT values were captured.")
    if stat_generate_count:
        print(f"Average generate count per request: {statistics.fmean(stat_generate_count):.4f}")

    print("\n[Workload]")
    summarize_latency("Total input tokens per request", stat_total_input_tokens, "")
    summarize_latency("Total output tokens per request", stat_total_output_tokens, "")

    print("\n[Retrieval Quality]")
    summarize_latency("Retrieval recall", stat_retrieval_recall, "")
    if stat_retrieval_hit:
        print(f"Retrieval hit@{doc_top_k}: mean={statistics.fmean(stat_retrieval_hit):.4f}")
    if stat_retrieval_all_hit:
        print(f"Retrieval all-support-hit@{doc_top_k}: mean={statistics.fmean(stat_retrieval_all_hit):.4f}")
    print("=================================\n")

    return predictions, answers, sample_ids


def main():
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    dataset = load_dataset(args.dataset_path)
    if args.dataset_limit is not None:
        if args.dataset_limit <= 0:
            raise ValueError("--dataset-limit must be a positive integer when set.")
        original_len = len(dataset)
        dataset = dataset[: args.dataset_limit]
        print(f"Truncated dataset to first {len(dataset)} rows (original {original_len})")
    queryset = load_queryset(args.queryset_path)
    filtered_queryset = queryset

    if str(args.dataset_type).lower() == "mintqa" and args.mintqa_min_hop is not None:
        query_source = queryset if queryset is not None else dataset
        original_len = len(query_source)
        filtered_queryset = [
            sample
            for sample in query_source
            if sample.get("metadata", {}).get("support_hops", -1) >= args.mintqa_min_hop
        ]
        print(
            f"Filtering to {len(filtered_queryset)} samples with hop >= {args.mintqa_min_hop} "
            f"(original {original_len})"
        )

    predictions, answers, sample_ids = run_msa(args, dataset, queryset=filtered_queryset)
    query_source = filtered_queryset if filtered_queryset is not None else dataset
    question_by_sample_id = {
        get_sample_id(row, i): (get_question(row) or "")
        for i, row in enumerate(query_source)
    }
    questions = [question_by_sample_id.get(sample_id, "") for sample_id in sample_ids]

    if args.dis_out_path is not None:
        metrics, faith_01_scores = evaluate_model_outputs(
            predictions,
            answers,
            questions=questions,
        )
        num_scored = min(len(predictions), len(faith_01_scores))
        if len(faith_01_scores) != len(predictions):
            print(
                f"Warning: got {len(faith_01_scores)} faithfulness scores for "
                f"{len(predictions)} predictions; only scored samples will be exported."
            )

        evaluated_samples_ids = []
        for i in range(num_scored):
            if faith_01_scores[i] == 0:
                evaluated_samples_ids.append(sample_ids[i])
        with open(args.dis_out_path, "w", encoding="utf-8") as f:
            json.dump({"evaluated_samples": evaluated_samples_ids}, f, ensure_ascii=False, indent=2)
        print(f"✅ Evaluation {len(evaluated_samples_ids)} results saved to {args.dis_out_path}")
    else:
        _, metrics = full_evaluation(predictions, answers, questions=questions)
        print(metrics)


if __name__ == "__main__":
    main()
