import argparse
import gc
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kblam.metrics_evaluator import evaluate_model_outputs
from kblam.utils.eval_utils import format_output_for_synthetic


SPECIAL_TOKEN_THRESHOLD = 128000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation inference for DAG-KV experiments. Currently supports DAG-as-Text."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dag_as_text",
        choices=["dag_as_text"],
        help="Ablation method to run.",
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to DAG-KV json/jsonl dataset.")
    parser.add_argument("--model-path", type=str, required=True, help="Base LLM path for text prompting.")
    parser.add_argument("--query-size", type=int, default=100, help="Number of samples to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed. 0 means take the first N samples.")
    parser.add_argument("--dag-kb-size", type=int, default=1, help="How many DAG samples to serialize into the prompt.")
    parser.add_argument("--max-kv-per-dag", type=int, default=None, help="Optional cap on serialized KV nodes per DAG.")
    parser.add_argument("--max-output-len", type=int, default=16, help="Maximum generated tokens.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max model length.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Optional tensor parallel size. Defaults to CUDA device count when needed.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional summary JSON output path.",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=None,
        help="Optional per-sample predictions JSONL output path.",
    )
    parser.add_argument(
        "--print-first-prompt",
        action="store_true",
        help="Print the first constructed prompt for debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample question/answer/prediction.",
    )
    return parser.parse_args()


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
        return data
    raise ValueError(f"Unsupported dataset format: {path}")


def select_dataset_rows(dataset: Sequence[Dict[str, Any]], query_size: int, seed: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    if query_size <= 0:
        raise ValueError("--query-size must be positive.")
    query_size = min(query_size, len(dataset))
    if seed == 0:
        indices = list(range(query_size))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset)), query_size))
    return [dataset[i] for i in indices], indices


def _load_prompt_tokenizer(model_path: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    return tokenizer


def _load_model_config(model_path: str):
    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def _get_model_type(model_path: str) -> str:
    config = _load_model_config(model_path)
    return str(getattr(config, "model_type", "") or "").lower()


def _resolve_tensor_parallel_size(args: argparse.Namespace) -> int:
    if args.tensor_parallel_size and args.tensor_parallel_size > 0:
        return args.tensor_parallel_size
    return max(1, torch.cuda.device_count())


def setup_llm(args: argparse.Namespace) -> Tuple[LLM, AutoTokenizer, str]:
    print(f"Loading model from {args.model_path}")
    prompt_tokenizer = _load_prompt_tokenizer(args.model_path)
    model_type = _get_model_type(args.model_path)
    is_qwen_family = model_type in {"qwen3", "qwen3_moe"} or "qwen" in args.model_path.lower()

    base_kwargs = {
        "model": args.model_path,
        "enforce_eager": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
    }

    if "llama" in args.model_path.lower():
        llm = LLM(**base_kwargs)
    elif is_qwen_family:
        llm = LLM(
            **base_kwargs,
            dtype="bfloat16",
            tensor_parallel_size=_resolve_tensor_parallel_size(args),
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
        )
    elif "deepseek" in args.model_path.lower():
        llm = LLM(
            **base_kwargs,
            dtype="bfloat16",
            tensor_parallel_size=_resolve_tensor_parallel_size(args),
            gpu_memory_utilization=min(args.gpu_memory_utilization, 0.8),
            trust_remote_code=True,
            max_model_len=max(args.max_model_len, 16384),
        )
    elif "olmo3-7b" in args.model_path.lower():
        llm = LLM(
            **base_kwargs,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
        )
    elif "olmo3-32b" in args.model_path.lower():
        llm = LLM(
            **base_kwargs,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
        )
    else:
        llm = LLM(
            **base_kwargs,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    return llm, prompt_tokenizer, model_type


def get_question(row: Dict[str, Any]) -> str:
    return str(row.get("Q", row.get("question", ""))).strip()


def get_answer(row: Dict[str, Any]) -> str:
    return str(row.get("A", row.get("answer", ""))).strip()


def _format_triple_item(triple: Dict[str, Any], idx: int) -> str:
    triple_type = str(triple.get("type", "")).strip()
    name = str(triple.get("name", "")).strip()
    relation = str(triple.get("description_type", "")).strip()
    description = str(triple.get("description", "")).strip()
    context_title = str(triple.get("context_title", "")).strip()
    kv_lists = triple.get("kv_lists") or []

    parts = [f"Triple {idx}:"]
    if context_title:
        parts.append(f"[title={context_title}]")
    if name and relation and description:
        parts.append(f"{name} | {relation} | {description}")
    elif name or relation or description:
        parts.append(" | ".join(part for part in [name, relation, description] if part))

    if triple_type:
        parts.append(f"(type={triple_type})")

    if kv_lists:
        kv = kv_lists[0]
        key_string = str(kv.get("key_string", "")).strip()
        value_string = str(kv.get("value_string", "")).strip()
        if key_string or value_string:
            parts.append(f"[{key_string} -> {value_string}]")

    return " ".join(part for part in parts if part).strip()


def _extract_sample_triples(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    top_level = row.get("triple_list") or row.get("triple_lists") or row.get("triples")
    if isinstance(top_level, list):
        return [triple for triple in top_level if isinstance(triple, dict)]

    context_items = row.get("context")
    if not isinstance(context_items, list):
        context_items = row.get("paragraphs")
    if not isinstance(context_items, list):
        return []

    triples: List[Dict[str, Any]] = []
    for ctx in context_items:
        if not isinstance(ctx, dict):
            continue
        ctx_title = str(ctx.get("title", "")).strip()
        ctx_triples = ctx.get("triple_list") or ctx.get("triple_lists") or ctx.get("triples") or []
        if not isinstance(ctx_triples, list):
            continue
        for triple in ctx_triples:
            if not isinstance(triple, dict):
                continue
            triple_copy = dict(triple)
            if ctx_title and "context_title" not in triple_copy:
                triple_copy["context_title"] = ctx_title
            triples.append(triple_copy)
    return triples


def build_triple_text(row: Dict[str, Any], group_label: str, max_kv_per_dag: Optional[int] = None) -> str:
    triple_list = _extract_sample_triples(row)
    if max_kv_per_dag is not None:
        triple_list = triple_list[:max_kv_per_dag]

    triple_lines = [
        _format_triple_item(triple, idx)
        for idx, triple in enumerate(triple_list, start=1)
        if isinstance(triple, dict)
    ]

    sections = [group_label]
    if triple_lines:
        sections.append("Triples:")
        sections.extend(triple_lines)
    return "\n".join(sections).strip()


def build_dag_as_text_context(
    dataset: Sequence[Dict[str, Any]],
    target_index: int,
    dag_kb_size: int,
    max_kv_per_dag: Optional[int],
    seed: int,
) -> str:
    rng = random.Random(seed + int(target_index))
    sample_ids = [int(target_index)]
    if dag_kb_size > 1:
        distractor_pool = [i for i in range(len(dataset)) if i != int(target_index)]
        rng.shuffle(distractor_pool)
        sample_ids.extend(distractor_pool[: max(0, dag_kb_size - 1)])
    rng.shuffle(sample_ids)

    graphs = []
    for order, sample_id in enumerate(sample_ids, start=1):
        graph_label = f"Candidate {order} (sample_id={sample_id})"
        graphs.append(build_triple_text(dataset[sample_id], graph_label, max_kv_per_dag=max_kv_per_dag))
    return "\n\n".join(graph for graph in graphs if graph.strip())


def _build_qwen_prompt(tokenizer: AutoTokenizer, question: str, context: str) -> str:
    system_prompt = (
        "You are a question-answering system.\n"
        "Use only the given triples to answer the question.\n"
        "Return exactly one short final answer.\n"
        "Do not explain. Do not write a full sentence.\n"
        "If multiple candidate groups are provided, use the triples that best support the question.\n"
    )
    user_prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Final answer:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt(model_path: str, tokenizer: AutoTokenizer, question: str, context: str) -> str:
    lower_model_path = model_path.lower()
    if "llama" in lower_model_path:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You answer questions with ONLY the exact answer phrase. "
            "Use only the given triples. "
            "Never add explanations, prefixes, or punctuation."
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer exactly:"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    if "deepseek" in lower_model_path:
        return (
            "<|system|>\n"
            "Answer the question using ONLY the given triples.\n"
            "Output ONLY the exact answer phrase in English.\n"
            "Do NOT add explanations, prefixes, suffixes, or punctuation.\n"
            "<|user|>\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:\n<|assistant|>\n"
        )
    if "qwen" in lower_model_path:
        return _build_qwen_prompt(tokenizer, question, context)
    if "olmo3" in lower_model_path:
        return (
            "<|system|>\n"
            "Answer the question using ONLY the given triples. "
            "Output ONLY the final answer phrase.\n"
            "<|user|>\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:\n<|assistant|>\n"
        )
    return (
        "Answer the question using ONLY the given triples.\n"
        "Output ONLY the final answer phrase.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:\n"
    )


def _count_non_special_token_ids(token_ids: List[int]) -> int:
    return sum(1 for tid in token_ids if isinstance(tid, int) and tid < SPECIAL_TOKEN_THRESHOLD)


def _count_prompt_input_tokens(tokenizer: AutoTokenizer, prompt_text: str) -> int:
    try:
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(prompt_text)
    non_special = _count_non_special_token_ids(token_ids)
    return non_special if non_special > 0 else len(token_ids)


def _extract_num_output_tokens(req_out) -> int:
    outputs = getattr(req_out, "outputs", None) or []
    total = 0
    for output in outputs:
        token_ids = getattr(output, "token_ids", None) or []
        if not token_ids:
            continue
        non_special = _count_non_special_token_ids(token_ids)
        total += non_special if non_special > 0 else len(token_ids)
    return total


def _extract_latency_from_request_output(req_out) -> Dict[str, Any]:
    out = {
        "ttft_model_ts": None,
        "prefill_model": None,
        "decode_time": None,
        "num_output_tokens": 0,
        "ttft_exact": False,
        "decode_exact": False,
    }

    first_token_time = getattr(req_out, "first_token_time", None)
    if first_token_time is not None:
        out["ttft_model_ts"] = float(first_token_time)
        out["ttft_exact"] = True

    out["num_output_tokens"] = _extract_num_output_tokens(req_out)

    metrics = getattr(req_out, "metrics", None)
    if metrics is None:
        return out

    if hasattr(metrics, "first_token_ts"):
        if out["ttft_model_ts"] is None:
            out["ttft_model_ts"] = float(metrics.first_token_ts)
            out["ttft_exact"] = True

        if hasattr(metrics, "first_token_latency"):
            out["prefill_model"] = float(metrics.first_token_latency)
        elif hasattr(metrics, "scheduled_ts"):
            out["prefill_model"] = float(metrics.first_token_ts - metrics.scheduled_ts)

        if hasattr(metrics, "last_token_ts"):
            out["decode_time"] = max(0.0, float(metrics.last_token_ts - metrics.first_token_ts))
            out["decode_exact"] = True

    if hasattr(metrics, "num_generation_tokens"):
        out["num_output_tokens"] = int(metrics.num_generation_tokens)

    return out


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    rank = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[rank]


def summarize_metric(name: str, values: List[float], unit: str) -> Dict[str, float]:
    if not values:
        print(f"[WARN] No values for {name}")
        return {}
    summary = {
        "mean": float(statistics.fmean(values)),
        "p50": float(_percentile(values, 50)),
        "p95": float(_percentile(values, 95)),
    }
    print(
        f"{name}: mean={summary['mean']:.4f} {unit}, "
        f"p50={summary['p50']:.4f} {unit}, p95={summary['p95']:.4f} {unit}"
    )
    return summary


def postprocess_prediction(pred: str) -> str:
    text = str(pred).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0].strip()
    if "<|end|>" in text:
        text = text.split("<|end|>")[0].strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return format_output_for_synthetic(text)


def run_dag_as_text(args: argparse.Namespace, dataset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    llm, tokenizer, _ = setup_llm(args)
    sampled_rows, sampled_indices = select_dataset_rows(dataset, args.query_size, args.seed)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_output_len,
        stop=["<|im_end|>", "<|end|>", "<|endoftext|>"],
    )

    predictions: List[str] = []
    answers: List[str] = []
    questions: List[str] = []
    samples_out: List[Dict[str, Any]] = []

    stat_ttft: List[float] = []
    stat_prefill: List[float] = []
    stat_tpot: List[float] = []
    stat_e2e: List[float] = []
    stat_input_tokens: List[float] = []
    stat_output_tokens: List[float] = []
    any_exact_ttft = False
    any_estimated_ttft = False
    any_exact_decode = False
    any_estimated_decode = False

    print(f"Running DAG-as-Text on {len(sampled_rows)} samples")
    for local_idx, (row, global_idx) in enumerate(zip(sampled_rows, sampled_indices)):
        question = get_question(row)
        answer = get_answer(row)
        context = build_dag_as_text_context(
            dataset=dataset,
            target_index=global_idx,
            dag_kb_size=args.dag_kb_size,
            max_kv_per_dag=args.max_kv_per_dag,
            seed=args.seed,
        )
        prompt = build_prompt(args.model_path, tokenizer, question, context)
        if args.print_first_prompt and local_idx == 0:
            print("========== First Prompt ==========")
            print(prompt)
            print("==================================")

        gen_start = time.perf_counter()
        req = llm.generate(prompt, sampling_params=sampling_params)[0]
        gen_elapsed = time.perf_counter() - gen_start

        raw_pred = str(req.outputs[0].text).strip()
        pred = postprocess_prediction(raw_pred)
        gt = format_output_for_synthetic(answer)

        metrics = _extract_latency_from_request_output(req)
        num_out = int(metrics.get("num_output_tokens") or 0)
        num_in = _count_prompt_input_tokens(tokenizer, prompt)
        ttft_model_ts = metrics.get("ttft_model_ts")
        prefill_model = metrics.get("prefill_model")
        decode_time = metrics.get("decode_time")

        ttft = None
        if ttft_model_ts is not None:
            ttft = max(0.0, float(ttft_model_ts) - gen_start)
            any_exact_ttft = True
        else:
            ttft = max(0.0, float(prefill_model) if prefill_model is not None else gen_elapsed)
            any_estimated_ttft = True

        if prefill_model is None:
            prefill_model = ttft

        if decode_time is None:
            decode_time = max(0.0, gen_elapsed - ttft)
            any_estimated_decode = True
        else:
            any_exact_decode = True

        tpot = decode_time / max(1, num_out)

        predictions.append(pred)
        answers.append(gt)
        questions.append(question)
        stat_ttft.append(float(ttft))
        stat_prefill.append(float(prefill_model))
        stat_tpot.append(float(tpot))
        stat_e2e.append(float(gen_elapsed))
        stat_input_tokens.append(float(num_in))
        stat_output_tokens.append(float(num_out))

        sample_out = {
            "sample_index": int(global_idx),
            "id": row.get("_id", row.get("id", global_idx)),
            "question": question,
            "answer": gt,
            "prediction": pred,
            "ttft_s": float(ttft),
            "prefill_s": float(prefill_model),
            "tpot_s": float(tpot),
            "e2e_s": float(gen_elapsed),
            "input_tokens": int(num_in),
            "output_tokens": int(num_out),
        }
        samples_out.append(sample_out)

        if args.verbose:
            print("------------------------------")
            print(f"[{local_idx}] sample={global_idx}")
            print(f"Q: {question}")
            print(f"GT: {gt}")
            print(f"PRED: {pred}")
            print(
                f"TTFT={sample_out['ttft_s']:.4f}s, "
                f"TPOT={sample_out['tpot_s']:.4f}s, "
                f"E2E={sample_out['e2e_s']:.4f}s"
            )

    metric_dict, faith_01_scores = evaluate_model_outputs(
        predictions,
        answers,
        questions=questions,
        skip_bert=True,
        skip_faithfulness=True,
    )
    accuracy = float(metric_dict.get("exact_match", 0.0))

    for idx, sample_out in enumerate(samples_out):
        sample_out["faithfulness01_score"] = (
            faith_01_scores[idx] if idx < len(faith_01_scores) else None
        )

    print("\n========== DAG-as-Text Summary ==========")
    print(f"Accuracy (Exact Match): {accuracy:.4f}")
    ttft_summary = summarize_metric("TTFT", stat_ttft, "s")
    prefill_summary = summarize_metric("Prefill", stat_prefill, "s")
    tpot_summary = summarize_metric("TPOT", stat_tpot, "s/token")
    e2e_summary = summarize_metric("E2E", stat_e2e, "s")
    input_summary = summarize_metric("Input tokens", stat_input_tokens, "tokens")
    output_summary = summarize_metric("Output tokens", stat_output_tokens, "tokens")
    if any_exact_ttft and not any_estimated_ttft:
        print("TTFT source: exact request first-token timestamp")
    elif any_exact_ttft:
        print("TTFT source: mixed exact first-token timestamp and fallback estimate")
    else:
        print("TTFT source: fallback estimate")
    if any_exact_decode and not any_estimated_decode:
        print("Decode/TPOT source: exact request decode timestamps")
    elif any_exact_decode:
        print("Decode/TPOT source: mixed exact decode timestamps and end-to-end estimate")
    else:
        print("Decode/TPOT source: end-to-end estimate")
    print("=========================================\n")

    summary = {
        "method": args.method,
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_path": os.path.abspath(args.model_path),
        "num_samples": len(sampled_rows),
        "dag_kb_size": args.dag_kb_size,
        "max_kv_per_dag": args.max_kv_per_dag,
        "accuracy": accuracy,
        "exact_match": accuracy,
        "metrics": metric_dict,
        "faithfulness01_scores": faith_01_scores,
        "latency": {
            "ttft_s": ttft_summary,
            "prefill_s": prefill_summary,
            "tpot_s": tpot_summary,
            "e2e_s": e2e_summary,
            "input_tokens": input_summary,
            "output_tokens": output_summary,
        },
        "samples": samples_out,
    }

    try:
        if hasattr(llm, "llm_engine"):
            llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset = read_json_or_jsonl(args.dataset_path)

    if args.method != "dag_as_text":
        raise ValueError(f"Unsupported method: {args.method}")

    summary = run_dag_as_text(args, dataset)

    if args.output_path:
        write_json(args.output_path, summary)
        print(f"Saved summary to {args.output_path}")
    if args.predictions_path:
        write_jsonl(args.predictions_path, summary["samples"])
        print(f"Saved predictions to {args.predictions_path}")


if __name__ == "__main__":
    main()
