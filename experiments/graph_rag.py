import os
import re
import gc
import json
import time
import argparse
import statistics
from collections import Counter
from typing import Any, Dict, List, Tuple
import random

import numpy as np
import torch

# -------- vLLM --------
from vllm import LLM, SamplingParams

# -------- 你项目中的评测 --------
from kblam.metrics_evaluator import full_evaluation
from kblam.kb_retriever import KBRetriever


def _extract_latency_from_request_output(req_out) -> Dict[str, Any]:
    """
    从 vLLM RequestOutput 中提取：
      prefill_model = first_token_time - first_scheduled_time
      decode_time   = finished_time - first_token_time
      num_output_tokens = 生成 token 数（排除特殊 token）
    """
    out = {"prefill_model": None, "decode_time": None, "num_output_tokens": 0}
    try:
        m = getattr(req_out, "metrics", None)
        if m is not None:
            out["prefill_model"]=getattr(m, "first_token_latency", None)
            ft = getattr(m, "first_token_ts", None)
            fin = getattr(m, "last_token_ts", None)
            if ft is not None and fin is not None:
                out["decode_time"] = max(0.0, float(fin) - float(ft))
    except Exception:
        pass

    try:
        if getattr(req_out, "outputs", None):
            token_ids = getattr(req_out.outputs[0], "token_ids", None)
            if token_ids:
                out["num_output_tokens"] = _count_non_special_token_ids(token_ids)
    except Exception:
        pass

    return out
def _summarize(name: str, values: List[float], unit: str = "s"):
    if not values:
        print(f"[WARN] No values for {name}")
        return
    mean_v = statistics.fmean(values)
    p50_v = _percentile(values, 50)
    p95_v = _percentile(values, 95)
    print(f"{name}: mean={mean_v:.4f}{unit}, p50={p50_v:.4f}{unit}, p95={p95_v:.4f}{unit}")


def _percentile(values: List[float], p: float) -> float:
    """计算百分位数"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = idx - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _count_non_special_token_ids(token_ids: List[int]) -> int:
    """计算非特殊 token 的数量"""
    # vLLM 特殊 token ID（根据不同模型可能不同，这里采用通用排除法）
    # 通常 0-3 是特殊 token
    return sum(1 for tid in token_ids if tid > 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Llama RAG with MuSiQue (TTFT & TPOT)')

    parser.add_argument('--n-samples', type=int, default=10, help='测试样本数量')

    # 模型参数
    parser.add_argument('--model-path', type=str,
                        default='/mnt/n0/models/llama3_8B_instruct/',
                        help='vLLM 模型路径（llama/deepseek/qwen 等）')

    # 检索参数
    parser.add_argument('--similarity-top-k', type=int, default=5, help='相似度检索 Top-K')

    parser.add_argument(
        "--hnsw_index_path",
        type=str,
        default=None,
        help="Path to HNSW index, set to None to disable HNSW retrieval",
    )
    parser.add_argument(
        "--base_embeder_path",
        type=str,
        default=None,
        help="Path to base embeder model",
    )

    parser.add_argument(
        "--precomputed_embed_keys_path", type=str, help="Path to precomputed key embeddings"
    )
    parser.add_argument(
        "--precomputed_embed_values_path",
        type=str,
        help="Path to precomputed value embeddings",
    )
    parser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the dataset"
    )

    parser.add_argument(
        "--test_dataset", type=str, help="Source of test KB (assumes KV pair format)"
    )
    args = parser.parse_args()

    # 加载数据集
    dataset_path=os.path.join(args.dataset_dir, args.test_dataset)
    if dataset_path.endswith(".jsonl"):
        dataset=[json.loads(line.strip()) for line in open(dataset_path)]
    elif dataset_path.endswith(".json"):
        dataset=json.load(open(dataset_path))
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")

    # 准备图检索器
    kb_retriever = KBRetriever(
        None,
        dataset,
        precomputed_embed_keys_path=args.precomputed_embed_keys_path,
        precomputed_embed_values_path=args.precomputed_embed_values_path,
        hnsw_index_path=args.hnsw_index_path,
        base_embeder_path=args.base_embeder_path,
    )

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    if 'llama' in args.model_path:
    # 使用VLLM加载模型
        llm = LLM(
            model=args.model_path,
            enforce_eager=True,
            disable_log_stats=False,
        )
    elif 'deepseek' in args.model_path or 'qwen' in args.model_path:
        n_gpus = torch.cuda.device_count()
        print(f"Detected {n_gpus} GPUs")
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=n_gpus,   
            gpu_memory_utilization=0.8, # save some memory for bert model  
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=16384,
        )
    elif 'olmo3-7b' in args.model_path.lower():
        # === 新增：OLMo-3-7B-Instruct ===
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            trust_remote_code=True,   # 必须
            enforce_eager=True,
            max_model_len=8192,
        )
    elif "olmo3-32b" in args.model_path.lower():
        llm = LLM(
            model=args.model_path,
            # quantization="awq",          # ⭐ 关键
            dtype="float16",             # ⭐ 不要用 bfloat16
            trust_remote_code=True,
            # enforce_eager=True,
            max_model_len=8192,          # 可后续调大
            # gpu_memory_utilization=0.90,
        )

    # 开始推理

    questions = [item["Q"] for item in dataset[:args.n_samples]]
    true_answers=[item["A"] for item in dataset[:args.n_samples]]

    predictions: List[str] = []
    stat_retrieval: List[float] = []
    stat_prefill: List[float] = []
    stat_ttft: List[float] = []
    stat_tpot: List[float] = []
    stat_tokens: List[int] = []
    stat_e2e: List[float] = []

    rerank_policy=2
    ann_topk=args.similarity_top_k
    hop_num=2


    for i, question in enumerate(questions):
        
        # ---- 检索阶段 ----
        # if args.oracle_retrieval:
        #     ctx_list = correct_paragraphs[i] if isinstance(correct_paragraphs[i], list) else [str(correct_paragraphs[i])]
        #     context = "\n\n".join(ctx_list)
        #     retrieval_time = 0.0
        # else:
        #     context, retrieval_time = retrieve_single_hop(question, paras, args.similarity_top_k)
        start_time=time.perf_counter()
        q_emb = kb_retriever.create_query_embeddings([question])[0]
        if rerank_policy == 1:
            idxs = kb_retriever.get_retrieve_idx_v1(q_emb, topk=ann_topk)        # (topk,)
        elif rerank_policy == 2:
            idxs = kb_retriever.get_retrieve_idx_v2(q_emb, topk=ann_topk)        # (topk,)
        elif rerank_policy == 3:
            idxs = kb_retriever.get_retrieve_idx_v3(q_emb, topk=ann_topk)        # (topk,)
        else:
            raise ValueError(f"Unknown rerank_policy: {rerank_policy}")
        idxs = (np.asarray(idxs, dtype=np.int64) // hop_num).tolist()
        # ------ 上下文拼接 ------
        context = ""
        # for idx in idxs:
        #     for triple in dataset[idx]["triple_lists"]:
        #         is_attr=triple["key_string"].lower().startswith("the ") and " of " in triple["key_string"].lower()
        #         if is_attr:
        #             context += triple["key_string"]+" is "+triple["description"]+ "\n\n"
        #         else:
        #             context += triple["key_string"]+" "+triple["description"]+ "\n\n"

        for idx in idxs:
            paragraphs = dataset[idx].get("paragraphs", [])
            if not paragraphs:
                continue
            for para in dataset[idx]["paragraphs"]:
                context += json.dumps(para, ensure_ascii=False) + "\n\n"

        
        retrieval_time=time.perf_counter()-start_time

        # ---- 生成阶段（计时）----
        gen_start = time.perf_counter()

        # Prompt 构造 + vLLM generate
        if "llama" in args.model_path.lower():
            prompt = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"You answer questions with ONLY the exact answer phrase. "
                f"Never add explanations, prefixes, or punctuation. "
                f"Examples:\n"
                f"Question: Who wrote '1984'? → George Orwell\n"
                f"Question: Capital of France? → Paris\n"
                f"Question: When was Einstein born? → 1879\n"
                f"<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer exactly: <|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
            resp = llm.generate(prompt)
            req_out = resp[0]
            pred = str(req_out.outputs[0].text).strip()
            if i==0:
                print(f"====== prompt ======")
                print(prompt)
                print(f"====== response ======")
                print(req_out)
                print(f"==========")

        elif "deepseek" in args.model_path.lower():
            prompt = (
                f"<|system|>\n"
                f"Answer the question using ONLY the given context. "
                f"Output ONLY the exact answer phrase in English. "
                f"Do NOT add any explanations, prefixes, suffixes, or punctuation.\n\n"
                f"--- Examples ---\n"
                f"Question: Who is the spouse of the Green performer? → Miquette Giraudy\n"
                f"Question: Capital of France? → Paris\n"
                f"Question: When was Einstein born? → 1879\n"
                f"--- End Examples ---\n"
                f"<|user|>\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer:\n"
                f"<|assistant|>\n"
            )
            sp = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=1024)
            resp = llm.generate(prompt, sampling_params=sp)
            req_out = resp[0]
            pred = str(req_out.outputs[0].text).strip()
            # 清理
            try:
                parts = pred.split("</think>")
                if len(parts) > 1:
                    pred = parts[1].strip()
            except Exception:
                pass
            if "Answer: " in pred:
                pred = pred.split("Answer: ")[-1].strip()

        elif "qwen" in args.model_path.lower():
            system_prompt = (
                "You answer questions with ONLY the exact answer phrase. "
                "Never add explanations, prefixes, or punctuation. "
                "Examples:\n"
                "Question: Who wrote '1984'? → George Orwell\n"
                "Question: Capital of France? → Paris\n"
                "Question: When was Einstein born? → 1879\n"
            )
            user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"
            prompt = (
                "<|system|>\n" + system_prompt + "\n<|end|>\n"
                "<|user|>\n" + user_prompt + "\n<|end|>\n"
                "<|assistant|>\n"
            )
            resp = llm.generate(prompt)
            req_out = resp[0]
            pred = str(req_out.outputs[0].text).strip()
            if "<|end|>" in pred:
                pred = pred.split("<|end|>")[0].strip()
            if "Explanation: " in pred:
                pred = pred.split("Explanation: ")[0].strip()
            pred = pred.split("\n")[0].strip()
        elif "olmo3-7b" in args.model_path.lower():
            system_prompt = (
                "You are a helpful assistant. "
                "Answer the question using ONLY the given context. "
                "Output ONLY the exact answer phrase. "
                "Do NOT add explanations or extra text."
            )

            user_prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )

            prompt = (
                "<|system|>\n" + system_prompt + "\n"
                "<|user|>\n" + user_prompt + "\n"
                "<|assistant|>\n"
            )

            # sp = SamplingParams(
            #     temperature=0.0,
            #     top_p=1.0,
            #     max_tokens=128,
            # )
            sp=SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=1024,
            )

            resp = llm.generate(prompt, sampling_params=sp)
            req_out = resp[0]
            pred = req_out.outputs[0].text.strip()
        elif "olmo3-32b" in args.model_path.lower():
            # ===== 1. 评测专用硬约束 Prompt =====
            prompt = (
                f"""
                You must answer with ONLY the final answer.
                Rules:
                - Use ONLY the information in the context.
                - Output a single short phrase or name.
                - Do NOT explain.
                - Do NOT add any extra words.
                - Do NOT repeat the question.
                - If the answer is not explicitly stated in the context, output: UNKNOWN.
                Context:
                {context}
                Question:
                {question}
                Answer:
                """
            )

            # ===== 2. 评测专用 SamplingParams =====
            sp = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=16,
                stop=[
                    "\n",
                    "\n\n",
                    "Context:",
                    "Question:",
                    "<|assistant|>",
                    "<|user|>",
                ],
            )

            resp = llm.generate(prompt, sampling_params=sp)
            print(f"==========================\n Prompts: {prompt} \nOutput: {resp}\n==========================")
            req_out = resp[0]
            pred = req_out.outputs[0].text.strip()

        else:
            print("Unknown model path (llama/deepseek/qwen).")
            req_out = None
            pred = ""

        gen_end = time.perf_counter()
        gen_elapsed = gen_end - gen_start

        # ---- 解析 vLLM 指标 & 计算 TTFT/TPOT ----
        metrics = _extract_latency_from_request_output(req_out) if req_out is not None else {}
        prefill_model = metrics.get("prefill_model", None)
        decode_time = metrics.get("decode_time", None)
        num_out = metrics.get("num_output_tokens", 0)

        # 回退策略：拿不到精确 prefill/decode 时，用 gen_elapsed 拆分
        if prefill_model is None or decode_time is None:
            # 优先保证非负与拆分合理
            if prefill_model is None and decode_time is not None:
                prefill_model = max(0.0, gen_elapsed - decode_time)
            elif decode_time is None and prefill_model is not None:
                decode_time = max(0.0, gen_elapsed - prefill_model)
            else:
                # 都不可得，平分或置 0
                prefill_model = max(0.0, gen_elapsed * 0.5)
                decode_time = max(0.0, gen_elapsed - prefill_model)

        # TTFT = 检索 + 模型 prefill
        ttft = retrieval_time + prefill_model
        # TPOT = decode_time / 输出 token 数
        tpot = decode_time / max(1, num_out)

        # ---- 记录统计 ----
        stat_retrieval.append(retrieval_time)
        stat_prefill.append(prefill_model)
        stat_ttft.append(ttft)
        stat_tpot.append(tpot)
        stat_tokens.append(num_out)
        stat_e2e.append(retrieval_time + gen_elapsed)

        if i % 1 == 0:
            print(f"\n[{i + 1}/{len(questions)}] Question: {question}")
            print("\n--- 模型输出 ---")
            print(pred)
            print(f" → Prediction: {pred}")
            print(f" → Ground Truth: {true_answers[i]}")
            # print(
            #     f" ⏱ retrieval={retrieval_time*1000:.1f} ms, "
            #     f"prefill(model)={prefill_model*1000:.1f} ms, "
            #     f"TTFT={ttft*1000:.1f} ms, "
            #     f"decode={decode_time*1000:.1f} ms, "
            #     f"TPOT={tpot*1000:.1f} ms/token, "
            #     f"out_tokens={num_out}, "
            #     f"E2E={(retrieval_time+gen_elapsed):.3f} s"
            # )
            # metrics = llm.llm_engine.get_metrics()
            # print("完整metrics数据：", metrics)

        predictions.append(pred)

    # ---- 汇总统计 ----
    print("\n========== Latency & Throughput ==========")
    _summarize("Retrieval time", stat_retrieval, "s")
    _summarize("Prefill time", stat_prefill, "s")
    _summarize("TTFT (retrieval+prefill)", stat_ttft, "s")
    _summarize("TPOT (time per output token)", stat_tpot, "s/token")
    if sum(stat_tokens) > 0 and sum(stat_e2e) > 0:
        overall_tps = sum(stat_tokens) / sum(stat_e2e)  # 含检索在内的整体 tokens/sec
        print(f"Throughput (output tokens / E2E seconds): {overall_tps:.2f} tok/s")
    print("==========================================\n")

    comparison_str, metrics = full_evaluation(predictions, true_answers)
    print(metrics)