# -*- coding: utf-8 -*-
"""
RAG + vLLM 推理，统计：
- TTFT = Retrieval(检索) + Prefill(模型首 token 前)
- TPOT = Decode 时间 / 输出 token 数（排除特殊 token）
并输出总体统计与 full_evaluation 指标。
"""

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

# 优化 CUDA 内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# -------- LlamaIndex (RAG) --------
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.core import Settings
except ImportError:
    from llama_index import Settings

# -------- vLLM --------
from vllm import LLM, SamplingParams

# -------- 你项目中的评测 --------
from kblam.metrics_evaluator import full_evaluation


# =========================
# 1) 文本归一化与基础评测
# =========================
def normalize_text(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return re.sub(r"[^\w\s]", " ", text)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(pred_tokens))
    recall = num_same / max(1, len(gt_tokens))
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def info_contain_score(prediction: str, ground_truth: str) -> bool:
    return normalize_text(ground_truth) in normalize_text(prediction)


# =========================
# 2) 参数与数据加载
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Llama RAG with MuSiQue (TTFT & TPOT)')
    # 数据集参数（建议直接给到 jsonl 路径）
    parser.add_argument('--dataset-path', type=str,
                        default='/mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl',
                        help='MuSiQue 数据集（jsonl）本地路径')
    parser.add_argument('--n-samples', type=int, default=10, help='测试样本数量')
    parser.add_argument('--dataset-type', type=str, default='musique', help='数据集类型（musique/squad）')

    # 模型参数
    parser.add_argument('--model-path', type=str,
                        default='/mnt/n0/models/llama3_8B_instruct/',
                        help='vLLM 模型路径（llama/deepseek/qwen 等）')

    # 检索参数
    parser.add_argument('--similarity-top-k', type=int, default=5, help='相似度检索 Top-K')
    parser.add_argument('--embedding-model', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2',
                        help='嵌入模型名称')
    parser.add_argument('--oracle-retrieval', action='store_true',
                        help='使用 oracle 检索：用 is_supporting 段落直接拼接为上下文')
    parser.add_argument('--kb-size', type=int, default=10, help='知识库段落数量')



    return parser.parse_args()


def load_musique_dataset(dataset_path: str, max_samples: int | None = None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            dataset.append(item)
            if max_samples and len(dataset) >= max_samples:
                break
    print(f"从本地加载了 {len(dataset)} 个 MuSiQue 样本")
    return dataset

def load_squad_dataset(dataset_path: str, kb_size: int, max_samples: int | None = None):
    #读取json文件
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data[:max_samples]
    # 首先收集所有可能的段落作为候选知识库
    all_paragraphs = []
    for item in data:
        all_paragraphs.append({
            "paragraph_text": item['context'],
            "is_supporting": False,
            # 截取第一句话作为title
            "title": item['context'].split('.')[0]
        })
    
    
    ret = []
    
    # 为每个QA对创建样本，确保每个样本的paragraphs包含kb_size个段落，其中必定包含正确段落
    for item in data:
        # 当前item的段落是正确段落
        correct_paragraph = {
            "paragraph_text": item['context'],
            "is_supporting": True,  # 标记为支持性段落
            "title": item['context'].split('.')[0]
        }
        
        # 从其他段落中随机选择kb_size-1个段落
        other_paragraphs = [p for p in all_paragraphs if p["paragraph_text"] != item['context']]
        # 如果其他段落不足，就重复使用
        while len(other_paragraphs) < kb_size - 1:
            other_paragraphs.extend(other_paragraphs)
        
        # 随机选择kb_size-1个其他段落
        selected_paragraphs = random.sample(other_paragraphs, kb_size - 1)
        
        # 组合正确段落和随机选择的段落
        paragraphs = [correct_paragraph] + selected_paragraphs
        # 打乱顺序，让正确段落的位置随机
        random.shuffle(paragraphs)
        
        # 为当前item中的每个QA对创建样本
        for qa in item['qas']:
            ret.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'paragraphs': paragraphs
            })
            if len(ret) >= max_samples:
                break
        if len(ret) >= max_samples:
            break
    
    return ret

def load_2wiki_dataset(dataset_path: str,
                       kb_size: int,
                       max_samples: int | None = None, source_type: str = '2wiki'):
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset=json.load(f)
    
    new_dataset = []
    for item in dataset:
        if item.get('source') != source_type:
            continue
        new_dataset.append(item)
    dataset=new_dataset[:max_samples]
    print(f"原始数据集大小: {len(new_dataset)}")
    print(f"过滤后的数据集大小: {len(dataset)}")

    # 1. 把所有“其它”段落展平成候选池
    candidate_paras = [p for item in new_dataset for p in item['paragraphs']]

    ret = []
    for item in dataset:
        # 2. 当前样本的正确段落
        correct_paras = [
            {
                "paragraph_text": p['paragraph_text'],
                "is_supporting": True,
                "title": p['title']
            }
            for p in item['paragraphs']
        ]

        # 3. 需要再抽多少条
        need = kb_size - len(correct_paras)
        if need < 0:
            # 如果正确段落本身就比 kb_size 多，直接截断或报错
            raise ValueError(
                f"kb_size({kb_size}) < 正确段落数({len(correct_paras)})"
            )

        # 4. 排除掉当前样本的段落，避免重复
        this_ids = {id(p) for p in item['paragraphs']}
        available = [p for p in candidate_paras if id(p) not in this_ids]

        if len(available) < need:
            raise ValueError("候选池不足，无法凑齐 kb_size 段落")

        chosen_wrong = random.sample(available, need)
        wrong_paras = [
            {
                "paragraph_text": p['paragraph_text'],
                "is_supporting": False,
                "title": p['title']
            }
            for p in chosen_wrong
        ]

        # 5. 合并 & 洗牌
        paragraphs = correct_paras + wrong_paras
        random.shuffle(paragraphs)

        ret.append({
            'question': item['question'],
            'answer': item['answer'],
            'paragraphs': paragraphs
        })

    return ret



def load_data(args):
    if args.dataset_type == 'musique':
        # return load_musique_dataset(args.dataset_path, args.n_samples)
        return load_2wiki_dataset(args.dataset_path, args.kb_size, args.n_samples, source_type=args.dataset_type)
    elif args.dataset_type == 'squad':
        return load_squad_dataset(args.dataset_path, args.kb_size, args.n_samples)
    elif args.dataset_type == '2wiki':
        return load_2wiki_dataset(args.dataset_path, args.kb_size, args.n_samples, source_type=args.dataset_type)
    elif args.dataset_type == 'hotpot':
        return load_2wiki_dataset(args.dataset_path, args.kb_size, args.n_samples, source_type=args.dataset_type)
    else:
        raise ValueError(f"未知数据集类型: {args.dataset_type}")


# ----------------------------
# 3. 配置 LLM 和 Embedding
# ----------------------------
def setup_models(args):
    print(f"Loading model from {args.model_path}...")
    if 'llama' in args.model_path:
    # 使用VLLM加载模型
        llm = LLM(
            model=args.model_path,
            enforce_eager=True,
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

    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
    Settings.embed_model = embed_model
    Settings.llm = None
    
    return llm


def clean_model(llm):
    if llm is not None:
        if hasattr(llm, "engine") and llm.engine is not None:
            try:
                llm.engine.shutdown()
                print("✅ vLLM engine shutdown complete.")
            except Exception as e:
                print(f"⚠️ engine shutdown failed: {e}")
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        print(f"✅ CUDA memory cleared. Remaining allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


# =========================
# 4) RAG 检索（计时）
# =========================
def retrieve_single_hop(question: str, paras: List[dict], top_k: int) -> Tuple[str, float]:
    t0 = time.perf_counter()

    docs = []
    for p in paras:
        text = f"{p['title']}: {p['paragraph_text']}"
        docs.append(Document(text=text))

    index = VectorStoreIndex.from_documents(docs)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    context = "\n\n".join([node.get_content() for node in nodes])

    t1 = time.perf_counter()
    retrieval_time = t1 - t0
    return context, retrieval_time


# =========================
# 5) vLLM 指标提取与统计
# =========================
SPECIAL_TOKEN_THRESHOLD = 128000  # 排除如 <|eot_id|>=128009 等特殊 token


def _count_non_special_token_ids(token_ids) -> int:
    if not token_ids:
        return 0
    return sum(1 for tid in token_ids if isinstance(tid, int) and tid < SPECIAL_TOKEN_THRESHOLD)


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
            ft = getattr(m, "first_token_time", None)
            fs = getattr(m, "first_scheduled_time", None)
            fin = getattr(m, "finished_time", None)
            if ft is not None and fs is not None:
                out["prefill_model"] = float(ft) - float(fs)
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


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[k]


def _summarize(name: str, values: List[float], unit: str = "s"):
    if not values:
        print(f"[WARN] No values for {name}")
        return
    mean_v = statistics.fmean(values)
    p50_v = _percentile(values, 50)
    p95_v = _percentile(values, 95)
    print(f"{name}: mean={mean_v:.4f}{unit}, p50={p50_v:.4f}{unit}, p95={p95_v:.4f}{unit}")


# =========================
# 6) 推理主循环（含 TTFT/TPOT）
# =========================
def run_rag_inference(
    args, llm, questions: List[str], paragraphs_list: List[List[dict]],
    answers: List[str], correct_paragraphs: List[List[str]]
):
    predictions: List[str] = []

    # 统计容器
    stat_retrieval: List[float] = []
    stat_prefill: List[float] = []
    stat_ttft: List[float] = []
    stat_tpot: List[float] = []
    stat_tokens: List[int] = []
    stat_e2e: List[float] = []

    for i, (question, paras) in enumerate(zip(questions, paragraphs_list)):
        

        # ---- 检索阶段 ----
        if args.oracle_retrieval:
            ctx_list = correct_paragraphs[i] if isinstance(correct_paragraphs[i], list) else [str(correct_paragraphs[i])]
            context = "\n\n".join(ctx_list)
            retrieval_time = 0.0
        else:
            context, retrieval_time = retrieve_single_hop(question, paras, args.similarity_top_k)

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

        if i % 100 == 0:
            print(f"\n[{i + 1}/{len(questions)}] Question: {question}")
            print("\n--- 模型输出 ---")
            print(pred)
            print(f" → Prediction: {pred}")
            print(f" → Ground Truth: {answers[i]}")
            print(
                f" ⏱ retrieval={retrieval_time*1000:.1f} ms, "
                f"prefill(model)={prefill_model*1000:.1f} ms, "
                f"TTFT={ttft*1000:.1f} ms, "
                f"decode={decode_time*1000:.1f} ms, "
                f"TPOT={tpot*1000:.1f} ms/token, "
                f"out_tokens={num_out}, "
                f"E2E={(retrieval_time+gen_elapsed):.3f} s"
            )

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

    return predictions


# =========================
# 7) 额外评测（可选）
# =========================
def evaluate_predictions(predictions: List[str], answers: List[str], n_samples: int):
    em_scores = [exact_match_score(p, g) for p, g in zip(predictions, answers)]
    f1_scores = [f1_score(p, g) for p, g in zip(predictions, answers)]
    contain_scores = [info_contain_score(p, g) for p, g in zip(predictions, answers)]
    em = sum(em_scores) / max(1, len(em_scores))
    f1 = sum(f1_scores) / max(1, len(f1_scores))
    contain = sum(contain_scores) / max(1, len(contain_scores))
    print("\n" + "=" * 50)
    print(f"Results on {n_samples} MuSiQue samples:")
    print(f"Exact Match (EM): {em:.2%}")
    print(f"F1 Score:        {f1:.2%}")
    print(f"Info Contain:    {contain:.2%}")
    print("=" * 50)
    return em, f1, contain


# =========================
# 8) 主函数
# =========================
def main():
    args = parse_args()

    # 加载数据
    dev_set = load_data(args)
    questions = [ex["question"] for ex in dev_set]
    answers = [ex["answer"] for ex in dev_set]
    paragraphs_list = [ex["paragraphs"] for ex in dev_set]

    # 每个样本的正确段落（oracle 用）
    correct_paragraphs = [[] for _ in dev_set]
    for i, ex in enumerate(dev_set):
        for p in ex["paragraphs"]:
            if p.get("is_supporting", False):
                correct_paragraphs[i].append(p["paragraph_text"])

    # 模型
    llm = setup_models(args)

    # 推理
    predictions = run_rag_inference(args, llm, questions, paragraphs_list, answers, correct_paragraphs)

    # 清理
    clean_model(llm)

    # # 评测（两种）
    # # 1) 简单三指标
    # evaluate_predictions(predictions, answers, args.n_samples)
    # 2) 你项目内的完整评测
    comparison_str, metrics = full_evaluation(predictions, answers)
    print(metrics)


if __name__ == "__main__":
    main()
