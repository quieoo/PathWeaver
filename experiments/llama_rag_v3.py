# -*- coding: utf-8 -*-
"""
RAG + HuggingFace Transformers 推理（v3, bnb-8bit, OLMo-3.1-32B）
- 使用 bitsandbytes 8bit（精度优先）
- 显存 45GB 可稳定运行
- 抽取式 QA / RAG 评测
"""

import os
import re
import gc
import json
import time
import argparse
import statistics
import random
from collections import Counter
from typing import List

import torch
import numpy as np

# -------- HuggingFace --------
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# -------- LlamaIndex (RAG) --------
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.core import Settings
except ImportError:
    from llama_index import Settings

# -------- 你项目中的评测 --------
from kblam.metrics_evaluator import full_evaluation


# =========================
# 1) 文本归一化与评测
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


# =========================
# 2) 参数解析
# =========================
def parse_args():
    parser = argparse.ArgumentParser("HF RAG (llama_rag_v3_bnb)")

    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, default="2wiki",
                        choices=["musique", "squad", "2wiki", "hotpot"])
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--kb-size", type=int, default=10)

    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--similarity-top-k", type=int, default=3)
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")

    parser.add_argument("--oracle-retrieval", action="store_true")

    return parser.parse_args()


# =========================
# 3) 数据加载（2Wiki / Hotpot）
# =========================
def load_2wiki_dataset(dataset_path: str,
                       kb_size: int,
                       max_samples: int,
                       source_type: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw = [x for x in raw if x.get("source") == source_type]
    raw = raw[:max_samples]

    all_paras = [p for item in raw for p in item["paragraphs"]]

    ret = []
    for item in raw:
        correct = [
            {
                "paragraph_text": p["paragraph_text"],
                "is_supporting": True,
                "title": p["title"],
            }
            for p in item["paragraphs"]
        ]

        need = kb_size - len(correct)
        candidates = [p for p in all_paras if p not in item["paragraphs"]]
        wrong = random.sample(candidates, need)

        wrong = [
            {
                "paragraph_text": p["paragraph_text"],
                "is_supporting": False,
                "title": p["title"],
            }
            for p in wrong
        ]

        paragraphs = correct + wrong
        random.shuffle(paragraphs)

        ret.append({
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
        })

    return ret


def load_data(args):
    if args.dataset_type in ["2wiki", "hotpot", "musique"]:
        return load_2wiki_dataset(
            args.dataset_path,
            args.kb_size,
            args.n_samples,
            args.dataset_type,
        )
    else:
        raise NotImplementedError


# =========================
# 4) HF 模型与 tokenizer（bnb-8bit）
# =========================
def setup_hf_model(args):
    print(f"Loading HF model (bitsandbytes 8bit): {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,   # 关键：FP16
        load_in_8bit=True,           # bitsandbytes 8bit
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return tokenizer, model


def setup_embedding(args):
    embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model
    )
    Settings.embed_model = embed_model
    Settings.llm = None


# =========================
# 5) Prompt 构造（chat_template）
# =========================
def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =========================
# 6) HF 推理 + 近似计时（评测专用）
# =========================
def hf_generate_with_timing(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 16,
):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    end = time.perf_counter()

    total_time = end - start
    num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]

    prefill_time = total_time * 0.3
    decode_time = total_time - prefill_time
    tpot = decode_time / max(1, num_new_tokens)

    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # 防止偶发扩写
    text = text.split("\n")[0].strip()

    return {
        "text": text,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "num_tokens": num_new_tokens,
        "tpot": tpot,
    }


# =========================
# 7) RAG 检索
# =========================
def retrieve_single_hop(question: str, paras: List[dict], top_k: int):
    docs = []
    for p in paras:
        docs.append(Document(text=f"{p['title']}: {p['paragraph_text']}"))

    index = VectorStoreIndex.from_documents(docs)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    context = "\n\n".join(n.get_content() for n in nodes)
    return context


# =========================
# 8) 主推理循环
# =========================
def run_rag_inference_v3(
    args,
    tokenizer,
    model,
    questions,
    paragraphs_list,
    answers,
    correct_paragraphs,
):
    predictions = []
    stat_ttft = []
    stat_tpot = []

    system_prompt = (
        "You must answer with ONLY the final answer.\n"
        "Rules:\n"
        "- Use ONLY the information in the context.\n"
        "- Output a single short phrase or name.\n"
        "- Do NOT explain.\n"
        "- Do NOT add extra words.\n"
        "- If the answer is not explicitly stated, output: UNKNOWN."
    )
    start_time = time.perf_counter()
    for i, (q, paras) in enumerate(zip(questions, paragraphs_list)):

        retrieval_time = time.perf_counter()
        if args.oracle_retrieval:
            context = "\n\n".join(correct_paragraphs[i])
        else:
            context = retrieve_single_hop(q, paras, args.similarity_top_k)
        retrieval_time = time.perf_counter() - retrieval_time

        user_prompt = f"Context:\n{context}\n\nQuestion: {q}"

        prompt = build_prompt(tokenizer, system_prompt, user_prompt)

        out = hf_generate_with_timing(
            model,
            tokenizer,
            prompt,
            max_new_tokens=16,
        )

        predictions.append(out["text"])
        stat_ttft.append(out["prefill_time"]+retrieval_time)
        stat_tpot.append(out["tpot"])

        if i % 1 == 0:
            print(f"\n[{i}] Q: {q}")
            print(f"Pred: {out['text']}")
            print(
                f"TTFT≈{out['prefill_time']:.3f}s | "
                f"TPOT≈{out['tpot']*1000:.1f} ms/token"
            )
    end_time = time.perf_counter()
    total_time = end_time - start_time

    print("\n=== Timing Summary ===")
    print(f"Avg TTFT ≈ {statistics.mean(stat_ttft):.3f}s")
    print(f"Avg TPOT ≈ {statistics.mean(stat_tpot)*1000:.1f} ms/token")
    print(f"qps: {len(questions) / total_time:.3f}")

    return predictions


# =========================
# 9) main
# =========================
def main():
    args = parse_args()
    setup_embedding(args)
    dev_set = load_data(args)

    questions = [x["question"] for x in dev_set]
    answers = [x["answer"] for x in dev_set]
    paragraphs_list = [x["paragraphs"] for x in dev_set]

    correct_paragraphs = [[] for _ in dev_set]
    for i, ex in enumerate(dev_set):
        for p in ex["paragraphs"]:
            if p.get("is_supporting"):
                correct_paragraphs[i].append(p["paragraph_text"])

    tokenizer, model = setup_hf_model(args)

    predictions = run_rag_inference_v3(
        args,
        tokenizer,
        model,
        questions,
        paragraphs_list,
        answers,
        correct_paragraphs,
    )

    _, metrics = full_evaluation(predictions, answers)
    print("\n=== Evaluation ===")
    print(metrics)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
