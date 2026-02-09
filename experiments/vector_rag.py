
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
import tqdm



# -------- LlamaIndex (RAG) --------
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.faiss import FAISSVectorStore
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.core import Settings
except ImportError:
    from llama_index import Settings

# -------- vLLM --------
from vllm import LLM, SamplingParams

# -------- 你项目中的评测 --------
from kblam.metrics_evaluator import full_evaluation

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
    parser.add_argument('--index-path', type=str,
                        default=None,
                        help='索引路径')
    parser.add_argument('--oracle-retrieval', action='store_true',
                        help='使用 oracle 检索：用 is_supporting 段落直接拼接为上下文')
    parser.add_argument('--kb-size', type=int, default=10, help='知识库段落数量')
    parser.add_argument('--without-knowledge', action='store_true',
                        help='不使用知识库')



    return parser.parse_args()


def setup_models(args):

    print(f"setting embed model to {args.embedding_model}")
    # embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
    if os.path.isdir(args.embedding_model):
        # 本地 embedding 模型
        embed_model = HuggingFaceEmbedding(
            model_name=args.embedding_model,
            trust_remote_code=True,
            # device="cuda" if torch.cuda.is_available() else "cpu",
            device="cpu",
        )
    else:
        # HuggingFace Hub 模型
        embed_model = HuggingFaceEmbedding(
            model_name=args.embedding_model,
            # device="cuda" if torch.cuda.is_available() else "cpu",
            device="cpu",
        )

    Settings.embed_model = embed_model
    Settings.llm = None

    print(f"Loading model from {args.model_path}...")
    if 'llama' in args.model_path:
    # 使用VLLM加载模型
        llm = LLM(
            model=args.model_path,
            enforce_eager=True,
            disable_log_stats=False, 
        )
    elif 'qwen2.5-72B' in args.model_path:
        llm = LLM(
            model=args.model_path,
            enforce_eager=True,
            disable_log_stats=False, 
            max_model_len=5000,
            gpu_memory_utilization=0.95,
        )
    elif 'deepseek' in args.model_path:
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
            # disable_log_stats=True, 
        )
    elif 'olmo3-7b' in args.model_path.lower():
        # === 新增：OLMo-3-7B-Instruct ===
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            trust_remote_code=True,   # 必须
            enforce_eager=True,
            max_model_len=8192,
            # disable_log_stats=True, 
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
            # disable_log_stats=True,
        )


    print("Setup models done")
    return llm


def load_dataset(dataset_path: str,):
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset=json.load(f)
    
    return dataset

def normalize_text(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return re.sub(r"[^\w\s]", " ", text)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

SPECIAL_TOKEN_THRESHOLD = 128000  # 排除如 <|eot_id|>=128009 等特殊 token


def _count_non_special_token_ids(token_ids) -> int:
    if not token_ids:
        return 0
    return sum(1 for tid in token_ids if isinstance(tid, int) and tid < SPECIAL_TOKEN_THRESHOLD)


def _extract_latency_from_request_output(req_out):
    out = {
        "prefill_model": None,
        "decode_time": None,
        "num_output_tokens": 0,
    }

    m = getattr(req_out, "metrics", None)
    if m is None:
        print("Warning: metrics is None")
        return out

    # ===== 新版 vLLM (RequestStateStats) =====
    if hasattr(m, "first_token_ts"):
        # prefill: schedule -> first token
        if hasattr(m, "first_token_latency"):
            out["prefill_model"] = float(m.first_token_latency)
        elif hasattr(m, "scheduled_ts"):
            out["prefill_model"] = float(m.first_token_ts - m.scheduled_ts)

        # decode: first token -> last token
        if hasattr(m, "last_token_ts"):
            out["decode_time"] = max(
                0.0, float(m.last_token_ts - m.first_token_ts)
            )

    # ===== token 数 =====
    if hasattr(m, "num_generation_tokens"):
        out["num_output_tokens"] = int(m.num_generation_tokens)
    else:
        try:
            token_ids = req_out.outputs[0].token_ids
            out["num_output_tokens"] = _count_non_special_token_ids(token_ids)
        except Exception:
            pass

    if out["prefill_model"] is None or out["decode_time"] is None:
        print(f"Warning: incomplete metrics: {m}")

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


def get_rag_retriever(docs, args):

    if os.path.exists(args.index_path):
        print(f"Load index from {args.index_path}")
        storage_context = StorageContext.from_defaults(persist_dir=args.index_path)
        index = load_index_from_storage(storage_context)
    else:
        print(f"Index not found in {args.index_path}, create index from documents...")
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=args.index_path)
        print(f"Index saved to {args.index_path}")
    
    retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)
    return retriever

def run_vector_rag(args, llm, dataset):

    docs=[]
    gold_contexts=[]
    for row in dataset:
        supporting_facts_titles=[]
        for sp in row['supporting_facts']:
            supporting_facts_titles.append(sp[0])
        for ctx in row['context']:
            title=ctx[0]
            sentence_text= " ".join(ctx[1])
            docs.append(Document(text=sentence_text, metadata={'title': title}))
            if title in supporting_facts_titles:
                gold_contexts.append(sentence_text)
    print(f"Loaded {len(docs)} documents")
    if not args.oracle_retrieval and not args.without_knowledge:
        # index = VectorStoreIndex.from_documents(docs)
        # retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)
        retriever = get_rag_retriever(docs, args)

    print("Create RAG engine done")
    predictions: List[str] = []
    answers: List[str] = []

    # 统计容器
    stat_retrieval: List[float] = []
    stat_prefill: List[float] = []
    stat_ttft: List[float] = []
    stat_tpot: List[float] = []
    stat_tokens: List[int] = []
    stat_e2e: List[float] = []

    dataset=dataset[:args.n_samples]
    for i, row in enumerate(dataset):
        question=row['question']
        answers.append(row['answer'])

        time_start=time.perf_counter()
        if args.oracle_retrieval:
            context = "\n\n".join(gold_contexts[2*i:2*i+2])
        elif args.without_knowledge:
            context = ""
        else:
            nodes = retriever.retrieve(question)
            context = "\n\n".join([node.get_content() for node in nodes])
        retrieval_time=time.perf_counter()-time_start

        gen_start=time.perf_counter()

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
            print(f"Warning: prefill_model={prefill_model}, decode_time={decode_time}")
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


        predictions.append(pred)

    # ---- 汇总统计 ----
    print("\n========== Latency & Throughput ==========")
    _summarize("Retrieval time", stat_retrieval, "s")
    _summarize("Prefill time", stat_prefill, "s")
    _summarize("TTFT (retrieval+prefill)", stat_ttft, "s")
    _summarize("TPOT (time per output token)", stat_tpot, "s/token")
    if len(stat_e2e) > 0 and sum(stat_e2e) > 0:
        overall_tps = len(stat_e2e) / sum(stat_e2e)  # 含检索在内的整体 tokens/sec
        print(f"Throughput (QPS): {overall_tps:.2f}")
        print(f"Average E2E time: {sum(stat_e2e)/len(stat_e2e):.2f} s")
    print("==========================================\n")

    return predictions, answers

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

def main():
    args=parse_args()
    dataset=load_dataset(args.dataset_path)

    llm=setup_models(args)

    predictions, answers = run_vector_rag(args, llm, dataset)

    clean_model(llm)
    comparison_str, metrics = full_evaluation(predictions, answers)
    print(metrics)


if __name__ == "__main__":
    main()