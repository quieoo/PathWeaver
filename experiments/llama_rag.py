import re
import time
import json
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Any
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import TextNode, Document

instruction_prompts = """
Please answer the question based on the given text with format: "The {property} of {name} is {description}"
"""

bert_score = evaluate.load("bertscore")

# 1. 按句切分，生成节点
def split_by_sentence(text):
    sents = re.split(r'(?<=[。！？.!?])', text)
    return [s.strip() for s in sents if s.strip()]

def build_nodes(documents, chunk_size=1):
    nodes = []
    for doc in documents:
        sents = split_by_sentence(doc.text)
        # 按chunk_size合并句子
        for i in range(0, len(sents), chunk_size):
            chunk = " ".join(sents[i:i+chunk_size])
            if chunk.strip():
                nodes.append(
                    TextNode(
                        text=chunk,
                        metadata={"source": doc.metadata.get("source", "")}
                    )
                )
    return nodes

# 2. 构建并持久化索引
def build_and_save_index(nodes, embed_model, persist_dir):
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index Saved to {persist_dir}")
    return index

# 3. 加载持久化后的索引
def load_index(persist_dir, embed_model):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    print(f"Index Loaded from {persist_dir}")
    return index

def _format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
    )

def _prune_for_llama(S: str) -> str:
    S = S.replace("<|eot_id|>", "")
    S = S.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    S = S.replace("<|start_header_id|>user<|end_header_id|>", "")
    S = S.replace("<|end_of_text|>", "")
    S = S.replace("assistant", "")
    return S


def perform_eval(model_path, kb_file, doc_file, Q_size, used_cached_index, index_dir, topk):
    embed_model = HuggingFaceEmbedding(
        model_name="/home/sdu/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/",
        device="cpu"
    )

    # ====== 索引构建/加载 ======
    if (used_cached_index):
        index = load_index(index_dir, embed_model)
    else:
        documents = SimpleDirectoryReader(input_files=[doc_file]).load_data()
        nodes = build_nodes(documents, 1)
        index = build_and_save_index(nodes, embed_model, index_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    retriever = index.as_retriever(similarity_top_k=topk)

    dataset = json.load(open(kb_file))
    Q_size = min(Q_size, len(dataset))

    random.seed(42)
    test_kb = random.sample(dataset, Q_size) 
    model_outputs = []
    answers = []
    results_dict = {}

    # ====== 计时器与token统计 ======
    total_search_time = 0.0
    total_gen_time = 0.0
    total_new_tokens = 0
    total_ttft = 0.0
    total_decode_time = 0.0

    for row in tqdm(test_kb[:Q_size]):
        Q = row["Q"]
        answer = row["description"]

        # ---- 检索计时 ----
        t_search_start = time.perf_counter()
        retrieved_nodes = retriever.retrieve(Q)
        t_search_end = time.perf_counter()
        total_search_time += (t_search_end - t_search_start)

        retrieved_texts = [n.get_text() for n in retrieved_nodes]
        Q_prompt = instruction_prompts + " ".join(retrieved_texts) + " " + Q
        
        query = _format_Q_llama(Q_prompt)
        inputs = tokenizer(query, return_tensors="pt").to(model.device)

        def my_streamer_callback(stream_end=False):
            """自定义 streamer 回调，用来统计 TTFT"""
            nonlocal first_token_received, ttft

            if not stream_end:
                if not first_token_received:
                    ttft = time.time() - ttft_start_time
                    # print(f"\n[TTFT] 第一个 token 出来耗时: {ttft:.4f} 秒\n")
                    first_token_received = True

        # 4. 给 TextStreamer 注册回调（在 transformers 里可以继承 TextStreamer 做定制）
        class TTFTStreamer(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False):
                if text.strip() != "":
                    my_streamer_callback(stream_end=stream_end)

        # 使用自定义的 streamer
        ttft_streamer = TTFTStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_start_time = time.time()
        ttft_start_time = time.time()
        first_token_received = False    
        ttft = None  

        with torch.autograd.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                output_attentions=True,
                return_dict_in_generate=True,
                # do_sample = True,
                streamer=ttft_streamer
            )
        gen_end_time = time.time()
        gen_time = gen_end_time - gen_start_time
        total_gen_time += gen_time
        total_ttft += ttft
        decode_time = gen_time - ttft
        total_decode_time += decode_time

        generated_ids = outputs.sequences.squeeze()  # 使用 .sequences 获取生成的 token ids
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = _prune_for_llama(response)

        if Q_prompt in response:
            response = response.split(Q_prompt, 1)[1]
        elif Q in response:
            response = response.split(Q, 1)[1]

        # ---- 新增：token 统计（以清洗前的输出为准）----
        try:
            token_ids = tokenizer(response, add_special_tokens=False, return_tensors=None,).input_ids
            if isinstance(token_ids[0], list):
                # 某些tokenizer可能返回嵌套；取第一条
                total_new_tokens += len(token_ids[0])
            else:
                total_new_tokens += len(token_ids)
        except Exception:
            # 兜底：无法分词则不计入
            pass
        
        pattern = r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)'
        match = re.search(pattern, response)
        if match:
            response = match.group(1).strip().rstrip('.')

        model_outputs.append(response)
        answers.append(answer)

    # ====== 评价指标 ======
    bertscore = bert_score.compute(
        predictions=model_outputs,
        references=answers,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        batch_size=16
    )
    for k, v in bertscore.items():
        if isinstance(v, list):
            results_dict[f"bert_score_{k}"] = float(np.mean(v))
            print(k, np.mean(v))

    # ====== 时间与速率指标 ======
    answered = max(1, len(model_outputs))

    # results_dict["index_build_time"]   = float(index_build_time)
    results_dict["total_time"]         = float(total_gen_time + total_search_time)
    results_dict["avg_ttft"]           = float(total_ttft) / answered * 1000
    results_dict["avg_search_time"]    = float(total_search_time) / answered * 1000
    results_dict["tpot"]               = float(total_decode_time / total_new_tokens) * 1000
    results_dict["num_queries"]        = int(answered)
    results_dict["total_new_tokens"]   = int(total_new_tokens)

    return model_outputs, answers, results_dict


def write_to_json(
    data: Any, filepath: str, indent: int = 4, encoding: str = "utf-8"
) -> bool:
    try:
        # Convert string path to Path object
        file_path = Path(filepath)

        # Write the JSON file
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=True,  # For consistent output
                default=str,  # Handle non-serializable objects by converting to string
            )
    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")

if __name__ == "__main__":
    model_outputs = []
    answers = []
    
    model_path = "/home/sdu/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/"
    kb_file = "./data/refusal/randomkb_10000.json"
    doc_file = './data/refusal/randomkb_100.txt'
    Q_size = 100
    outlier_ratio = 0.5
    topk = 10
    used_cached_index = True
    index_dir = "./index/kb10000"
    model_outputs, answers, results = perform_eval(model_path, kb_file, doc_file, Q_size, used_cached_index, index_dir, topk)

    output_path = './results_ttft/rag10000_topk10.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        for m, a in zip(model_outputs, answers):
            f.write(f"Model Output: {m}\n")
            f.write(f"Answer      : {a}\n")
    write_to_json(results, "./results_ttft/rag10000_topk10.json")

