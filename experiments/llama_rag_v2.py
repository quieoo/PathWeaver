import re
import os
import json
import argparse
import torch
# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.core import Settings
except ImportError:
    from llama_index import Settings


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from vllm import LLM
from vllm import SamplingParams

from kblam.metrics_evaluator import full_evaluation
import gc, time


# ----------------------------
# 1. 评价指标函数
# ----------------------------
def normalize_text(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return re.sub(r"[^\w\s]", " ", text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)

def info_contain_score(prediction, ground_truth):
    return normalize_text(ground_truth) in normalize_text(prediction)

def parse_args():
    parser = argparse.ArgumentParser(description='Llama RAG with MuSiQue Dataset')
    
    # 数据集参数
    parser.add_argument('--dataset-path', type=str, default='/mnt/n0/datasets/MuSiQue/',
                        help='MuSiQue数据集本地路径')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='测试样本数量，建议从 5-10 开始测试')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, default='/mnt/n0/models/llama3_8B_instruct/',
                        help='模型路径')
    
    # 检索参数
    parser.add_argument('--similarity-top-k', type=int, default=5,
                        help='相似度搜索返回的top k个结果')
    parser.add_argument('--embedding-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='嵌入模型名称')
    
    parser.add_argument('--oracle-retrieval', action='store_true',
                        help='是否使用oracle检索（基于段落）')
    return parser.parse_args()

# ----------------------------
# 2. 加载 MuSiQue 数据（取前 N 个样本）
# ----------------------------
# 参数将在main函数中通过argparse获取

# 从本地加载MuSiQue数据集
def load_musique_dataset(dataset_path, max_samples=None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            dataset.append(item)
            
            if max_samples and len(dataset) >= max_samples:
                break
    
    print(f"从本地加载了 {len(dataset)} 个MuSiQue样本")
    return dataset
# 加载数据集
def load_data(args):
    return load_musique_dataset(args.dataset_path, args.n_samples)


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
    import torch, gc
    if llm is not None:
        if hasattr(llm, "engine") and llm.engine is not None:
            try:
                llm.engine.shutdown()
                print("✅ vLLM engine shutdown complete.")
            except Exception as e:
                print(f"⚠️ engine shutdown failed: {e}")
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    print(f"✅ CUDA memory cleared. Remaining allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    
# ----------------------------
# 4. 显式检索 + 手动调用 LLM
# ----------------------------

def retrieve_single_hop(question, paras, args):
    # 构建文档
    docs = []
    for p in paras:
        text = f"{p['title']}: {p['paragraph_text']}"
        docs.append(Document(text=text))

    # 构建索引
    index = VectorStoreIndex.from_documents(docs)

    retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)
    nodes = retriever.retrieve(question)

    # 拼接检索到的上下文
    context = "\n\n".join([node.get_content() for node in nodes])

    return context

def run_rag_inference(args, llm, questions, paragraphs_list, answers, correct_paragraphs):
    predictions = []
    
    for i, (question, paras) in enumerate(zip(questions, paragraphs_list)):
        print(f"\n[{i+1}/{args.n_samples}] Question: {question}")

        if args.oracle_retrieval:
            context=correct_paragraphs[i]
        else:
            context = retrieve_single_hop(question, paras, args)
        # context = retrieve_multi_hop(question, paras, args, local_llm=llm)
        if "llama" in args.model_path:
            em_enhanced_prompt=(
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
            response = llm.generate(em_enhanced_prompt)
            pred = str(response[0].outputs[0].text).strip()
        elif "deepseek" in args.model_path:
            deepseek_prompt = (
                f"<|system|>\n"
                f"Answer the question using ONLY the given context. "
                f"Output ONLY the exact answer phrase in English. "
                f"Do NOT add any explanations, prefixes (like 'Answer:', 'The answer is'), suffixes, or punctuation.\n\n"
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

            sp=SamplingParams(
                temperature=0.5,
                top_p=0.95,
                max_tokens=1024,
            )
            response = llm.generate(deepseek_prompt, sampling_params=sp)
            pred = str(response[0].outputs[0].text).strip()

            if i % 10==0:
                # print("\n--- 第一个样本全部上下文 ---")
                # print(paras)
                # print("\n--- 第一个样本正确段落 ---")
                # print(correct_paragraphs[i])
                # print("\n--- RAG输出的上下文 ---")
                # print(context)
                print("\n--- 模型输出 ---")
                print(pred)

            try:
                parts = pred.split("</think>")
                if len(parts) > 1:
                    pred = parts[1].strip()
            except Exception as e:
                print(f"Warning: processing prediction: {e}")
            
            # 取“Answer:"后的内容
            if "Answer: " in pred:
                pred = pred.split("Answer: ")[-1].strip()
        elif "qwen" in args.model_path:
            system_prompt = (
                f"You answer questions with ONLY the exact answer phrase. "
                f"Never add explanations, prefixes, or punctuation. "
                f"Examples:\n"
                f"Question: Who wrote '1984'? → George Orwell\n"
                f"Question: Capital of France? → Paris\n"
                f"Question: When was Einstein born? → 1879\n"
            )
            user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"
            prompt = (
                "<|system|>\n" + system_prompt + "\n<|end|>\n"
                "<|user|>\n" + user_prompt + "\n<|end|>\n"
                "<|assistant|>\n"
            )
            response = llm.generate(prompt)
            pred = str(response[0].outputs[0].text).strip()

            # 去除“<|end|>”及其之后的内容
            if "<|end|>" in pred:
                pred = pred.split("<|end|>")[0].strip()
            # 去除“Explanation:”及其之后的内容
            if "Explanation: " in pred:
                pred = pred.split("Explanation: ")[0].strip()
            # 只保留第一行
            pred = pred.split("\n")[0].strip()
        else:
            print("Unknown model path")

        predictions.append(pred)
        print(f"  → Prediction: {pred}")
        print(f"  → Ground Truth: {answers[i]}")
    
    return predictions



# ----------------------------
# 5. 计算评价指标
# ----------------------------
def evaluate_predictions(predictions, answers, n_samples):
    em_scores = [exact_match_score(p, g) for p, g in zip(predictions, answers)]
    f1_scores = [f1_score(p, g) for p, g in zip(predictions, answers)]
    contain_scores = [info_contain_score(p, g) for p, g in zip(predictions, answers)]

    em = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores)
    contain = sum(contain_scores) / len(contain_scores)

    print("\n" + "="*50)
    print(f"Results on {n_samples} MuSiQue samples:")
    print(f"Exact Match (EM): {em:.2%}")
    print(f"F1 Score:         {f1:.2%}")
    print(f"Info Contain:     {contain:.2%}")
    print("="*50)
    
    return em, f1, contain


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载数据
    dev_set = load_data(args)
    questions = [ex["question"] for ex in dev_set]
    answers = [ex["answer"] for ex in dev_set]
    paragraphs_list = [ex["paragraphs"] for ex in dev_set]
    
    # 每个样本中的正确答案所在段落
    correct_paragraphs = [[] for _ in dev_set]
    for i, ex in enumerate(dev_set):
        correct_paragraph = None
        for p in ex["paragraphs"]:
            if p["is_supporting"]:
                correct_paragraphs[i].append(p["paragraph_text"])
                
    # 设置模型
    llm = setup_models(args)
    
    # 执行RAG推理
    predictions = run_rag_inference(args, llm, questions, paragraphs_list, answers, correct_paragraphs)
    
    # 清理模型
    clean_model(llm)

    # 评估结果
    # evaluate_predictions(predictions_copy, answers_copy, args.n_samples)

    # 完整评估

    comparison_str, metrics = full_evaluation(predictions, answers)
    print(metrics)
 




if __name__ == "__main__":
    main()