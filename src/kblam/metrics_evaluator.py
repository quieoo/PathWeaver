import numpy as np
import evaluate
from bert_score import score
import torch
import re
import string
import os

# Faithfulness 本地嵌入模型
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# GPT 评估器（若有 OpenAI Key）
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ======================
# 🔹 文本标准化
# ======================
def normalize_text(text: str) -> str:
    """Standardize text for fair comparison."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================
# 🔹 Exact Match
# ======================
def calculate_exact_match(model_outputs: list[str], references: list[str]) -> float:
    matches = 0
    for pred, ref in zip(model_outputs, references):
        if normalize_text(pred) == normalize_text(ref):
            matches += 1
    return matches / len(model_outputs)


# ======================
# 🔹 F1-Overlap
# ======================
def f1_overlap(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    common = set(pred_tokens) & set(ref_tokens)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_faithfulness_local(model_outputs, references):
    """Use sentence-transformer embeddings to compute semantic faithfulness."""
    if not _HAS_ST:
        print("⚠️ sentence-transformers not installed. Skipping faithfulness metric.")
        return None
    print("Calculating Faithfulness (semantic embedding similarity)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    emb_pred = model.encode(model_outputs, convert_to_tensor=True, show_progress_bar=False)
    emb_ref = model.encode(references, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(emb_pred, emb_ref).diagonal()
    return float(cosine_scores.mean())

# ======================
# 🔹 Faithfulness 检测 - 阿里云百炼 API 版
# ======================
def compute_faithfulness_bailian(model_outputs, references, model_name="qwen-plus", absolute=False):

    import re, os
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("⚠️ No DASHSCOPE_API_KEY found in environment. Using local faithfulness instead.")
        return compute_faithfulness_local(model_outputs, references)

    # print(f"Calculating Faithfulness using Bailian ({model_name}) in batch mode...")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # --- Step 1: 构造批量 prompt ---
    prompt_lines = []
    for i, (pred, ref) in enumerate(zip(model_outputs, references)):
        prompt_lines.append(f"{i+1}. 模型输出: {pred}\n参考答案: {ref}\n评分:")
    prompt = "\n".join(prompt_lines)

    # --- Step 2: 一次性请求模型 ---
    try:
        if absolute:
            output_format="输出格式：如果语义一致则输出1，否则输出0。不要解释。"
        else:
            output_format="输出格式：每一行仅输出一个小数（0.0~1.0），与输入编号对应。不要解释。"
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": (
                    "你是一名文本一致性评估助手。请判断每对文本在事实层面的语义一致程度。\n"
                    f"{output_format}"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content.strip()
        # print(f"🔹 {i} Raw model output:{content}")
    except Exception as e:
        print(f"⚠️ DashScope request error: {e}")
        return 0.0

    lines = [l.strip() for l in content.splitlines() if l.strip()]
    scores = []

    for line in lines:
        # 匹配 0 ~ 1 之间的小数或整数
        match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", line)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    scores.append(val)
            except ValueError:
                continue

    if not scores:
        print("⚠️ No valid scores found in model output.")
        return 0.0
        
    # --- Step 4: 数量对齐 ---
    if len(scores) != len(model_outputs):
        print(f"⚠️ Warning: Expected {len(model_outputs)} scores, got {len(scores)}")
        # 如果分数太少，补齐均值
        if len(scores) < len(model_outputs):
            mean_val = np.mean(scores)
            scores += [mean_val] * (len(model_outputs) - len(scores))
        else:
            scores = scores[: len(model_outputs)]

    avg_score = float(np.mean(scores))
    return avg_score


def compute_faithfulness_gpt(model_outputs, references):
    """Use GPT model to judge factual consistency."""
    if not _HAS_OPENAI:
        print("⚠️ OpenAI SDK not installed. Skipping GPT faithfulness metric.")
        return None
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️ No OpenAI API key found. Using local faithfulness instead.")
        return compute_faithfulness_local(model_outputs, references)

    print("Calculating Faithfulness using GPT-based evaluation...")
    client = OpenAI()
    scores = []
    for pred, ref in zip(model_outputs, references):
        prompt = f"""
            You are a factual consistency evaluator.
            Compare the MODEL OUTPUT and REFERENCE and rate consistency from 0 (not faithful) to 1 (fully consistent).

            MODEL OUTPUT: {pred}
            REFERENCE: {ref}

            Answer ONLY with a number between 0 and 1.
            """
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score = float(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error in GPT faithfulness scoring: {e}")
            score = 0.0
        scores.append(score)
    return float(np.mean(scores))


# ======================
# 🔹 综合评估函数
# ======================
def evaluate_model_outputs(model_outputs: list[str], references: list[str], lang: str = "en", bert_device: str = "cpu") -> dict:
    if len(model_outputs) != len(references):
        raise ValueError(
            f"Number of model outputs ({len(model_outputs)}) does not match number of references ({len(references)})"
        )

    results_dict = {}

    # --- ROUGE ---
    try:
        print("Calculating ROUGE scores...")
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=model_outputs, references=references)
        for key, value in rouge_scores.items():
            results_dict[key] = float(value)
        print("✅ ROUGE computed successfully.")
    except Exception as e:
        print(f"❌ Error calculating ROUGE: {e}")

    # --- Exact Match ---
    try:
        em_score = calculate_exact_match(model_outputs, references)
        results_dict["exact_match"] = float(em_score)
        print(f"✅ Exact Match (EM): {em_score:.4f}")
    except Exception as e:
        print(f"❌ Error calculating Exact Match: {e}")

    # --- F1-Overlap ---
    try:
        f1_scores = [f1_overlap(p, r) for p, r in zip(model_outputs, references)]
        f1_mean = float(np.mean(f1_scores))
        results_dict["f1_overlap"] = f1_mean
        print(f"✅ F1-Overlap: {f1_mean:.4f}")
    except Exception as e:
        print(f"❌ Error calculating F1-Overlap: {e}")

    # --- BERT ---
    from transformers import AutoTokenizer, AutoModel
    import os

    os.environ["TRANSFORMERS_USE_SAFETENSORS"] = "1"

    MODEL_NAME = "microsoft/deberta-xlarge-mnli"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,   # 关键
        torch_dtype="auto",
        device_map=bert_device,
    )
    model.eval()
    from bert_score.scorer import BERTScorer
    import torch
    from tqdm import tqdm

    scorer = BERTScorer(
        model_type=None,        # 不让它自己加载
        num_layers=None,
        batch_size=4,
        device=bert_device,
        lang="en",
    )
    print(f"🔸 BERTScore model loaded: {MODEL_NAME}")
    
    # 🔥 手动注入（关键步骤）
    scorer._model = model
    scorer._tokenizer = tokenizer

    def bert_score_with_progress_and_gpu_monitor(candidates, references):
        """带进度条+GPU监控的BERTScore打分函数"""
        total_samples = len(candidates)
        batch_size = scorer.batch_size
        total_batches = (total_samples + batch_size - 1) // batch_size  # 计算总批次
        
        # 初始化返回结果
        all_P, all_R, all_F1 = [], [], []
        
        print(f"\n📊 开始BERTScore打分 | 总样本数: {total_samples} | 批次大小: {batch_size} | 总批次: {total_batches}")
        print("="*80)
        
        # 分批推理 + 进度条展示
        with torch.no_grad():  # GPU必备：关闭梯度，省显存+提速
           for idx in tqdm(range(total_batches), desc="📈 打分进度", leave=True, colour="green"):
                # 切分当前批次数据
                start = idx * batch_size
                end = min((idx + 1) * batch_size, total_samples)
                batch_cand = candidates[start:end]
                batch_ref = references[start:end]
                
                # 单批次打分
                batch_P, batch_R, batch_F1 = scorer.score(batch_cand, batch_ref)
                
                # 收集结果
                all_P.append(batch_P.cpu())
                all_R.append(batch_R.cpu())
                all_F1.append(batch_F1.cpu())
                
                # ✅ 实时GPU状态监控 (每5个批次打印一次，避免刷屏)
                if bert_device == "cuda" and idx % 5 == 0:
                    used_gpu_mem = torch.cuda.memory_allocated() / 1024**3  # 已用显存(GB)
                    max_gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 # 峰值显存(GB)
                    gpu_util = torch.cuda.utilization()  # GPU利用率(%)
                    cpu_util = psutil.cpu_percent()      # CPU利用率(%)
                    print(f"\n🖥️ GPU监控 | 显存占用: {used_gpu_mem:.2f}GB | 峰值显存: {max_gpu_mem:.2f}GB | GPU利用率: {gpu_util}% | CPU利用率: {cpu_util}%")
        
        # 合并所有批次结果
        P = torch.cat(all_P)
        R = torch.cat(all_R)
        F1 = torch.cat(all_F1)
        
        print("="*80)
        print(f"✅ 打分完成！整体P均值: {P.mean().item():.4f} | R均值: {R.mean().item():.4f} | F1均值: {F1.mean().item():.4f}")
        return P, R, F1

    # ===================== 执行打分 =====================
    # 替换成你自己的 model_outputs 和 references 即可
    P, R, F1 = bert_score_with_progress_and_gpu_monitor(model_outputs, references)

    # P, R, F1 = scorer.score(
    #     model_outputs,
    #     references,
    # )

    if P is not None:
        results_dict["bert_score_precision"] = float(np.mean(P.numpy()))
        results_dict["bert_score_recall"] = float(np.mean(R.numpy()))
        results_dict["bert_score_f1"] = float(np.mean(F1.numpy()))
        print("✅ BERTScore computed successfully.")

    # --- Faithfulness ---
    try:
        if "OPENAI_API_KEY" in os.environ:
            faith = compute_faithfulness_gpt(model_outputs, references)
        elif "DASHSCOPE_API_KEY" in os.environ:
            if len(model_outputs) > 20:
                print(f"🔸 Large input detected ({len(model_outputs)} samples), splitting into batches of 20...")
                all_scores = []
                for i in range(0, len(model_outputs), 20):
                    batch_score = compute_faithfulness_bailian(
                        model_outputs[i:i+20],
                        references[i:i+20],
                        model_name="qwen-plus"
                    )
                    all_scores.append(batch_score)
                faith = float(np.mean(all_scores))
            else:
                faith = compute_faithfulness_bailian(model_outputs, references, model_name="qwen-plus")
        else:
            print("🔸 No API key found for Faithfulness, using local model...")
            faith=None
            # faith = compute_faithfulness_local(model_outputs, references)
        if faith is not None:
            results_dict["faithfulness"] = faith
            print(f"✅ Faithfulness: {faith:.4f}")
    except Exception as e:
        print(f"❌ Error calculating Faithfulness: {e}")

    # --- Faithfulness01 ---
    try:
        if "OPENAI_API_KEY" in os.environ:
            faith = compute_faithfulness_gpt(model_outputs, references)
        elif "DASHSCOPE_API_KEY" in os.environ:
            if len(model_outputs) > 20:
                print(f"🔸 Large input detected ({len(model_outputs)} samples), splitting into batches of 20...")
                all_scores = []
                for i in range(0, len(model_outputs), 20):
                    batch_score = compute_faithfulness_bailian(
                        model_outputs[i:i+20],
                        references[i:i+20],
                        model_name="qwen-plus",
                        absolute=True
                    )
                    all_scores.append(batch_score)
                faith = float(np.mean(all_scores))
            else:
                faith = compute_faithfulness_bailian(model_outputs, references, model_name="qwen-plus", absolute=True)
        else:
            print("🔸 No API key found for Faithfulness01, using local model...")
            faith=None
            # faith = compute_faithfulness_local(model_outputs, references)
        if faith is not None:
            results_dict["faithfulness01"] = faith
            print(f"✅ Faithfulness01: {faith:.4f}")
    except Exception as e:
        print(f"❌ Error calculating Faithfulness01: {e}")

    return results_dict


# ======================
# 🔹 可视化对比
# ======================
def print_comparison(model_outputs: list[str], references: list[str]) -> str:
    results = []
    for i, (output, reference) in enumerate(zip(model_outputs, references)):
        results.append(f"Sample {i}:")
        results.append(f"  Model output: {output}")
        results.append(f"  Ground truth: {reference}")
        results.append("  " + "-"*50)
    return "\n".join(results)


# ======================
# 🔹 总控函数
# ======================
def full_evaluation(model_outputs: list[str], references: list[str], lang: str = "en", bert_device: str = "cpu") -> tuple[str, dict]:
    # 打印前5个样本
    N=5
    print(f"===== First {N} Samples =====")
    print(print_comparison(model_outputs[:N], references[:N]))

    comparison_str = print_comparison(model_outputs, references)
    metrics = evaluate_model_outputs(model_outputs, references, lang, bert_device)
    return comparison_str, metrics


# 只执行EM、ROUGE评估
def simple_evaluation(model_outputs: list[str], references: list[str], lang: str = "en") -> dict:
    results_dict = {}
    # --- Exact Match ---
    try:
        em_score = calculate_exact_match(model_outputs, references)
        results_dict["exact_match"] = float(em_score)
    except Exception as e:
        print(f"❌ Error calculating Exact Match: {e}")

    # --- ROUGE ---
    try:
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=model_outputs, references=references)
        for key, value in rouge_scores.items():
            results_dict[key] = float(value)
    except Exception as e:
        print(f"❌ Error calculating ROUGE: {e}")

    return results_dict

# ======================
# 🔹 示例用法
# ======================
if __name__ == "__main__":
    example_outputs = [
        "The capital of France is Paris.",
        "Python is a programming language created by Guido van Rossum."
    ]
    example_references = [
        "Paris is the capital and most populous city of France.",
        "Python is an interpreted, high-level and general-purpose programming language."
    ]
    
    print("\n===== Example Evaluation =====")
    comparison, metrics = full_evaluation(example_outputs, example_references)
    
    print("\n===== Final Evaluation Metrics =====")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
