import argparse
import json
import os
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止 tokenizers 内部多线程与后续并行冲突


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings", "text-embedding-v4"],
    )
    parser.add_argument("--dataset_type", type=str, default="synthetic_data")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


# ======================
# ✅ 优化版 embedding 计算
# ======================
def compute_embeddings_fast(model, string_list, batch_size=256, num_workers=0):
    """更快的 SentenceTransformer 批处理实现"""
    loader = DataLoader(string_list, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embeds = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Encoding ({model.device})", total=len(loader)):
            emb = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=len(batch),
            )
            all_embeds.append(emb)
    return np.concatenate(all_embeds, axis=0)


# ======================
# ✅ 主函数
# ======================
if __name__ == "__main__":
    args = parser_args()
    key_strings, value_strings = [], []

    # ---- Step 1: 读入数据并生成 reformatted JSON ----
    if args.dataset_type == "multi_wiki_qa_train":
        sid = 0
        reformatted_data = []
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line.strip()) for line in f]

        for doc in dataset:
            for triple in doc["triples"]:
                key_strings.append(triple["key_string"])
                value_strings.append(triple["description"])
            reformatted_data.append(
                json.dumps({**doc, "start_id": sid, "num_triples": len(doc["triples"])})
            )
            sid += len(doc["triples"])

        with open(args.dataset_path, "w", encoding="utf-8") as f:
            f.write("\n".join(reformatted_data))

    elif args.dataset_type == "musique":
        sid = 0
        reformatted_data = []
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for sample in dataset:
            new_paras = []
            for paragraph in sample["paragraphs"]:
                num_triples = len(paragraph["triples"])
                for triple in paragraph["triples"]:
                    key_strings.append(f"The {triple['Relation']} of {triple['Head']} is")
                    value_strings.append(triple["Tail"])
                new_paras.append({**paragraph, "start_id": sid, "num_triples": num_triples})
                sid += num_triples
            reformatted_data.append({**sample, "paragraphs": new_paras})

        with open(args.dataset_path, "w", encoding="utf-8") as f:
            json.dump(reformatted_data, f, indent=2, ensure_ascii=False)

    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    # ---- Step 2: 创建模型并计算 embeddings ----
    print(f"Computing embeddings for {len(key_strings)} triples using {args.model_name}")
    model = SentenceTransformer(args.model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # ✅ 若显存允许，可开启半精度以提速
    try:
        model = model.half()
        print("⚡ FP16 mode enabled.")
    except Exception:
        print("FP16 not supported, fallback to FP32.")


    # ✅ 一次性计算 key / value
    key_embeds = compute_embeddings_fast(model, key_strings, args.batch_size)
    value_embeds = compute_embeddings_fast(model, value_strings, args.batch_size)


    def verify_embedding_order(key_strings, key_embeds, model):
        print("🔍 Verifying embedding order consistency...")
        test_indices = list(range(5)) + list(range(len(key_strings) - 5, len(key_strings)))
        cosine = torch.nn.functional.cosine_similarity(
            torch.tensor(model.encode([key_strings[i] for i in test_indices], normalize_embeddings=True)),
            torch.tensor(key_embeds[test_indices]),
        )
        print(f"Mean similarity (first+last samples): {cosine.mean().item():.4f}")
        if cosine.mean().item() < 0.95:
            print("⚠️ WARNING: Embedding order mismatch suspected!")
        else:
            print("✅ Embedding order verified.")

    verify_embedding_order(key_strings, key_embeds, model)


    # ---- Step 3: 保存 ----
    base_name = os.path.basename(args.dataset_path).rsplit(".", 1)[0]
    output_dir = os.path.dirname(args.dataset_path) or "."
    save_name = args.model_name.replace("/", "-")

    key_path = os.path.join(output_dir, f"{base_name}_{save_name}_embd_key.npy")
    value_path = os.path.join(output_dir, f"{base_name}_{save_name}_embd_value.npy")

    np.save(key_path, key_embeds)
    np.save(value_path, value_embeds)
    print(f"✅ Saved key embeddings → {key_path}")
    print(f"✅ Saved value embeddings → {value_path}")
