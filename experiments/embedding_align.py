import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
from kblam.kb_encoder import KBEncoder
import torch


# ======= 1. Load dataset (2wiki format) =======
DATASET_PATH = "/mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json"

# ======= 2. Load embeddings computed by embedding_v2.py =======
KEY_EMB_PATH = "/mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy"
VAL_EMB_PATH = "/mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy"

# SBERT / QWen encoder for approximating Q-embedding
QUERY_ENCODER_PATH = "/mnt/n0/models/qwen-embedding-0.6B"

ENCODER_PATH="/mnt/n0/KBLAM/KBLaM/experiments/train/2wiki1_1.0/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_6800_encoder/"

ENCODER_NAME="qwen-embedding-0.6B"
OUTDIM=4096*(32//3+1)

# =========================================================
# ====== 2. Load Dataset ==================================
# =========================================================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


# =========================================================
# ====== 3. Load KB Embeddings =============================
# =========================================================
key_embds = np.load(KEY_EMB_PATH)     # shape [N, d]
val_embds = np.load(VAL_EMB_PATH)     # shape [N, d]

print(f"Loaded key_embds: {key_embds.shape}, val_embds: {val_embds.shape}")


# =========================================================
# ====== 4. Precompute Triple Offsets ======================
# =========================================================
triple_offset = []
offset = 0
for sample in dataset:
    triple_offset.append(offset)
    offset += len(sample["triple_lists"])


# Helper: cosine similarity
def cos(a, b):
    return cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0]


def compute_encode_similarity(sample_id):
    encoder = KBEncoder(
        encoder_name=ENCODER_NAME.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=OUTDIM,
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
    )

    sample = dataset[sample_id]
    start = triple_offset[sample_id]

    t1 = sample["triple_lists"][0]   # first-hop triple
    t2 = sample["triple_lists"][1]   # second-hop triple

    idx1_key = start + 0
    idx1_val = start + 0
    idx2_key = start + 1
    idx2_val = start + 1

    key2_emb  = key_embds[idx2_key]
    key1_emb  = key_embds[idx1_key]
    val1_emb  = val_embds[idx1_val]
    val2_emb  = val_embds[idx2_val]

    encode_key1 = encoder.encode_key(base_emb=key1_emb)
    encode_val1 = encoder.encode_val(base_emb=val1_emb)
    encode_key2 = encoder.encode_key(base_emb=key2_emb)
    encode_val2 = encoder.encode_val(base_emb=val2_emb)


    encode_key1 = encode_key1.cpu().to(torch.float32).detach().numpy()
    encode_val1 = encode_val1.cpu().to(torch.float32).detach().numpy()
    encode_key2 = encode_key2.cpu().to(torch.float32).detach().numpy()
    encode_val2 = encode_val2.cpu().to(torch.float32).detach().numpy()


    print(f"sim(encode_val1, encode_key2) = {cos(encode_val1, encode_key2):.4f}")
    print(f"sim(encode_key1, encode_val1) = {cos(encode_key1, encode_val1):.4f}")
    print(f"sim(encode_val2, encode_key2) = {cos(encode_val2, encode_key2):.4f}")
    


def diagnose_sample(sample_id):

    QUERY_ENCODER = SentenceTransformer(QUERY_ENCODER_PATH)


    sample = dataset[sample_id]
    start = triple_offset[sample_id]

    t1 = sample["triple_lists"][0]   # first-hop triple
    t2 = sample["triple_lists"][1]   # second-hop triple

    idx1_key = start + 0
    idx1_val = start + 0
    idx2_key = start + 1
    idx2_val = start + 1

    desc1_emb = val_embds[idx1_val]
    key2_emb  = key_embds[idx2_key]
    key1_emb  = key_embds[idx1_key]
    val1_emb  = val_embds[idx1_val]
    val2_emb  = val_embds[idx2_val]

    print("\n==================================================")
    print(f"Sample #{sample_id}")
    print("Triple1:", t1)
    print("Triple2:", t2)

    # ------------------------------------------------------
    # 5.1 Local similarities
    # ------------------------------------------------------
    print("\n---- [1] Local Similarity Diagnostics ----")
    print(f"sim(desc1, key2)        = {cos(desc1_emb, key2_emb):.4f}")
    print(f"sim(desc1, key1)        = {cos(desc1_emb, key1_emb):.4f}")
    print(f"sim(key1, key2)         = {cos(key1_emb,  key2_emb):.4f}")
    print(f"sim(value1, value2)     = {cos(val1_emb,  val2_emb):.4f}")
    print(f"sim(desc1, desc1)       = {cos(desc1_emb, desc1_emb):.4f}")

    # ------------------------------------------------------
    # 5.2 desc1 vs ALL keys (rank analysis)
    # ------------------------------------------------------
    print("\n---- [2] desc1 vs ALL key_embds Rank ----")
    sims = cosine_similarity(desc1_emb.reshape(1,-1), key_embds)[0]
    rank = (-sims).argsort().tolist().index(idx2_key) + 1   # 1-based rank
    print(f"(desc1 -> key2) rank among ALL keys = {rank} / {len(key_embds)}")
    print(f"Top-5 similar key indices: {np.argsort(-sims)[:5]}")

    # ------------------------------------------------------
    # 5.3 (Optional) Q embedding (using SBERT)
    # ------------------------------------------------------
    query_text = sample["Q"]
    q_emb = QUERY_ENCODER.encode(query_text)

    print("\n---- [3] Q Embedding vs key1/key2 ----")
    print(f"sim(Q, key1) = {cos(q_emb, key1_emb):.4f}")
    print(f"sim(Q, key2) = {cos(q_emb, key2_emb):.4f}")

    print("==================================================\n")

    print("\n---- [4] Q vs ALL key_embds Rank (key1 position) ----")
    sims_q = cosine_similarity(q_emb.reshape(1,-1), key_embds)[0]
    rank_q_key1 = (-sims_q).argsort().tolist().index(idx1_key) + 1
    print(f"(Q -> key1) rank among ALL keys = {rank_q_key1} / {len(key_embds)}")
    print(f"Top-5 similar to Q: {np.argsort(-sims_q)[:5]}")

    print("==================================================\n")


def compute_q_key1_ranks(dataset, key_embds, triple_offset, QUERY_ENCODER, batch_size=128):
    """
    计算所有样本的 (Q vs ALL key_embds) 的排名 (key1 的位置)
    并绘制 CDF。
    """

    num_samples = len(dataset)
    num_keys = key_embds.shape[0]
    dim = key_embds.shape[1]

    print(f"Total samples: {num_samples}, KB size: {num_keys}, dim: {dim}")

    # -------------------------------
    # 1. 预编码所有 Query 的 embedding（批处理）
    # -------------------------------
    print("\nEncoding all Queries into embeddings (batched)...")

    Q_list = [dataset[i]["Q"] for i in range(num_samples)]
    Q_embds = np.zeros((num_samples, dim), dtype=np.float32)

    for start in tqdm(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)
        batch = Q_list[start:end]
        Q_emb = QUERY_ENCODER.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        Q_embds[start:end] = Q_emb.astype(np.float32)

    # -------------------------------
    # 2. 构建 key1 索引列表
    # -------------------------------
    key1_indices = [triple_offset[i] for i in range(num_samples)]
    key1_indices = np.array(key1_indices)

    # -------------------------------
    # 3. 批量计算 Q vs ALL key 相似度 + 排名
    # -------------------------------
    print("\nComputing Q vs ALL keys rank (batched)...")

    ranks = np.zeros(num_samples, dtype=np.int32)

    # 预归一化 key_embds（若未归一化）
    key_norm = key_embds / (np.linalg.norm(key_embds, axis=1, keepdims=True) + 1e-9)
    Q_norm = Q_embds / (np.linalg.norm(Q_embds, axis=1, keepdims=True) + 1e-9)

    for start in tqdm(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)

        # (B, d) @ (d, N) → (B, N)
        sims = Q_norm[start:end] @ key_norm.T

        # 获取 key1 的 true similarity
        true_scores = sims[np.arange(end-start), key1_indices[start:end]]

        # 排序并计算 rank（越大越前）
        sorted_idx = np.argsort(-sims, axis=1)

        # 找到 key1 的排名位置
        for i in range(end-start):
            ranks[start+i] = np.where(sorted_idx[i] == key1_indices[start+i])[0][0] + 1

    print("\nFinished. Example ranks:", ranks[:10])

    # -------------------------------
    # 4. 绘制 CDF 曲线
    # -------------------------------
    print("Plotting CDF for key1 rank...")

    sorted_ranks = np.sort(ranks)
    cdf_y = np.arange(1, num_samples+1) / num_samples

    plt.figure(figsize=(8,6))
    plt.plot(sorted_ranks, cdf_y)
    plt.xscale("log")
    plt.xlabel("Rank of key1 (log scale)")
    plt.ylabel("CDF")
    plt.title("CDF of Q vs ALL key_embds Rank (key1 position)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    # plt.show()

    # 保存
    plt.savefig("q_key1_rank_cdf.png", dpi=300, bbox_inches="tight")

    return ranks

# =========================================================
# ====== 6. Run diagnostics for first few samples ==========
# =========================================================
for i in range(5):
    # diagnose_sample(i)
    compute_encode_similarity(i)



# =========================================================
# ====== Evaluate Q vs ALL key_embds Rank (key1 position) ==========
# =========================================================

# ranks = compute_q_key1_ranks(
#     dataset=dataset,
#     key_embds=key_embds,
#     triple_offset=triple_offset,
#     QUERY_ENCODER=QUERY_ENCODER,
# )

