import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
from sentence_transformers import SentenceTransformer


# =============== 工具函数 ===============

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def compute_similarity_matrix(q_embs, k_embs, chunk_size=2048, device="cuda"):
    """
    分块计算相似度矩阵，避免OOM
    所有结果存放在 CPU 上 (float32)
    """
    q_embs = normalize(q_embs).to(device)
    k_embs = normalize(k_embs).to(device)
    N = q_embs.size(0)

    sim_matrix = torch.empty((N, N), dtype=torch.float32)

    for i in tqdm(range(0, N, chunk_size), desc="Computing similarity (chunked)"):
        q_chunk = q_embs[i:i + chunk_size]
        sim_chunk = (q_chunk @ k_embs.T).cpu()
        sim_matrix[i:i + chunk_size] = sim_chunk
        del q_chunk, sim_chunk
        torch.cuda.empty_cache()

    return sim_matrix


def retrieval_metrics(sim_matrix: torch.Tensor, chunk_size: int = 5000):
    """
    高效计算Top1 / Top5 / MRR（分块排序，支持大规模矩阵）
    """
    N = sim_matrix.size(0)
    ranks_all = []

    for i in tqdm(range(0, N, chunk_size), desc="Evaluating retrieval metrics (chunked)"):
        end = min(i + chunk_size, N)
        sims_chunk = sim_matrix[i:end].to(torch.float32)
        sorted_idx = torch.argsort(sims_chunk, dim=1, descending=True)
        target = torch.arange(i, end).unsqueeze(1)
        rank = (sorted_idx == target).nonzero(as_tuple=False)[:, 1] + 1
        ranks_all.append(rank)
        del sims_chunk, sorted_idx, target, rank
        torch.cuda.empty_cache()

    ranks = torch.cat(ranks_all).numpy()
    mrr = float(np.mean(1.0 / ranks))
    top1 = float(np.mean(ranks == 1))
    top5 = float(np.mean(ranks <= 5))
    return {"MRR": mrr, "Top1": top1, "Top5": top5}


# =============== 编码器封装 ===============

class BaseTextEncoder:
    """原始文本空间编码器（无投影）"""
    def __init__(self, model_dir: str, device="cuda"):
        self.device = device
        self.model = SentenceTransformer(model_dir, device=device)

    @torch.no_grad()
    def encode(self, texts, batch_size=512):
        """编码文本（分块+进度条）"""
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding (base encoder)"):
            batch_texts = texts[i:i + batch_size]
            emb = self.model.encode(
                batch_texts, convert_to_tensor=True, device=self.device
            )
            all_embs.append(emb)
        return torch.cat(all_embs, dim=0)


class KBTrainedEncoder(torch.nn.Module):
    """加载训练后的 Encoder.pt，输出 projector_k 之后的 embedding"""
    def __init__(self, base_encoder_name, trained_encoder_path: str, out_dim: int, device="cuda"):
        super().__init__()
        from kblam.kb_encoder import KBEncoder
        self.device = device
        self.encoder = KBEncoder(
            encoder_name=base_encoder_name,
            projector_type="linear",
            endpoint_url="",
            out_dim=out_dim,
            frozen_base_model=True,
            device=torch.device(device),
        )
        self.encoder.load_state_dict(torch.load(trained_encoder_path, map_location=device))
        self.encoder.eval()

    @torch.no_grad()
    def encode_qk(self, embeds, batch_size=64):
        """
        分块编码，结果转CPU释放显存
        """
        all_embs = []
        for i in tqdm(range(0, len(embeds), batch_size), desc="Encoding (trained encoder)"):
            chunk = embeds[i:i + batch_size]
            if isinstance(chunk, np.ndarray):
                chunk = torch.from_numpy(chunk)
            elif not isinstance(chunk, torch.Tensor):
                raise TypeError(f"Unexpected type for chunk: {type(chunk)}")

            chunk = chunk.to(self.device).float()
            emb = self.encoder.key_layernorm(self.encoder.projector_k(chunk)).bfloat16()
            all_embs.append(emb.cpu())
            del emb, chunk
            torch.cuda.empty_cache()

        return torch.cat(all_embs, dim=0)


# =============== 主流程 ===============

def main(dataset_path, base_encoder, trained_encoder_path, out_dim, device="cuda", post_sample=-1):
    print(f"[INFO] Loading dataset from {dataset_path}")
    raw_data = json.load(open(dataset_path))

    # 随机采样
    if post_sample > 0:
        post_sample = min(post_sample, len(raw_data))
        sample_ids = np.random.choice(len(raw_data), post_sample, replace=False)
        data = [raw_data[i] for i in sample_ids]
    else:
        data = raw_data

    # 过滤无效样本
    data = [d for d in data if d.get("Q") and d.get("key_string")]
    Qs = [d["Q"] for d in data]
    Ks = [d["key_string"] for d in data]
    print(f"[INFO] Loaded {len(data)} valid samples")

    # ---- 原始空间 ----
    base_enc = BaseTextEncoder(base_encoder, device=device)
    print("[Stage 1] Encoding in original text space...")
    q_base = base_enc.encode(Qs)
    k_base = base_enc.encode(Ks)

    print("[Stage 1] Computing similarity matrix...")
    sim_pre = compute_similarity_matrix(q_base, k_base)
    metrics_pre = retrieval_metrics(sim_pre)
    print(f"[Result - PreEncoder] MRR={metrics_pre['MRR']:.4f} | Top1={metrics_pre['Top1']:.4f} | Top5={metrics_pre['Top5']:.4f}")

    # ---- 训练后 Key 空间 ----
    kb_enc = KBTrainedEncoder(base_encoder, trained_encoder_path, out_dim, device=device)
    print("[Stage 2] Encoding in trained key space...")
    q_post = kb_enc.encode_qk(q_base)
    k_post = kb_enc.encode_qk(k_base)

    print("[Stage 2] Computing similarity matrix...")
    sim_post = compute_similarity_matrix(q_post, k_post)
    metrics_post = retrieval_metrics(sim_post)
    print(f"[Result - PostKey]    MRR={metrics_post['MRR']:.4f} | Top1={metrics_post['Top1']:.4f} | Top5={metrics_post['Top5']:.4f}")

    # ---- 相似度统计 ----
    sim_diag_pre = sim_pre.diag().cpu().numpy()
    sim_diag_post = sim_post.diag().cpu().numpy()
    print(f"[Average diag cos-sim] Pre={sim_diag_pre.mean():.4f}, Post={sim_diag_post.mean():.4f}")

    rho, _ = spearmanr(sim_diag_pre, sim_diag_post)
    tau, _ = kendalltau(sim_diag_pre, sim_diag_post)
    print(f"[Spearman rho] {rho:.4f} | [Kendall tau] {tau:.4f}")

    print("\n========== [SUMMARY] ==========")
    print(f"Samples = {len(data)} | Dim = {out_dim}")
    print(f"Pre:  MRR={metrics_pre['MRR']:.4f}, Top1={metrics_pre['Top1']:.4f}, Top5={metrics_pre['Top5']:.4f}")
    print(f"Post: MRR={metrics_post['MRR']:.4f}, Top1={metrics_post['Top1']:.4f}, Top5={metrics_post['Top5']:.4f}")
    print(f"DiagCos: Pre={sim_diag_pre.mean():.4f}, Post={sim_diag_post.mean():.4f}")
    print(f"RankCorr: Spearman={rho:.4f}, Kendall={tau:.4f}")
    print("================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate semantic alignment (Pre vs PostKey)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSON file containing Q and key_string")
    parser.add_argument("--base_encoder", type=str, required=True, help="SentenceTransformer model path or name")
    parser.add_argument("--trained_encoder_path", type=str, required=True, help="Path to trained encoder .pt")
    parser.add_argument("--out_dim", type=int, default=135168, help="Output dimension of the trained encoder")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--post_sample", type=int, default=-1, help="Sample size for post-encoder evaluation (-1 for all)")
    args = parser.parse_args()

    main(args.dataset_path, args.base_encoder, args.trained_encoder_path, args.out_dim, args.device, args.post_sample)
