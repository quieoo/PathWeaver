import torch
import zlib


_PATH_ADJ_CACHE = {}

# 这版缓存不够稳定：形状一样就重用，忽略实际邻接矩阵的内容不同
# def get_cached_adj_t(kb_adj, *, K, device, dtype):
#     key = (device, dtype, K)
#     if key not in _PATH_ADJ_CACHE:
#         adj = kb_adj.coalesce().to(device=device, dtype=dtype)
#         if adj.size(0) != K:
#             adj = adj[:K, :K]
#         adj_t = adj.transpose(0, 1).to_sparse_csr()
#         _PATH_ADJ_CACHE[key] = adj_t
#     return _PATH_ADJ_CACHE[key]

_PATH_ADJ_CACHE = {}

def _adj_fingerprint_coo(kb_adj: torch.Tensor, K: int) -> int:
    """
    Return a stable fingerprint for a sparse COO adj.
    We hash a small, deterministic view of (indices, values, shape).
    """
    adj = kb_adj.coalesce()
    if adj.size(0) != K:
        adj = adj[:K, :K].coalesce()

    idx = adj.indices()
    val = adj.values()

    # 为了避免 hash 成本太高：只取前/后若干元素（对链式图也足够区分）
    # 如果你担心碰撞，把 sample 改大一点，比如 4096
    s = min(idx.size(1), 2048)
    if s < idx.size(1):
        sel = torch.cat([torch.arange(s//2), torch.arange(idx.size(1)-s//2, idx.size(1))]).to(idx.device)
        idx = idx[:, sel]
        val = val[sel]

    # 转 CPU bytes，做 crc32（快、稳定）
    b = bytearray()
    b += int(adj.size(0)).to_bytes(4, "little", signed=False)
    b += int(adj.size(1)).to_bytes(4, "little", signed=False)
    b += idx.detach().cpu().to(torch.int64).numpy().tobytes()
    # values 只用来区分（你的 values 基本全 1，也可以不加）
    b += val.detach().cpu().to(torch.float32).numpy().tobytes()
    return zlib.crc32(b)

def get_cached_adj_t(kb_adj, *, K, device, dtype):
    fp = _adj_fingerprint_coo(kb_adj, K)
    key = (device, dtype, K, fp)

    if key not in _PATH_ADJ_CACHE:
        adj = kb_adj.coalesce().to(device=device, dtype=dtype)
        if adj.size(0) != K:
            adj = adj[:K, :K].coalesce()
        adj_t = adj.transpose(0, 1).to_sparse_csr()
        _PATH_ADJ_CACHE[key] = adj_t

    return _PATH_ADJ_CACHE[key]


def apply_kblam_path_attention(
    *,
    attn_weights: torch.Tensor,   # (B,H,Q,K_total)
    kb_len: int,
    kb_adj: torch.Tensor,
    kb_config,
    layer_idx: int,
):
    """
    完整复用 llama3_model.py 中的 path_attn 逻辑
    """
    if not (
        kb_config.path_attn
        and kb_adj is not None
        and layer_idx % kb_config.kb_layer_frequency == 0
    ):
        return attn_weights

    alpha_kb = attn_weights[..., :kb_len]
    B, H, Q, K = alpha_kb.shape

    if kb_adj.is_sparse:
        idx = kb_adj.indices()
        if idx.size(0) == 2:
            M = B * H * Q
            alpha2 = alpha_kb.reshape(M, K)
            adj_t = get_cached_adj_t(
                kb_adj,
                K=K,
                device=alpha_kb.device,
                dtype=alpha_kb.dtype,
            )
            # beta2 = torch.sparse.mm(adj_t, alpha2.transpose(0, 1))
            beta2 = torch.sparse.mm(
                adj_t.float(),
                alpha2.transpose(0, 1).float(),
            )
            beta2 = beta2.to(alpha_kb.dtype)
            
            beta_kb = beta2.transpose(0, 1).reshape(B, H, Q, K)
        else:
            # 非共享图（原逻辑）
            beta_chunks = []
            alpha_flat = alpha_kb.reshape(B, -1, K)
            vals = kb_adj.values()
            for b in range(B):
                mask = idx[0] == b
                if mask.any():
                    rows = idx[1, mask]
                    cols = idx[2, mask]
                    adj_b = torch.sparse_coo_tensor(
                        torch.stack([rows, cols]),
                        vals[mask],
                        (K, K),
                        device=vals.device,
                        dtype=alpha_kb.dtype,
                    ).coalesce()
                else:
                    adj_b = torch.sparse_coo_tensor(
                        torch.empty((2, 0), device=alpha_kb.device, dtype=torch.long),
                        torch.empty((0,), device=alpha_kb.device, dtype=alpha_kb.dtype),
                        (K, K),
                    )
                # beta_flat = torch.sparse.mm(
                #     adj_b.transpose(0, 1),
                #     alpha_flat[b].transpose(0, 1),
                # ).transpose(0, 1)

                # FIX: bfloat16不支持sparse mm
                beta_flat = torch.sparse.mm(
                    adj_b.transpose(0, 1).float(),
                    alpha_flat[b].transpose(0, 1).float(),
                ).transpose(0, 1)
                beta_flat = beta_flat.to(alpha_kb.dtype)

                beta_chunks.append(beta_flat.view(H, Q, K))
            beta_kb = torch.stack(beta_chunks, dim=0)
    else:
        A = kb_adj.to(device=alpha_kb.device, dtype=alpha_kb.dtype)
        beta_kb = alpha_kb @ A

    mix_ratio = getattr(kb_config, "path_attn_mix_ratio", 1.0)
    beta_kb = mix_ratio * beta_kb + (1.0 - mix_ratio) * alpha_kb

    beta_kb = beta_kb / beta_kb.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    other = attn_weights[..., kb_len:]
    other_sum = other.sum(dim=-1, keepdim=True)
    denom = (1.0 + other_sum).clamp_min(1e-9)

    return torch.cat(
        [beta_kb / denom, other / denom],
        dim=-1,
    )
