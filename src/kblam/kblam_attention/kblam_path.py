import torch


_PATH_ADJ_CACHE = {}


def get_cached_adj_t(kb_adj, *, K, device, dtype):
    key = (device, dtype, K)
    if key not in _PATH_ADJ_CACHE:
        adj = kb_adj.coalesce().to(device=device, dtype=dtype)
        if adj.size(0) != K:
            adj = adj[:K, :K]
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
            beta2 = torch.sparse.mm(adj_t, alpha2.transpose(0, 1))
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
                beta_flat = torch.sparse.mm(
                    adj_b.transpose(0, 1),
                    alpha_flat[b].transpose(0, 1),
                ).transpose(0, 1)
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
