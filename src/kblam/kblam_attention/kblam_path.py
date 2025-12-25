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
    attn_weights: torch.Tensor,     # (B, H, Q, K_total)
    kb_len: int,
    kb_adj: torch.Tensor,           # sparse COO, shape (B?, K, K)
    kb_config,
    layer_idx: int,
):
    """
    Path-based KB attention propagation (post-softmax).

    核心约束：
    - torch.sparse.mm **必须在 fp32 上执行**
    - 结果再 cast 回原 dtype
    """

    # ------------------------------------------------------------
    # 0. 条件检查
    # ------------------------------------------------------------
    if not (
        kb_config.path_attn
        and kb_adj is not None
        and layer_idx % kb_config.kb_layer_frequency == 0
    ):
        return attn_weights

    # ------------------------------------------------------------
    # 1. 取 KB attention 子块
    # ------------------------------------------------------------
    # alpha_kb: (B, H, Q, K)
    alpha_kb = attn_weights[..., :kb_len]
    B, H, Q, K = alpha_kb.shape
    orig_dtype = alpha_kb.dtype
    device = alpha_kb.device

    # ------------------------------------------------------------
    # 2. flatten 成 (B, K, H*Q)
    # ------------------------------------------------------------
    alpha_flat = (
        alpha_kb
        .permute(0, 3, 1, 2)          # (B, K, H, Q)
        .reshape(B, K, H * Q)         # (B, K, HQ)
    )

    # ------------------------------------------------------------
    # 3. sparse adjacency 处理
    # ------------------------------------------------------------
    if not kb_adj.is_sparse:
        raise RuntimeError("apply_kblam_path_attention expects sparse kb_adj")

    idx = kb_adj.indices()

    # ============================================================
    # Case A: 共享图 (idx.size(0) == 2)
    # ============================================================
    if idx.size(0) == 2:
        # --- cache key ---
        cache_key = (device, K)

        if cache_key not in _PATH_ADJ_CACHE:
            adj = kb_adj.coalesce().to(device=device, dtype=torch.float32)
            adj_t = adj.transpose(0, 1).to_sparse_csr()
            _PATH_ADJ_CACHE[cache_key] = adj_t
        else:
            adj_t = _PATH_ADJ_CACHE[cache_key]

        # --- sparse mm (fp32) ---
        beta_flat = torch.sparse.mm(
            adj_t,
            alpha_flat.reshape(K, B * H * Q).float(),
        )
        beta_flat = beta_flat.reshape(K, B, H, Q).permute(1, 0, 2, 3)
        beta_flat = beta_flat.reshape(B, K, H * Q)

    # ============================================================
    # Case B: 非共享图 (idx.size(0) == 3)
    # ============================================================
    else:
        beta_flat = torch.zeros(
            (B, K, H * Q),
            device=device,
            dtype=torch.float32,   # ← 直接 fp32
        )

        for b in range(B):
            # 构造 batch b 的 sparse adjacency
            mask = idx[0] == b
            sub_idx = idx[1:, mask]
            sub_val = kb_adj.values()[mask]

            adj_b = torch.sparse_coo_tensor(
                sub_idx,
                sub_val,
                size=(K, K),
                device=device,
                dtype=torch.float32,   # ← 强制 fp32
            ).coalesce()

            # sparse mm（fp32）
            beta_b = torch.sparse.mm(
                adj_b.transpose(0, 1),
                alpha_flat[b].float(),
            )
            beta_flat[b] = beta_b

    # ------------------------------------------------------------
    # 4. reshape 回 (B, H, Q, K)
    # ------------------------------------------------------------
    beta_kb = (
        beta_flat
        .reshape(B, K, H, Q)
        .permute(0, 2, 3, 1)   # (B, H, Q, K)
    )

    # ------------------------------------------------------------
    # 5. 与原 attention 混合（残差）
    # ------------------------------------------------------------
    alpha_kb_fp32 = alpha_kb.float()
    beta_kb = beta_kb + alpha_kb_fp32

    # ------------------------------------------------------------
    # 6. 重新归一化（只对 KB 部分）
    # ------------------------------------------------------------
    beta_kb = beta_kb / (beta_kb.sum(dim=-1, keepdim=True) + 1e-9)

    # ------------------------------------------------------------
    # 7. cast 回原 dtype，拼回完整 attention
    # ------------------------------------------------------------
    beta_kb = beta_kb.to(orig_dtype)

    attn_weights = torch.cat(
        [beta_kb, attn_weights[..., kb_len:]],
        dim=-1,
    )

    return attn_weights