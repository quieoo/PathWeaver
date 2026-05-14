import torch
import zlib


# -----------------------------------------------------------------------------
# Path attention tracing utilities
# -----------------------------------------------------------------------------
# Usage:
#   from kblam_path import enable_path_attn_trace, set_path_attn_trace_context,
#       dump_path_attn_trace
#   enable_path_attn_trace(True, store_raw=True)
#   set_path_attn_trace_context(sample_id="...")
#   ... run generation/eval ...
#   dump_path_attn_trace("/tmp/path_attn_trace.pt")
#
# The trace stores every KB token's attention before and after path propagation.
# It does NOT need answer-KV labels; those can be joined offline later.
_PATH_ATTN_TRACE = {
    "enabled": False,
    "records": [],
    "context": {},
    "store_raw": True,
    "store_kb_normalized": True,
    "cpu_dtype": torch.float32,
    "max_records": None,
}


def is_path_attn_trace_enabled() -> bool:
    """Return whether path-attention tracing is currently enabled."""
    return bool(_PATH_ATTN_TRACE["enabled"])


def enable_path_attn_trace(
    enabled: bool = True,
    *,
    store_raw: bool = True,
    store_kb_normalized: bool = True,
    cpu_dtype: torch.dtype = torch.float32,
    max_records: int | None = None,
):
    """
    Enable/disable collection of KB attention before/after DAG path propagation.

    Args:
        enabled: Turn tracing on/off.
        store_raw: Store raw KB attention slices alpha_kb / beta_kb_final.
            Shapes are (B, H, Q, K). This is the most complete but largest mode.
        store_kb_normalized: Store KB-normalized attention slices. These are usually
            the best tensors for answer-KV rank / top-k recall analysis because the
            comparison is only among KB tokens.
        cpu_dtype: dtype used after moving traces to CPU.
        max_records: Optional cap to avoid unbounded memory growth.
    """
    _PATH_ATTN_TRACE["enabled"] = bool(enabled)
    _PATH_ATTN_TRACE["store_raw"] = bool(store_raw)
    _PATH_ATTN_TRACE["store_kb_normalized"] = bool(store_kb_normalized)
    _PATH_ATTN_TRACE["cpu_dtype"] = cpu_dtype
    _PATH_ATTN_TRACE["max_records"] = max_records


def clear_path_attn_trace():
    """Remove all in-memory path-attention trace records."""
    _PATH_ATTN_TRACE["records"].clear()


def set_path_attn_trace_context(**metadata):
    """
    Set metadata that will be copied into every subsequently collected record.

    Typical fields:
        sample_id, dataset, question, generation_step, prompt_len, etc.
    """
    if not is_path_attn_trace_enabled():
        return
    _PATH_ATTN_TRACE["context"].clear()
    _PATH_ATTN_TRACE["context"].update(metadata)


def update_path_attn_trace_context(**metadata):
    """Update metadata copied into subsequently collected trace records."""
    if not is_path_attn_trace_enabled():
        return
    _PATH_ATTN_TRACE["context"].update(metadata)


def backfill_path_attn_trace_records(**metadata):
    """
    Update already-collected records that belong to the current trace context.

    This is useful for metadata that is only known after generation finishes,
    such as the model's final decoded output.
    """
    if not is_path_attn_trace_enabled():
        return

    matcher = {
        k: v for k, v in _PATH_ATTN_TRACE["context"].items() if k not in metadata
    }
    if not matcher:
        return

    for record in reversed(_PATH_ATTN_TRACE["records"]):
        record_context = record.get("context")
        if not isinstance(record_context, dict):
            continue
        if all(record_context.get(k) == v for k, v in matcher.items()):
            record_context.update(metadata)


def get_path_attn_trace_records():
    """Return the live in-memory records list."""
    return _PATH_ATTN_TRACE["records"]


def _to_trace_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(dtype=_PATH_ATTN_TRACE["cpu_dtype"]).cpu()


def _kb_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _maybe_collect_path_attn_trace(
    *,
    alpha_kb: torch.Tensor,
    beta_kb_final: torch.Tensor,
    kb_len: int,
    kb_adj,
    kb_config,
    layer_idx: int,
):
    """
    Collect one path-attention trace record.

    alpha_kb: KB attention before path propagation, shape (B,H,Q,K).
    beta_kb_final: KB attention after path propagation and after the same final
        renormalization used by apply_kblam_path_attention, shape (B,H,Q,K).
    """
    if not _PATH_ATTN_TRACE["enabled"]:
        return

    max_records = _PATH_ATTN_TRACE["max_records"]
    records = _PATH_ATTN_TRACE["records"]
    if max_records is not None and len(records) >= max_records:
        return

    with torch.no_grad():
        record = {
            "layer_idx": int(layer_idx),
            "kb_len": int(kb_len),
            "shape": tuple(alpha_kb.shape),
            "path_attn_mix_ratio": float(getattr(kb_config, "path_attn_mix_ratio", 1.0)),
            "kb_layer_frequency": int(getattr(kb_config, "kb_layer_frequency", 1)),
            "context": dict(_PATH_ATTN_TRACE["context"]),
        }

        # Store a lightweight graph fingerprint so traces can be grouped by graph.
        try:
            if kb_adj is not None and getattr(kb_adj, "is_sparse", False):
                record["adj_fingerprint"] = int(_adj_fingerprint_coo(kb_adj, kb_len))
                record["adj_nnz"] = int(kb_adj.coalesce()._nnz())
            elif kb_adj is not None:
                record["adj_fingerprint"] = None
                record["adj_nnz"] = int((kb_adj != 0).sum().item())
        except Exception:
            record["adj_fingerprint"] = None
            record["adj_nnz"] = None

        if _PATH_ATTN_TRACE["store_raw"]:
            record["alpha_kb"] = _to_trace_cpu(alpha_kb)
            record["beta_kb"] = _to_trace_cpu(beta_kb_final)

        if _PATH_ATTN_TRACE["store_kb_normalized"]:
            record["alpha_kb_norm"] = _to_trace_cpu(_kb_normalize(alpha_kb))
            record["beta_kb_norm"] = _to_trace_cpu(_kb_normalize(beta_kb_final))

        records.append(record)


def dump_path_attn_trace(save_path: str, *, clear: bool = True):
    """
    Persist collected path-attention traces to disk with torch.save.

    Returns:
        save_path
    """
    payload = {
        "records": _PATH_ATTN_TRACE["records"],
        "num_records": len(_PATH_ATTN_TRACE["records"]),
        "store_raw": _PATH_ATTN_TRACE["store_raw"],
        "store_kb_normalized": _PATH_ATTN_TRACE["store_kb_normalized"],
    }
    torch.save(payload, save_path)
    if clear:
        clear_path_attn_trace()
    return save_path


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


def _apply_kblam_path_attention_impl(
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

    # mix_ratio = getattr(kb_config, "path_attn_mix_ratio", 1.0)
    mix_ratio = getattr(kb_config, "path_attn_mix_ratio", 0.7)
    beta_kb = mix_ratio * beta_kb + (1.0 - mix_ratio) * alpha_kb

    beta_kb = beta_kb / beta_kb.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    other = attn_weights[..., kb_len:]
    other_sum = other.sum(dim=-1, keepdim=True)
    denom = (1.0 + other_sum).clamp_min(1e-9)

    return torch.cat(
        [beta_kb / denom, other / denom],
        dim=-1,
    )


def apply_kblam_path_attention(
    *,
    attn_weights: torch.Tensor,   # (B,H,Q,K_total)
    kb_len: int,
    kb_adj: torch.Tensor,
    kb_config,
    layer_idx: int,
):
    """
    Wrapper around the original DAG path-attention implementation.

    When tracing is disabled, this is equivalent to the original implementation.
    When tracing is enabled via enable_path_attn_trace(...), it stores all KB-token
    attention scores before and after path propagation for offline structural
    reasoning analysis.
    """
    should_trace = (
        _PATH_ATTN_TRACE["enabled"]
        and kb_config is not None
        and getattr(kb_config, "path_attn", False)
        and kb_adj is not None
        and layer_idx % getattr(kb_config, "kb_layer_frequency", 1) == 0
        and kb_len > 0
    )

    if should_trace:
        alpha_kb = attn_weights[..., :kb_len]

    out = _apply_kblam_path_attention_impl(
        attn_weights=attn_weights,
        kb_len=kb_len,
        kb_adj=kb_adj,
        kb_config=kb_config,
        layer_idx=layer_idx,
    )

    if should_trace:
        beta_kb_final = out[..., :kb_len]
        _maybe_collect_path_attn_trace(
            alpha_kb=alpha_kb,
            beta_kb_final=beta_kb_final,
            kb_len=kb_len,
            kb_adj=kb_adj,
            kb_config=kb_config,
            layer_idx=layer_idx,
        )

    return out
