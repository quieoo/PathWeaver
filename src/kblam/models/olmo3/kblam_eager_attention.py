import math
import torch
import torch.nn as nn
from typing import Optional

from transformers.utils.generic import TransformersKwargs
from transformers.models.llama.modeling_llama import repeat_kv

from kblam.models.kblam_config import KBLaMConfig
from kblam.kblam_attention import (
    apply_kblam_attention,
    apply_kblam_path_attention,
)
from kblam.kblam_attention.kblam_injector import apply_kblam_sep_query_head


def kblam_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    *,
    # ===== KBLaM 扩展参数 =====
    query_states_2: Optional[torch.Tensor] = None,
    kb_kvs: Optional[tuple] = None,
    kb_config: Optional[KBLaMConfig] = None,
    kb_adj: Optional[torch.Tensor] = None,
    **kwargs: TransformersKwargs,
):
    """
    等价于 HF eager_attention_forward + KBLaM 注入

    query/key/value:
        (B, H, Q, D) / (B, H_kv, K, D) before repeat_kv
    """

    # ------------------------------------------------------------
    # 0. repeat KV（HF 原逻辑）
    # ------------------------------------------------------------
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # ------------------------------------------------------------
    # 1. ⭐ KBLaM Hook 1：prepend KB KV + 扩 attention_mask
    #    （完全复用你 llama3 已验证的 injector）
    # ------------------------------------------------------------
    if kb_kvs is not None and kb_config is not None:
        key_states, value_states, attention_mask, kb_len = apply_kblam_attention(
            query_states=query,
            query_states_2=query_states_2,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            kb_kvs=kb_kvs,
            kb_config=kb_config,
            kb_adj=kb_adj,
            layer_idx=int(module.layer_idx),
            num_heads=int(module.num_heads),
            head_dim=int(module.head_dim),
            num_hidden_layers=int(module.config.num_hidden_layers),
        )
    else:
        kb_len = 0

    # ------------------------------------------------------------
    # 2. attention logits（HF 原逻辑）
    # ------------------------------------------------------------
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # ------------------------------------------------------------
    # 3. mask（HF 原逻辑）
    # ------------------------------------------------------------
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # ------------------------------------------------------------
    # 4. ⭐ KBLaM Hook 2：sep_query_head（softmax 前）
    # ------------------------------------------------------------
    if (
        kb_len > 0
        and kb_config is not None
        and kb_config.sep_query_head
        and query_states_2 is not None
    ):
        kb_keys = key_states[..., :kb_len, :]  # (B,H,kb_len,D)
        attn_weights = apply_kblam_sep_query_head(
            attn_weights=attn_weights,
            query_states_2=query_states_2,
            kb_keys=kb_keys,
            kb_len=kb_len,
            kb_config=kb_config,
            layer_idx=int(module.layer_idx),
            head_dim=int(module.head_dim),
        )

    # ------------------------------------------------------------
    # 5. softmax（HF 原逻辑，fp32）
    # ------------------------------------------------------------
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)

    # ------------------------------------------------------------
    # 6. ⭐ KBLaM Hook 3：path_attn（softmax 后）
    # ------------------------------------------------------------
    if (
        kb_len > 0
        and kb_config is not None
        and kb_config.path_attn
        and kb_adj is not None
    ):
        attn_weights = apply_kblam_path_attention(
            attn_weights=attn_weights,
            kb_len=kb_len,
            kb_adj=kb_adj,
            kb_config=kb_config,
            layer_idx=int(module.layer_idx),
        )

    # ------------------------------------------------------------
    # 7. dropout + matmul V（HF 原逻辑）
    # ------------------------------------------------------------
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
