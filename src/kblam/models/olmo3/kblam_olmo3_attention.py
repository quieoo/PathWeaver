# coding=utf-8
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3Attention,
    apply_rotary_pos_emb,
)
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.utils.generic import TransformersKwargs

from kblam.models.kblam_config import KBLaMConfig
from kblam.kblam_attention import (
    apply_kblam_attention,
    apply_kblam_path_attention,
)
from kblam.kblam_attention.kblam_injector import apply_kblam_sep_query_head


def replace_attention_with_kblam(model):
    """
    将 HF OLMo3 的每层 self_attn 替换为 KBLAMOlmo3Attention，并严格加载权重。
    """
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        old = layer.self_attn
        new = KBLAMOlmo3Attention(model.config, layer_idx=layer_idx)

        # 权重对齐
        # new.load_state_dict(old.state_dict(), strict=True)
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        print(
            f"Layer {layer_idx} load_state_dict: "
            f"missing={missing}, unexpected={unexpected}"
        )

        # device/dtype 对齐（你前面已经验证这是必要的）
        p = next(old.parameters())
        new.to(device=p.device, dtype=p.dtype)

        layer.self_attn = new

    # 打印确认
    print("Attention replacement done.")
    print("Layer0 self_attn:", type(model.model.layers[0].self_attn).__name__)

class KBLAMOlmo3Attention(Olmo3Attention):
    """
    OLMo3Attention + KBLaM injector (eager attention only)

    设计原则：
    - 除 KBLaM hook 外，行为与 HF Olmo3Attention 完全一致
    - 不改 cache / RoPE / mask / scaling 语义
    - KBLaM 逻辑全部委托给 kblam_attention/*
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # ===== KBLaM: 独立 query head（用于 sep_query_head）=====
        # 初始行为：等价于普通 query（权重拷贝）
        # self.q_proj_new = nn.Linear(
        #     self.config.hidden_size,
        #     self.num_attention_heads * self.head_dim,
        #     bias=self.q_proj.bias is not None,
        # )
        out_dim = self.q_proj.weight.shape[0]   # = num_heads * head_dim
        in_dim = self.q_proj.weight.shape[1]    # = hidden_size

        self.q_proj_new = nn.Linear(
            in_dim,
            out_dim,
            bias=self.q_proj.bias is not None,
        )
        self.q_proj_new.load_state_dict(self.q_proj.state_dict(), strict=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        *,
        # ===== KBLaM 扩展参数 =====
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
        kb_adj: Optional[torch.Tensor] = None,
        **kwargs: TransformersKwargs,
    ):
        """
        返回：
            attn_output: (B, T, hidden)
            attn_weights: (B, H, Q, K_total) 或 None
        """

        # ------------------------------------------------------------
        # 1. Q / K / V 投影（HF 原逻辑）
        # ------------------------------------------------------------
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        query_states_2 = self.q_norm(self.q_proj_new(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        query_states_2 = query_states_2.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # ------------------------------------------------------------
        # 2. RoPE（HF 原逻辑）
        # ------------------------------------------------------------
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # ------------------------------------------------------------
        # 3. KV cache（HF 原逻辑）
        # ------------------------------------------------------------
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # ------------------------------------------------------------
        # 4. repeat KV（HF eager attention 逻辑）
        # ------------------------------------------------------------
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ------------------------------------------------------------
        # 5. ⭐ KBLaM Hook 1：prepend KB KV + 扩 mask
        # ------------------------------------------------------------
        if kb_kvs is not None and kb_config is not None:
            num_heads = self.q_proj.weight.shape[0] // self.head_dim

            key_states, value_states, attention_mask, kb_len = apply_kblam_attention(
                query_states=query_states,
                query_states_2=query_states_2,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                kb_kvs=kb_kvs,
                kb_config=kb_config,
                kb_adj=kb_adj,
                layer_idx=int(self.layer_idx),
                num_heads=int(num_heads),
                head_dim=int(self.head_dim),
                num_hidden_layers=int(self.config.num_hidden_layers),
            )
        else:
            kb_len = 0

        # if self.layer_idx == 0:
        #     print(
        #         "[Phase1] query:", query_states.shape,
        #         "key:", key_states.shape
        #     )

        # ------------------------------------------------------------
        # 6. attention logits（HF eager attention 逻辑）
        # ------------------------------------------------------------
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # ------------------------------------------------------------
        # 7. ⭐ KBLaM Hook 2：sep_query_head（softmax 前）
        # ------------------------------------------------------------
        if (
            kb_len > 0
            and kb_config is not None
            and kb_config.sep_query_head
        ):
            kb_keys = key_states[..., :kb_len, :]
            attn_weights = apply_kblam_sep_query_head(
                attn_weights=attn_weights,
                query_states_2=query_states_2,
                kb_keys=kb_keys,
                kb_len=kb_len,
                kb_config=kb_config,
                layer_idx=int(self.layer_idx),
                head_dim=int(self.head_dim),
            )

        # ------------------------------------------------------------
        # 8. softmax（fp32）+ dropout（HF 原逻辑）
        # ------------------------------------------------------------
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # if self.layer_idx == 0 and kb_config.sep_query_head:
        #     kb_attn = attn_weights[..., :kb_len]
        #     text_attn = attn_weights[..., kb_len:]

        #     print(
        #         "[Phase3][AttnRatio]",
        #         "kb_mean=", kb_attn.mean().item(),
        #         "text_mean=", text_attn.mean().item(),
        #     )
        # if self.layer_idx == 0 and kb_len > 0:
        #     kb_part = attn_weights[..., :kb_len]
        #     print(
        #         "[Phase1] KB attn stats:",
        #         "mean=", kb_part.mean().item(),
        #         "max=", kb_part.max().item(),
        #     )

        # if self.layer_idx == 0:
        #     kb_attn = attn_weights[..., :kb_len]
        #     text_attn = attn_weights[..., kb_len:]

        #     print(
        #         "[Phase3][AttnRatio]",
        #         "kb_mean=", kb_attn.mean().item(),
        #         "text_mean=", text_attn.mean().item(),
        #     )

        if (
            kb_len > 0
            and kb_config is not None
            and kb_config.path_attn
            and kb_adj is not None
        ):
            # --------------------------------------------------------
            # 9. ⭐ KBLaM Hook 3：path attention（softmax 后）
            # --------------------------------------------------------
            attn_weights = apply_kblam_path_attention(
                attn_weights=attn_weights,
                kb_len=kb_len,
                kb_adj=kb_adj,
                kb_config=kb_config,
                layer_idx=int(self.layer_idx),
            )

        attn_weights = nn.functional.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        # ------------------------------------------------------------
        # 10. matmul V（HF 原逻辑）
        # ------------------------------------------------------------
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # ------------------------------------------------------------
        # 11. output projection（HF 原逻辑）
        # ------------------------------------------------------------
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
