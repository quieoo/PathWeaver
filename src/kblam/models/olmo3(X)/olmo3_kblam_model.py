# coding=utf-8
"""
KBLaM-compatible OLMo3 implementation (fixed for closer HF behavior)
===================================================================
✔ Q / K RMSNorm (loads from HF)
✔ Sliding Window Attention
✔ Minimal YARN RoPE (via config.olmo3_rope_scaling)
✔ HF OLMo3-style LayerNorm naming/placement
✔ Return logits (CausalLMOutputWithPast) for easy comparison/generation
Compatible with Transformers 4.46
"""

import math
from typing import Optional, List, Tuple

import torch
from torch import nn

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from kblam.models.llama3_model import KblamLlamaAttention


# ============================================================
# 1. Config
# ============================================================

class Olmo3Config(LlamaConfig):
    model_type = "kblam_olmo3"

    def __init__(
        self,
        sliding_window: int = 4096,
        layer_types: Optional[List[str]] = None,
        rms_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sliding_window = sliding_window
        self.rms_norm_eps = rms_norm_eps

        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types


# ============================================================
# 2. OLMo2 / OLMo3 RMSNorm
# ============================================================

class Olmo2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(var + self.eps)
        return (self.weight * hidden_states).to(dtype)


# ============================================================
# 3. Minimal YARN RoPE (inference-only approximation)
# ============================================================

class MinimalYarnRotaryEmbedding(LlamaRotaryEmbedding):
    """
    Minimal YARN-style scaling (inference only).
    Uses config.olmo3_rope_scaling dict:
      - original_max_position_embeddings
      - factor
      - attention_factor
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        original_max_position_embeddings: int = 8192,
        factor: float = 8.0,
        attention_factor: float = 1.0,
    ):
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)
        self.orig_max_pos = int(original_max_position_embeddings)
        self.factor = float(factor)
        self.attention_factor = float(attention_factor)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        cos, sin = super().forward(x, position_ids)

        # more stable than global max for mixed batches
        max_pos = int(position_ids[:, -1].max())
        if max_pos <= self.orig_max_pos:
            return cos, sin

        # Minimal scaling (not exact HF YARN)
        scale = (max_pos / self.orig_max_pos) ** (1.0 / self.factor)
        scale = scale * self.attention_factor
        return cos * scale, sin * scale


# ============================================================
# 4. OLMo3 Attention (KBLaM-compatible)
# ============================================================

class KblamOlmo3Attention(KblamLlamaAttention):
    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = (
            config.sliding_window
            if self.attention_type == "sliding_attention"
            else None
        )

        self.q_norm = Olmo2RMSNorm(
            config.num_attention_heads * self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Olmo2RMSNorm(
            config.num_key_value_heads * self.head_dim,
            eps=config.rms_norm_eps,
        )

        # ---- Enable Minimal YARN if loader provided olmo3_rope_scaling ----
        rope_cfg = getattr(config, "olmo3_rope_scaling", None)
        if isinstance(rope_cfg, dict) and rope_cfg.get("rope_type") == "yarn":
            self.rotary_emb = MinimalYarnRotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000.0),
                original_max_position_embeddings=rope_cfg.get("original_max_position_embeddings", 8192),
                factor=rope_cfg.get("factor", 8.0),
                attention_factor=rope_cfg.get("attention_factor", 1.0),
            )

    def _apply_path_attention(self, attn_weights, kb_adj, kb_config):
        kb_len = kb_adj.size(-1)
        alpha_kb = attn_weights[:, :, :, :kb_len]

        if kb_adj.is_sparse:
            B, H, Q, _ = alpha_kb.shape
            alpha_flat = alpha_kb.reshape(B, -1, kb_len)
            adj = kb_adj.coalesce().to(alpha_kb.device, alpha_kb.dtype)

            beta_chunks = []
            for b in range(B):
                beta_flat = torch.sparse.mm(
                    adj.transpose(0, 1),
                    alpha_flat[b].transpose(0, 1)
                ).transpose(0, 1)
                beta_chunks.append(beta_flat.view(H, Q, kb_len))
            beta_kb = torch.stack(beta_chunks, dim=0)
        else:
            A = kb_adj.to(attn_weights.device).unsqueeze(1).unsqueeze(2)
            beta_kb = torch.matmul(alpha_kb.unsqueeze(-2), A).squeeze(-2)

        mix = getattr(kb_config, "path_attn_mix_ratio", 1.0)
        beta_kb = mix * beta_kb + (1.0 - mix) * alpha_kb
        beta_kb = beta_kb / beta_kb.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        new_attn = attn_weights.clone()
        new_attn[:, :, :, :kb_len] = beta_kb
        new_attn = new_attn / new_attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return new_attn

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        kb_kvs=None,
        kb_config=None,
        kb_adj=None,
    ):
        """
        Experiment A Attention Forward (fixed return signature):
        - Q/K RMSNorm: DISABLED
        - RoPE: DISABLED
        - Return (attn_output, attn_weights=None, past_key_value)
        """

        bsz, q_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # ------------------------------------------------
        # 1. Project to Q / K / V
        # ------------------------------------------------
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # ------------------------------------------------
        # 2. Shape to (B, H, Q, D)
        # ------------------------------------------------
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ------------------------------------------------
        # 3. ❌ Q/K RMSNorm — DISABLED (Experiment A)
        # ------------------------------------------------
        # q = self.q_norm(q)
        # k = self.k_norm(k)

        # ------------------------------------------------
        # 4. ❌ RoPE — DISABLED
        # ------------------------------------------------
        # no rotary position embedding

        # ------------------------------------------------
        # 5. Attention scores
        # ------------------------------------------------
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # ------------------------------------------------
        # 6. Add attention mask
        # ------------------------------------------------
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # ------------------------------------------------
        # 7. Softmax
        # ------------------------------------------------
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = attn_probs.to(v.dtype)

        # ------------------------------------------------
        # 8. Attention output
        # ------------------------------------------------
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, hidden_dim)

        # ------------------------------------------------
        # 9. Output projection
        # ------------------------------------------------
        attn_output = self.o_proj(attn_output)

        # ------------------------------------------------
        # 10. Cache (unused in Experiment A)
        # ------------------------------------------------
        next_past_key_value = None

        # ⚠️ 返回三个值，严格匹配 DecoderLayer
        return attn_output, None, next_past_key_value




# ============================================================
# 5. Decoder Layer (HF OLMo3-style)
# ============================================================

class KblamOlmo3DecoderLayer(nn.Module):
    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.self_attn = KblamOlmo3Attention(config, layer_idx)
        self.mlp = LlamaMLP(config)

        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        kb_kvs=None,
        kb_config=None,
        kb_adj=None,
        **kwargs,
    ):
        # Attention block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, _, pkv = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            kb_kvs=kb_kvs,
            kb_config=kb_config,
            kb_adj=kb_adj,
        )
        hidden_states = residual + hidden_states

        # FFN block
        residual = hidden_states
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, pkv


# ============================================================
# 6. Model (returns logits)
# ============================================================

class KblamOlmo3Model(LlamaPreTrainedModel):
    config_class = Olmo3Config

    def __init__(self, config: Olmo3Config):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [KblamOlmo3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (OLMo3 sets tie_word_embeddings=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def _make_causal_mask(self, q_len: int, k_len: int, device, dtype):
        # shape: (1, 1, q_len, k_len)
        # allow attending to <= current position
        mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)  # upper triangle above diagonal is -inf
        return mask.unsqueeze(0).unsqueeze(0)

    def _expand_padding_mask(
        self,
        attention_mask_2d: torch.Tensor,
        q_len: int,
        k_len: int,
        dtype,
    ):
        """
        attention_mask_2d: (B, k_len), 1 = keep, 0 = pad
        returns: (B, 1, q_len, k_len) with 0 for keep, -inf for pad
        """
        bsz = attention_mask_2d.size(0)

        # bool mask: True where pad
        pad_mask = attention_mask_2d == 0  # (B, k_len)

        # start from zeros
        expanded = torch.zeros(
            (bsz, 1, q_len, k_len),
            device=attention_mask_2d.device,
            dtype=dtype,
        )

        # fill pad positions with -inf
        expanded = expanded.masked_fill(
            pad_mask[:, None, None, :],
            float("-inf"),
        )
        return expanded

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        cache_position=None,
        kb_kvs=None,
        kb_config=None,
        kb_adj=None,
        **kwargs,
    ):
        """
        HF-aligned forward for KBLaM-compatible OLMo3 (no-cache baseline).

        Key properties:
        - mask-aware position_ids
        - explicit cache_position as RoPE time axis
        - safe causal + padding attention mask (no NaN)
        """

        # ------------------------------------------------
        # 0. Embeddings
        # ------------------------------------------------
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, q_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        mask_dtype = torch.float32

        # ------------------------------------------------
        # 1. cache_position (RoPE time axis)
        # ------------------------------------------------
        # HF no-cache forward equivalent
        if cache_position is None:
            cache_position = torch.arange(
                q_len,
                device=device,
            )

        # ------------------------------------------------
        # 2. position_ids (mask-aware, HF semantics)
        # ------------------------------------------------
        if position_ids is None:
            if attention_mask is not None and attention_mask.dim() == 2:
                # HF behavior:
                #   position_ids = cumsum(attention_mask) - 1
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = cache_position.unsqueeze(0)

        # ------------------------------------------------
        # 3. Build causal + padding attention mask (4D)
        # ------------------------------------------------
        k_len = q_len  # no-cache baseline

        # (1, 1, q_len, k_len)
        causal_4d = self._make_causal_mask(
            q_len=q_len,
            k_len=k_len,
            device=device,
            dtype=mask_dtype,
        )

        if attention_mask is None:
            attention_mask_4d = causal_4d
        else:
            if attention_mask.dim() == 2:
                # (B, 1, q_len, k_len), 0 keep / -inf pad
                pad_4d = self._expand_padding_mask(
                    attention_mask,
                    q_len=q_len,
                    k_len=k_len,
                    dtype=mask_dtype,
                )
                attention_mask_4d = causal_4d + pad_4d
            else:
                # already 4D (assume inverted mask)
                attention_mask_4d = attention_mask.to(mask_dtype)

        # ------------------------------------------------
        # 4. Decoder layers
        # ------------------------------------------------
        hidden_states = inputs_embeds
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            hidden_states, pkv = layer(
                hidden_states,
                attention_mask_4d,
                position_ids,
                None if past_key_values is None else past_key_values[i],
                use_cache,
                cache_position,
                kb_kvs,
                kb_config,
                kb_adj,
            )
            if use_cache:
                next_cache.append(pkv)

        # ------------------------------------------------
        # 5. Final norm + LM head
        # ------------------------------------------------
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=next_cache,
            hidden_states=None,
            attentions=None,
        )



__all__ = ["Olmo3Config", "KblamOlmo3Model"]
