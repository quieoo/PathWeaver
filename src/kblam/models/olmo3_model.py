# coding=utf-8
"""
Minimal, self-contained OLMo-3 model definition compatible with older
`transformers` (e.g., 4.46.x).  This mirrors the official architecture
closely enough to load the original checkpoints without relying on a
newer library release or `trust_remote_code`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig


class Olmo3Config(PretrainedConfig):
    model_type = "olmo3"

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = None,
        pad_token_id: int = 1,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool = True,
        use_cache: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads})"
            )


class Olmo3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(dtype)


class Olmo3RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, T)
        seq_len = position_ids.max().item() + 1
        # (T, dim/2)
        freqs = torch.outer(torch.arange(seq_len, device=position_ids.device, dtype=self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class Olmo3Attention(nn.Module):
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.q_norm = Olmo3RMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(self.num_kv_heads * self.head_dim, config.rms_norm_eps)

        self.dropout = config.attention_dropout
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_norm(self.q_proj(hidden_states))
        k = self.k_norm(self.k_proj(hidden_states))
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rotary(q, k, cos[:, :, :seq_len, :], sin[:, :, :seq_len, :])

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.to(q.dtype)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))
        if self.sliding_window is not None:
            idx = torch.arange(seq_len, device=hidden_states.device)
            causal_mask &= idx[None, :] >= idx[:, None] - self.sliding_window

        attn_scores = attn_scores.masked_fill(~causal_mask, torch.finfo(attn_scores.dtype).min)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(v.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class Olmo3MLP(nn.Module):
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = getattr(F, config.hidden_act) if hasattr(F, config.hidden_act) else F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Olmo3DecoderLayer(nn.Module):
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.self_attn = Olmo3Attention(config)
        self.mlp = Olmo3MLP(config)
        self.post_attention_layernorm = Olmo3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.post_attention_layernorm(
            self.self_attn(hidden_states, cos, sin, attention_mask)
        )
        hidden_states = hidden_states + self.post_feedforward_layernorm(self.mlp(hidden_states))
        return hidden_states


def _expand_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # Convert to additive mask with shape (B, 1, 1, T)
    expanded = attention_mask[:, None, None, :]
    expanded = (1.0 - expanded.to(dtype)) * torch.finfo(dtype).min
    return expanded


class Olmo3Model(PreTrainedModel):
    config_class = Olmo3Config
    base_model_prefix = "model"

    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([Olmo3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Olmo3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = Olmo3RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        cos, sin = self.rotary(position_ids)
        attn_mask = _expand_attention_mask(attention_mask, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attn_mask)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class Olmo3ForCausalLM(PreTrainedModel):
    config_class = Olmo3Config
    base_model_prefix = "model"

    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.model = Olmo3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = value.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.last_hidden_state,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)

