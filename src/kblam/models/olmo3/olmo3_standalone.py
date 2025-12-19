# olmo3_standalone.py
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.activations import ACT2FN

from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers import LogitsProcessorList, SuppressTokensLogitsProcessor
from transformers import LogitsProcessor  # 导入基类

class DeviceAwareSuppressTokensLogitsProcessor(LogitsProcessor):  # ✅ 正确继承
    """自动处理设备迁移的token禁止处理器"""
    
    def __init__(self, suppress_tokens):
        super().__init__()
        self.suppress_tokens_list = suppress_tokens
    
    def __call__(self, input_ids, scores):
        device = scores.device
        suppress_tokens = torch.tensor(
            self.suppress_tokens_list, 
            dtype=torch.long, 
            device=device
        )
        
        scores = scores.clone()
        vocab_tensor = torch.arange(scores.shape[-1], device=device)
        suppress_token_mask = torch.isin(vocab_tensor, suppress_tokens)
        scores[..., suppress_token_mask] = -float("inf")
        return scores

        
class HFWrapperForGeneration(PreTrainedModel, GenerationMixin):
    """
    A minimal HF-compatible wrapper that allows using `generate()`
    with a standalone OLMo3 model.

    This class ONLY handles generation logic.
    All attention / forward computation is delegated to the wrapped model.
    """

    config_class = PretrainedConfig

    def __init__(self, olmo_model, tokenizer):
        # ---- Build a minimal HF config ----
        config = PretrainedConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        super().__init__(config)

        self.model = olmo_model
        self.tokenizer = tokenizer

        # ---- Required flags for decoder-only models ----
        self.config.is_encoder_decoder = False
        self.config.use_cache = False  # you do not implement KV cache
        self.config.tie_word_embeddings = True

        # GenerationMixin expects this
        self.main_input_name = "input_ids"
        
        # 预计算并缓存需要禁止的token
        self._banned_tokens = self._get_banned_tokens()
        print(f"已识别 {len(self._banned_tokens)} 个需要禁止的特殊token")

    # --------------------------------------------------
    # Forward: delegate to standalone model
    # --------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _get_banned_tokens(self):
        return []

    # --------------------------------------------------
    # 重写generate方法以应用token禁止
    # --------------------------------------------------
    def generate(self, **kwargs):
        """重写生成方法，自动应用禁止token的LogitsProcessor"""
        # 获取或创建logits_processor
        logits_processor = kwargs.get('logits_processor', LogitsProcessorList())
        
        if not any(isinstance(p, DeviceAwareSuppressTokensLogitsProcessor) for p in logits_processor) and self._banned_tokens:
            logits_processor.append(
                DeviceAwareSuppressTokensLogitsProcessor(self._banned_tokens)
            )
        
        kwargs['logits_processor'] = logits_processor
        return super().generate(**kwargs)

    # --------------------------------------------------
    # Required by GenerationMixin
    # --------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # --------------------------------------------------
    # Optional but IMPORTANT:
    # Prevent HF from trying to access .get_input_embeddings()
    # on the wrapper itself
    # --------------------------------------------------
    def get_input_embeddings(self):
        return self.model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings
# ============================================================
# Config
# ============================================================

@dataclass
class Olmo3Config:
    vocab_size: int = 50304
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 2048

    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    attention_dropout: float = 0.0
    sliding_window: Optional[int] = None

    pad_token_id: int = 1

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0


# ============================================================
# RMSNorm (OLMo2/3 style)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


# ============================================================
# Rotary Embedding (default / llama3-compatible)
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.theta = config.rope_theta

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # x: (B, T, D)
        seq_len = x.shape[1]
        pos = position_ids[0].float()

        freqs = torch.outer(pos, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ============================================================
# Attention
# ============================================================
class Olmo3SelfAttention(nn.Module):
    """
    对齐 HF key:
      model.layers.N.self_attn.{q_proj,q_norm,k_proj,k_norm,v_proj,o_proj}.weight
    """
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.num_kv_heads * self.head_dim, config.rms_norm_eps)

        self.dropout = config.attention_dropout
        self.sliding_window = config.sliding_window

    def forward(self, x, cos, sin, attention_mask=None):
        B, T, _ = x.shape

        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.to(q.dtype)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        if self.sliding_window is not None:
            idx = torch.arange(T, device=x.device)
            causal = causal & (idx[None, :] >= idx[:, None] - self.sliding_window)

        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores + attention_mask[:, None, None, :].to(scores.dtype)

        attn = F.softmax(scores, dim=-1).to(v.dtype)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)                 # (B, H, T, D)
        out = out.transpose(1, 2).contiguous()      # (B, T, H, D)
        out = out.view(B, T, -1)                     # (B, T, hidden)
        return self.o_proj(out)

# ============================================================
# MLP
# ============================================================

class Olmo3MLP(nn.Module):
    """
    对齐 HF key:
      model.layers.N.mlp.{gate_proj,up_proj,down_proj}.weight
    """
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))



# ============================================================
# Decoder Layer
# ============================================================

class Olmo3DecoderLayer(nn.Module):
    """
    对齐 HF key:
      model.layers.N.post_attention_layernorm.weight
      model.layers.N.post_feedforward_layernorm.weight
    """
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.self_attn = Olmo3SelfAttention(config)
        self.mlp = Olmo3MLP(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, attention_mask=None):
        x = x + self.post_attention_layernorm(
            self.self_attn(x, cos, sin, attention_mask)
        )
        x = x + self.post_feedforward_layernorm(self.mlp(x))
        return x



# ============================================================
# Model
# ============================================================
class Olmo3StandaloneModel(nn.Module):
    """
    HF 对齐版 Standalone OLMo3

    state_dict keys 对齐：
      model.embed_tokens.weight
      model.layers.N.*
      model.norm.weight
      lm_head.weight
    """
    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.config = config

        # ---------- HF 对齐关键点 ----------
        self.model = nn.Module()

        self.model.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.model.layers = nn.ModuleList(
            [Olmo3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # ---------- RoPE ----------
        self.rotary = RotaryEmbedding(config)

        # ---------- LM head ----------
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # 权重共享（HF 行为）
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.model.embed_tokens(input_ids)

        cos, sin = self.rotary(x, position_ids)

        for layer in self.model.layers:
            x = layer(x, cos, sin, attention_mask)

        x = self.model.norm(x)
        logits = self.lm_head(x)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
