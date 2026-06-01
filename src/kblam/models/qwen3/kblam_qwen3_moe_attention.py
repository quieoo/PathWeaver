# coding=utf-8
from __future__ import annotations

import copy
import json
import os
import types
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from kblam.kblam_attention import (
    apply_kblam_attention,
    apply_kblam_path_attention,
)
from kblam.kblam_attention.kblam_injector import apply_kblam_sep_query_head
from kblam.models.kblam_config import KBLaMConfig


def replace_attention_with_kblam(model):
    """Replace each Qwen3-MoE self-attention layer with its KBLaM variant."""
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        old = layer.self_attn
        new = KBLAMQwen3MoeAttention(model.config, layer_idx=layer_idx)

        new.load_state_dict(old.state_dict(), strict=False)

        param = next(old.parameters())
        new.to(device=param.device, dtype=param.dtype)
        layer.self_attn = new

    print("Qwen3-MoE attention replacement done.")
    print("Layer0 self_attn:", type(model.model.layers[0].self_attn).__name__)


def _load_hf_sharded_state_dict(ckpt_dir: str) -> dict:
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = set(weight_map.values())

    state_dict = {}
    for shard in shard_files:
        shard_path = os.path.join(ckpt_dir, shard)
        state_dict.update(load_file(shard_path))

    return state_dict


def _load_qwen3_moe_state_dict(ckpt_dir: str) -> dict:
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    single_path = os.path.join(ckpt_dir, "model.safetensors")

    if os.path.exists(index_path):
        return _load_hf_sharded_state_dict(ckpt_dir)
    if os.path.exists(single_path):
        return load_file(single_path)

    raise FileNotFoundError(
        f"No Qwen3-MoE safetensors checkpoint found under {ckpt_dir}"
    )


def load_qwen3_moe_query_head(model, ckpt_dir: str):
    """Load q_proj_new tensors from a checkpoint directory into an injected model."""
    state_dict = _load_qwen3_moe_state_dict(ckpt_dir)
    query_head_state = {
        name: value for name, value in state_dict.items() if "q_proj_new" in name
    }
    if not query_head_state:
        raise RuntimeError(f"No q_proj_new tensors found in {ckpt_dir}")
    missing, unexpected = model.load_state_dict(query_head_state, strict=False)
    missing = [name for name in missing if "q_proj_new" in name]
    if missing:
        raise RuntimeError(
            "[FATAL] q_proj_new missing after query head load:\n" + "\n".join(missing)
        )
    if unexpected:
        print(f"[WARN] Unexpected query head tensors: {unexpected}")


def _prepare_inputs_for_generation_qwen3_moe(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=True,
    kb_kvs: Optional[tuple] = None,
    kb_config: Optional[KBLaMConfig] = None,
    kb_adj: Optional[torch.Tensor] = None,
    **kwargs,
):
    past_length = 0
    cache_length = None
    max_cache_length = None

    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            past_length = (
                cache_position[0]
                if cache_position is not None
                else past_key_values.get_seq_length()
            )

            if hasattr(past_key_values, "get_max_length"):
                max_len = past_key_values.get_max_length()
                if max_len is not None:
                    max_cache_length = torch.tensor(max_len, device=input_ids.device)
                    cache_length = torch.min(max_cache_length, past_length)
                else:
                    cache_length = past_length
            else:
                cache_length = past_length
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    model_inputs = copy.copy(kwargs)
    if inputs_embeds is not None and past_key_values is None:
        model_inputs["inputs_embeds"] = inputs_embeds
    else:
        model_inputs["input_ids"] = input_ids.contiguous()

    input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    if cache_position is None:
        cache_position = torch.arange(
            past_length, past_length + input_length, device=input_ids.device
        )
    elif use_cache:
        cache_position = cache_position[-input_length:]

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "kb_kvs": kb_kvs,
            "kb_config": kb_config,
            "kb_adj": kb_adj,
        }
    )
    return model_inputs


def _attach_qwen3_moe_generation_helpers(model):
    model.prepare_inputs_for_generation = types.MethodType(
        _prepare_inputs_for_generation_qwen3_moe,
        model,
    )
    return model


def load_kblam_qwen3_moe_model(
    *,
    base_model_dir: str,
    checkpoint_dir: str | None,
    device: str | torch.device | None = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = None,
):
    """
    Canonical loader for KBLaM-Qwen3-MoE.

    Order:
      1. load base Qwen3-MoE
      2. replace attention with KBLaM variant
      3. optionally load trained checkpoint
      4. assert integrity
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=(device_map is not None or device is None),
    )
    if device_map is None and device is not None:
        model = model.to(device)

    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"

    replace_attention_with_kblam(model)
    _attach_qwen3_moe_generation_helpers(model)
    model.config.use_cache = False

    if checkpoint_dir is not None:
        state_dict = _load_qwen3_moe_state_dict(checkpoint_dir)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        q_missing = [name for name in missing if "q_proj_new" in name]
        if q_missing:
            raise RuntimeError(
                "[FATAL] q_proj_new missing after state_dict load:\n"
                + "\n".join(q_missing)
            )
        if unexpected:
            print(f"[WARN] Unexpected tensors while loading Qwen3-MoE checkpoint: {unexpected}")

    _assert_kblam_qwen3_moe_integrity(model)
    model.eval()
    return model


def _assert_kblam_qwen3_moe_integrity(model):
    q_proj_new = [
        (name, param)
        for name, param in model.named_parameters()
        if "q_proj_new" in name
    ]

    if not q_proj_new:
        raise RuntimeError(
            "[FATAL] q_proj_new not found. Checkpoint was not loaded with KBLaM structure."
        )

    print(f"[OK] KBLaM-Qwen3-MoE loaded correctly ({len(q_proj_new)} q_proj_new tensors)")


class KBLAMQwen3MoeAttention(Qwen3MoeAttention):
    """Qwen3-MoE attention with KBLaM hooks injected into the eager path."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        out_dim = self.q_proj.weight.shape[0]
        in_dim = self.q_proj.weight.shape[1]
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
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
        kb_adj: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        output_attentions = bool(kwargs.get("output_attentions", False))
        use_sep_query_head = (
            kb_config is not None
            and kb_config.sep_query_head
            and int(self.layer_idx) % int(kb_config.kb_layer_frequency) == 0
        )

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        query_states_2 = None
        if use_sep_query_head:
            query_states_2 = self.q_norm(self.q_proj_new(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if kb_kvs is not None and kb_config is not None:
            kb_keys, kb_vals = kb_kvs
            if kb_keys.device != key_states.device:
                kb_keys = kb_keys.to(device=key_states.device, dtype=key_states.dtype)
            if kb_vals.device != value_states.device:
                kb_vals = kb_vals.to(device=value_states.device, dtype=value_states.dtype)
            if kb_adj is not None and kb_adj.device != key_states.device:
                kb_adj = kb_adj.to(key_states.device)

            num_heads = self.q_proj.weight.shape[0] // self.head_dim
            key_states, value_states, attention_mask, kb_len = apply_kblam_attention(
                query_states=query_states,
                query_states_2=query_states_2,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                kb_kvs=(kb_keys, kb_vals),
                kb_config=kb_config,
                kb_adj=kb_adj,
                layer_idx=int(self.layer_idx),
                num_heads=int(num_heads),
                head_dim=int(self.head_dim),
                num_hidden_layers=int(self.config.num_hidden_layers),
            )
        else:
            kb_len = 0

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if kb_len > 0 and use_sep_query_head and query_states_2 is not None:
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

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

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
                layer_idx=int(self.layer_idx),
            )

        attn_weights = nn.functional.dropout(
            attn_weights,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights
