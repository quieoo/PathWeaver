# coding=utf-8
"""
Local loader for KBLaM-compatible OLMo3 (Transformers 4.46 friendly)
===================================================================
Features:
- Parse config.json from local model dir (no AutoConfig needed)
- Preserve OLMo3 YARN rope_scaling for our MinimalYARN embedding via config.olmo3_rope_scaling
- Provide a 4.46-compatible rope_scaling dict (with "type") to satisfy KBLaM _init_rope()
- Load multi-shard safetensors
- Remap HF keys -> our model keys
- Load lm_head.weight
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
from safetensors.torch import load_file

from kblam.models.olmo3.olmo3_kblam_model import Olmo3Config, KblamOlmo3Model


_CONFIG_ALIASES = {
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_key_value_heads",
    "d_model": "hidden_size",
    "d_ff": "intermediate_size",
}

_LLAMA_KEYS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "initializer_range",
    "attention_bias",
    "attention_dropout",
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "rms_norm_eps",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "tie_word_embeddings",
    "use_cache",
}


def remap_hf_to_kblam_keys(hf_key: str) -> Optional[str]:
    """
    Map HF OLMo3 state_dict key -> our model key.
    Return None to skip.
    """
    # backbone
    if hf_key.startswith("model.embed_tokens."):
        return hf_key.replace("model.", "")
    if hf_key.startswith("model.layers."):
        return hf_key.replace("model.", "")
    if hf_key.startswith("model.norm."):
        return hf_key.replace("model.", "")

    # head
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # skip anything else
    return None


def build_olmo3_config_from_dir(model_dir: str) -> Olmo3Config:
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    # aliases
    for k_old, k_new in _CONFIG_ALIASES.items():
        if k_old in raw and k_new not in raw:
            raw[k_new] = raw[k_old]

    llama_kwargs = {k: raw[k] for k in _LLAMA_KEYS if k in raw}

    # Avoid double-inject rms_norm_eps (we pass explicitly below)
    llama_kwargs.pop("rms_norm_eps", None)

    # OLMo3 extras
    sliding_window = raw.get("sliding_window", 4096)
    layer_types = raw.get("layer_types", None)
    rms_norm_eps = raw.get("rms_norm_eps", 1e-5)

    # --- Rope scaling handling (critical for 4.46 + YARN) ---
    raw_rope_scaling = raw.get("rope_scaling", None)

    # If yarn: keep full dict for our model, and give 4.46 a compat rope_scaling {"type": ...}
    if isinstance(raw_rope_scaling, dict) and raw_rope_scaling.get("rope_type") == "yarn":
        llama_kwargs["olmo3_rope_scaling"] = raw_rope_scaling
        # Provide 4.46-compatible dict to avoid KeyError in KBLaM _init_rope()
        # This is NOT true YARN; real YARN behavior is implemented in our model via olmo3_rope_scaling.
        llama_kwargs["rope_scaling"] = {
            "type": "linear",
            "factor": float(raw_rope_scaling.get("factor", 1.0)),
        }
    else:
        # non-yarn: keep only if it already has "type", else drop
        if isinstance(raw_rope_scaling, dict) and "type" in raw_rope_scaling:
            llama_kwargs["rope_scaling"] = raw_rope_scaling
        else:
            llama_kwargs.pop("rope_scaling", None)

    cfg = Olmo3Config(
        **llama_kwargs,
        sliding_window=sliding_window,
        layer_types=layer_types,
        rms_norm_eps=rms_norm_eps,
    )
    return cfg


def load_kblam_olmo3_from_local(
    model_dir: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Tuple[KblamOlmo3Model, Olmo3Config]:
    """
    Load our KBLaM-compatible OLMo3 from a local HF-style directory.
    """

    cfg = build_olmo3_config_from_dir(model_dir)
    model = KblamOlmo3Model(cfg)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model.to(dtype=torch_dtype)

    sd = model.state_dict()
    loaded_keys = set()

    model_dir = Path(model_dir)

    shard_paths = sorted(model_dir.glob("model-*.safetensors"))
    if not shard_paths:
        # Some repos may use a single file name
        single = model_dir / "model.safetensors"
        if single.exists():
            shard_paths = [single]
        else:
            raise FileNotFoundError(f"No model-*.safetensors or model.safetensors found in {model_dir}")

    for shard in shard_paths:
        part = load_file(str(shard))
        remapped: Dict[str, torch.Tensor] = {}

        for k, v in part.items():
            new_k = remap_hf_to_kblam_keys(k)
            if new_k is None:
                continue
            if new_k in sd and sd[new_k].shape == v.shape:
                remapped[new_k] = v
                loaded_keys.add(new_k)

        model.load_state_dict(remapped, strict=False)

    # --- sanity check: only expected missing remain ---
    expected_missing = [
        k for k in sd.keys()
        if (
            "q_proj_new" in k      # HF has no q_proj_new
            or "score_shift" in k  # KBLaM-only param in attention base
        )
    ]

    real_missing = [
        k for k in sd.keys()
        if (k not in loaded_keys) and (k not in expected_missing)
    ]

    print("lm_head loaded:", "lm_head.weight" in loaded_keys)


    if real_missing:
        print("⚠️ WARNING: unexpected missing keys:")
        for k in real_missing[:40]:
            print(" ", k)
        if len(real_missing) > 40:
            print(f"... ({len(real_missing)} total)")

    model.to(device)
    model.eval()
    return model, cfg
