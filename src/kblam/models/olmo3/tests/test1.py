# coding=utf-8
"""
Phase 0 + Phase 1 sanity test for KBLaM-compatible OLMo3
=======================================================

Covers:
  Phase 0:
    - tokenizer invariance
    - config / weight loading sanity
  Phase 1:
    - minimal forward correctness
    - determinism (no hidden state pollution)

NO:
  - HF logits alignment
  - KB
  - cache / generate
"""

import torch
from transformers import AutoTokenizer

from kblam.models.olmo3.olmo3_loader import load_kblam_olmo3_from_local


# -------------------------
# Config
# -------------------------
MODEL_DIR = "/home/sdu/zhu/models/olmo3-7b/"
DEVICE = "cuda"
DTYPE = "bfloat16"

PROMPT = "The capital of France is"


# -------------------------
# Phase 0.1 Tokenizer check
# -------------------------
def phase0_tokenizer_check():
    print("\n[Phase 0.1] Tokenizer invariance check")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

    ids = tok.encode(PROMPT)
    decoded = tok.decode(ids, skip_special_tokens=False)

    print("Prompt         :", PROMPT)
    print("Token IDs      :", ids)
    print("Decoded (raw)  :", decoded)

    assert isinstance(ids, list)
    assert len(ids) > 0
    assert decoded.strip() != ""

    print("✔ Tokenizer sanity OK")


# -------------------------
# Phase 0.2 Loader sanity
# -------------------------
def phase0_loader_check():
    print("\n[Phase 0.2] Model loading sanity check")

    model, cfg = load_kblam_olmo3_from_local(
        model_dir=MODEL_DIR,
        device=DEVICE,
        dtype=DTYPE,
    )

    # Minimal config checks
    assert cfg.vocab_size > 0
    assert cfg.hidden_size > 0
    assert cfg.num_hidden_layers > 0

    # lm_head must exist
    assert hasattr(model, "lm_head")
    assert model.lm_head.weight is not None

    print("✔ Model & config loaded successfully")
    return model, cfg


# -------------------------
# Phase 1.1 Minimal forward
# -------------------------
@torch.no_grad()
def phase1_minimal_forward(model, cfg):
    print("\n[Phase 1.1] Minimal forward correctness")

    x = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(1, 8),
        device=DEVICE,
    )

    out = model(input_ids=x, use_cache=False)

    assert out.logits is not None
    assert out.logits.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()

    print("Logits shape:", tuple(out.logits.shape))
    print("✔ Forward pass OK")


# -------------------------
# Phase 1.2 Determinism
# -------------------------
@torch.no_grad()
def phase1_determinism(model, cfg):
    print("\n[Phase 1.2] Determinism (no hidden-state pollution)")

    x = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(1, 8),
        device=DEVICE,
    )

    out1 = model(input_ids=x, use_cache=False).logits
    out2 = model(input_ids=x, use_cache=False).logits

    max_diff = (out1 - out2).abs().max().item()

    print("Max |logits1 - logits2| =", max_diff)

    assert max_diff == 0.0, "Non-deterministic output detected"

    print("✔ Determinism OK")


# -------------------------
# Main
# -------------------------
def main():
    print("========== Phase 0 + Phase 1 Test ==========")

    phase0_tokenizer_check()
    model, cfg = phase0_loader_check()
    phase1_minimal_forward(model, cfg)
    phase1_determinism(model, cfg)

    print("\n🎉 ALL Phase 0 + Phase 1 tests PASSED")


if __name__ == "__main__":
    main()
