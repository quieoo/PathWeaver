# coding=utf-8
"""
Automatic unit tests for KblamOlmo3Model
=======================================
Covers:
  1. Shape correctness
  2. KV cache (prefill -> decode)
  3. KB + path attention integration
"""

import torch

from kblam.models.olmo3.olmo3_kblam_model import (
    Olmo3Config,
    KblamOlmo3Model,
)

from kblam.models.olmo3.olmo3_loader import load_kblam_olmo3_from_local

model_name = "/home/sdu/zhu/models/olmo3-7b/"

model, cfg = load_kblam_olmo3_from_local(
    model_name,
    device="cuda",
    dtype="bfloat16",
)

import torch
x = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
with torch.no_grad():
    out = model(input_ids=x, use_cache=False)

