#!/usr/bin/env python3
"""Manual Qwen3 backbone smoke test for CUDA or Ascend NPU.

This file intentionally does not use the ``test_*.py`` name, so pytest will
not load a 4B model during the regular test suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.models.qwen3.kblam_qwen3_attention import resolve_runtime_device


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1].parent / "models" / "qwen3-4B-Instruct"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--device",
        default="npu",
        choices=("auto", "cpu", "cuda", "npu"),
        help="Execution device. 'auto' prefers CUDA, then NPU, then CPU.",
    )
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    device = resolve_runtime_device(None if args.device == "auto" else args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()

    model_inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = model_inputs.input_ids.shape[-1]
    completion = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
    print(f"device={device}")
    print(f"prompt={args.prompt}")
    print(f"completion={completion}")


if __name__ == "__main__":
    main()
