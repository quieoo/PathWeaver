# coding=utf-8
"""
Export HF reference logits (Transformers >= 4.57)
=================================================
Run this script in an environment where you can load official OLMo3 HF model
(e.g., transformers 4.57).

It saves:
  - logits (float32 on CPU)
  - input_ids, attention_mask (CPU)
  - tokenizer info (pad/eos ids)
so you can compare against your KBLaM-OLMo3 implementation on another env.

Example:
  python export_hf_ref_logits.py \
    --model_dir /path/to/olmo3-7B-instruct \
    --text "The capital of France is" \
    --out ref_logits.pt \
    --dtype bfloat16 \
    --device cuda \
    --last_token_only
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Local HF model directory (OLMo3)")
    ap.add_argument("--text", required=True, help="Prompt text")
    ap.add_argument("--out", default="ref_logits.pt", help="Output .pt file")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--use_fast", action="store_true", help="Use fast tokenizer (default: False)")
    ap.add_argument("--last_token_only", action="store_true", help="Save only last-token logits")
    ap.add_argument("--max_length", type=int, default=None, help="Optional truncation max_length")
    args = ap.parse_args()

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=args.use_fast)

    # Ensure pad_token exists for consistent attention_mask behavior
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_kwargs = dict(return_tensors="pt")
    if args.max_length is not None:
        tok_kwargs.update(dict(truncation=True, max_length=args.max_length))

    inputs = tokenizer(args.text, **tok_kwargs)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map=None,  # keep simple; you can switch to "auto" if you want
    ).to(args.device)
    model.eval()

    input_ids = input_ids.to(args.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # (B, T, V)

    # Move to CPU float32 for stable comparison across envs
    logits = logits.float().cpu()
    input_ids_cpu = input_ids.cpu()
    attention_mask_cpu = attention_mask.cpu() if attention_mask is not None else None

    if args.last_token_only:
        logits_to_save = logits[:, -1, :].contiguous()
    else:
        logits_to_save = logits.contiguous()

    payload = {
        "logits": logits_to_save,
        "input_ids": input_ids_cpu,
        "attention_mask": attention_mask_cpu,
        "text": args.text,
        "decoded": tokenizer.decode(input_ids_cpu[0], skip_special_tokens=False),
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "model_dir": args.model_dir,
    }

    torch.save(payload, args.out)
    print(f"Saved reference logits to: {args.out}")
    print("Logits shape:", tuple(logits_to_save.shape))
    print("Decoded:", payload["decoded"])


if __name__ == "__main__":
    main()


# conda activate olmo3
# python export_hf_ref_logits.py \
#   --model_dir /home/sdu/zhu/models/olmo3-7b/ \
#   --text "The capital of France is" \
#   --out ref_last.pt \
#   --dtype bfloat16 \
#   --device cuda \
#   --last_token_only