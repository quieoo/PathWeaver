# coding=utf-8
"""
Minimal logits alignment script (no Transformers 4.57 required)
==============================================================
Workflow:

(A) On a machine/env where you can run the official HF OLMo3:
    - Run the same prompt with HF model
    - Save reference logits (e.g. torch.save({"logits": logits.cpu()}, "ref.pt")

(B) On current env (Transformers 4.46 + KBLaM):
    - Load our model from local dir
    - Compute logits
    - Compare against ref.pt

This script implements (B) and can also save "our logits" for debugging.

Usage:
  python compare_logits_min.py --model_dir /path/to/olmo3-7B-instruct --text "Hello" --ref ref.pt
  python compare_logits_min.py --model_dir ... --text "Hello" --save ours.pt
"""

import argparse
import torch
from transformers import AutoTokenizer

from kblam.models.olmo3.olmo3_loader import load_kblam_olmo3_from_local



def load_ref_logits(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "logits" in obj:
            return obj["logits"]
        if "ref_logits" in obj:
            return obj["ref_logits"]
    raise ValueError(f"Unrecognized reference file format: {path}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Local HF-style OLMo3 directory")
    ap.add_argument("--text", required=True, help="Prompt text")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--ref", default=None, help="Path to reference logits file (torch.save)")
    ap.add_argument("--save", default=None, help="Path to save our logits (torch.save)")
    ap.add_argument("--last_token_only", action="store_true", help="Compare only last-token logits")
    args = ap.parse_args()

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    # OLMo3 typically uses eos as pad in many setups; keep safe here
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model, cfg = load_kblam_olmo3_from_local(
        model_dir=args.model_dir,
        device=device,
        dtype=args.dtype,
    )

    inputs = tokenizer(args.text, return_tensors="pt").to(device)

    print("input_ids:", inputs["input_ids"][0].tolist())
    print("attention_mask:", inputs.get("attention_mask", None))

    # input_ids = inputs["input_ids"].to(device)

    # out = model(input_ids=input_ids, use_cache=False)

    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        use_cache=False,
    )

    logits = out.logits.float().cpu()  # compare in fp32


    print("Any NaN in logits:", torch.isnan(logits).any().item())


    if args.last_token_only:
        logits_cmp = logits[:, -1, :]
    else:
        logits_cmp = logits

    if args.save is not None:
        torch.save({"logits": logits_cmp}, args.save)
        print(f"Saved our logits to: {args.save}")

    if args.ref is None:
        print("No --ref provided; done.")
        print("Our logits shape:", tuple(logits_cmp.shape))
        return

    ref_logits = load_ref_logits(args.ref).float()

    if ref_logits.shape != logits_cmp.shape:
        raise RuntimeError(
            f"Shape mismatch: ref {tuple(ref_logits.shape)} vs ours {tuple(logits_cmp.shape)}"
        )

    diff = (ref_logits - logits_cmp).abs()
    mae = diff.mean().item()
    mx = diff.max().item()

    # top-1 agreement
    ref_top1 = ref_logits.argmax(dim=-1)
    ours_top1 = logits_cmp.argmax(dim=-1)
    top1_acc = (ref_top1 == ours_top1).float().mean().item()

    print("==== Logits Alignment ====")
    print("shape:", tuple(logits_cmp.shape))
    print("MAE :", mae)
    print("MAX :", mx)
    print("Top1 agreement:", top1_acc)


if __name__ == "__main__":
    main()


# conda activate kblam
# python compare_logits.py \
#   --model_dir /home/sdu/zhu/models/olmo3-7b/ \
#   --text "The capital of France is" \
#   --ref ref_last.pt \
#   --last_token_only