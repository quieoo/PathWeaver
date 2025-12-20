# src/kblam/models/olmo3/test_phase1_1.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.models.olmo3.kblam_olmo3_attention import KBLAMOlmo3Attention


def replace_attention_with_kblam(model):
    """
    Replace all attention modules with our subclass while keeping weights identical.
    TF 4.57 Olmo3Attention requires (config, layer_idx).
    """
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        old_attn = layer.self_attn

        # Create new attention with same config + correct layer_idx
        new_attn = KBLAMOlmo3Attention(model.config, layer_idx=layer_idx)

        # Copy weights
        new_attn.load_state_dict(old_attn.state_dict(), strict=True)

        # Move to same device/dtype as old_attn
        new_attn.to(device=next(old_attn.parameters()).device, dtype=next(old_attn.parameters()).dtype)

        layer.self_attn = new_attn


@torch.no_grad()
def main():
    # Hard-disable dynamo/compile path to avoid "module name not valid identifier" issues
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    device = "cuda"
    dtype = torch.bfloat16

    model_name = "/home/sdu/zhu/models/olmo3-7b/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load two separate models (avoid deepcopy)
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    model_kblam = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    # Replace attention in the second model
    replace_attention_with_kblam(model_kblam)

    # Test input
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)

    # Forward
    out_ref = model_ref(**inputs).logits
    out_new = model_kblam(**inputs).logits

    # Metrics
    diff = (out_ref - out_new).abs()
    max_diff = diff.max().item()

    top1_ref = out_ref.argmax(dim=-1)
    top1_new = out_new.argmax(dim=-1)
    top1_agree = (top1_ref == top1_new).float().mean().item()

    print("=== Phase 1.1 Verification ===")
    print("logits shape:", tuple(out_ref.shape))
    print("max |diff|  :", max_diff)
    print("top1 agree  :", top1_agree)

    # Assertions
    assert torch.isfinite(out_ref).all() and torch.isfinite(out_new).all()
    assert max_diff == 0.0, "Logits differ after attention replacement!"
    assert top1_agree == 1.0, "Top1 mismatch after attention replacement!"

    print("✅ Phase 1.1 PASSED")


if __name__ == "__main__":
    main()
