import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.models.olmo3.kblam_olmo3_attention import (
    KBLAMOlmo3Attention,
    kblam_calls_reset,
    kblam_calls_get,
)

os.environ["TORCHDYNAMO_DISABLE"] = "1"


def replace_attention_with_kblam(model):
    for layer_idx, layer in enumerate(model.model.layers):
        old = layer.self_attn
        new = KBLAMOlmo3Attention(model.config, layer_idx=layer_idx)
        new.load_state_dict(old.state_dict(), strict=True)
        new.to(device=next(old.parameters()).device, dtype=next(old.parameters()).dtype)

        layer.self_attn = new


@torch.no_grad()
def main():
    model_id = "/home/sdu/zhu/models/olmo3-7b/"
    tok = AutoTokenizer.from_pretrained(model_id)
    device = "cuda"
    dtype = torch.bfloat16

    text = "The capital of France is"
    inputs = tok(text, return_tensors="pt").to(device)


    model_ref = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    
    model_kblam = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    replace_attention_with_kblam(model_kblam)

    model_ref.eval()
    model_kblam.eval()

    # --- Phase 2: reset counter ---
    kblam_calls_reset()

    out_ref = model_ref(**inputs).logits
    out_new = model_kblam(**inputs, kb_kvs=None, kb_adj=None, kb_config=None).logits

    max_diff = (out_ref - out_new).abs().max().item()
    top1_agree = (out_ref.argmax(-1) == out_new.argmax(-1)).float().mean().item()

    calls = kblam_calls_get()
    num_layers = model_kblam.config.num_hidden_layers

    print("=== Phase 2.0 Verification ===")
    print(f"max |diff|  : {max_diff}")
    print(f"top1 agree  : {top1_agree}")
    print(f"apply_kblam calls (layers counted): {len(calls)}/{num_layers}")
    print(f"min calls per layer: {min(calls.values()) if calls else 0}")
    print(f"max calls per layer: {max(calls.values()) if calls else 0}")

    assert max_diff == 0.0, "Phase 2.0 FAILED: logits changed"
    assert top1_agree == 1.0, "Phase 2.0 FAILED: top1 mismatch"
    assert len(calls) == num_layers, "Phase 2.0 FAILED: not called in every layer"
    assert min(calls.values()) >= 1, "Phase 2.0 FAILED: some layers not called"

    print("✅ Phase 2.0 PASSED")


if __name__ == "__main__":
    main()
