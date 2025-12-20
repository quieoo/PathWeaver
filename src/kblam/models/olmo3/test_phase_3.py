import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.models.olmo3.kblam_olmo3_attention import (
    KBLAMOlmo3Attention,
)
from kblam.models.kblam_config import KBLaMConfig

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

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    replace_attention_with_kblam(model)
    model.eval()

    # baseline
    out0 = model(**inputs, kb_config=KBLaMConfig(debug_bias=0.0)).logits

    # small bias
    out1 = model(**inputs, kb_config=KBLaMConfig(debug_bias=1e-4)).logits

    # larger bias
    out2 = model(**inputs, kb_config=KBLaMConfig(debug_bias=5e-4)).logits

    diff01 = (out1 - out0).abs().max().item()
    diff12 = (out2 - out1).abs().max().item()

    top0 = out0.argmax(-1)
    top1 = out1.argmax(-1)
    top2 = out2.argmax(-1)

    print("=== Phase 3.0 Verification ===")
    print(f"max |out1 - out0| : {diff01}")
    print(f"max |out2 - out1| : {diff12}")
    print(f"top0 == top1 : {(top0 == top1).float().mean().item()}")
    print(f"top1 == top2 : {(top1 == top2).float().mean().item()}")

    assert diff01 > 0.0, "Phase 3.0 FAILED: no numerical change"
    assert diff12 > diff01, "Phase 3.0 FAILED: non-monotonic change"
    assert torch.isfinite(out2).all(), "Phase 3.0 FAILED: NaN/Inf"

    print("✅ Phase 3.0 PASSED")


if __name__ == "__main__":
    main()
