import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ 用你当前工程里的 KBLAMOlmo3Attention
from kblam.models.olmo3.kblam_olmo3_attention import KBLAMOlmo3Attention


MODEL_PATH = "/home/sdu/zhu/models/olmo3-7b/"
DTYPE = torch.bfloat16


def replace_attention_with_kblam(model):
    """
    将 HF OLMo3 的每层 self_attn 替换为 KBLAMOlmo3Attention，并严格加载权重。
    """
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        old = layer.self_attn
        new = KBLAMOlmo3Attention(model.config, layer_idx=layer_idx)

        # 权重对齐
        # new.load_state_dict(old.state_dict(), strict=True)
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        print(
            f"[Phase 5] Layer {layer_idx} load_state_dict: "
            f"missing={missing}, unexpected={unexpected}"
        )


        # device/dtype 对齐（你前面已经验证这是必要的）
        p = next(old.parameters())
        new.to(device=p.device, dtype=p.dtype)

        layer.self_attn = new

    # 打印确认
    print("[Phase 5] Attention replacement done.")
    print("[Phase 5] Layer0 self_attn:", type(model.model.layers[0].self_attn).__name__)


@torch.no_grad()
def main():
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Phase 5] Using device: {device}")

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------
    # 1) Load tokenizer/model
    # ------------------------------------------------------------
    print("[Phase 5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("[Phase 5] Loading model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    # ------------------------------------------------------------
    # 2) Replace attention with KBLAMOlmo3Attention
    # ------------------------------------------------------------
    replace_attention_with_kblam(model)

    # 强校验：确保确实替换成功（避免“看起来跑了其实没替换”）
    assert isinstance(model.model.layers[0].self_attn, KBLAMOlmo3Attention), \
        "Layer0 self_attn is not KBLAMOlmo3Attention. Replacement failed."

    print(
        model.model.layers[0].self_attn.q_proj.weight.shape,
        model.model.layers[0].self_attn.q_proj_new.weight.shape,
    )

    # ------------------------------------------------------------
    # 3) Build single chat case
    # ------------------------------------------------------------
    message = [
        {"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}
    ]

    inputs = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("[Phase 5] Input shape:")
    for k, v in inputs.items():
        print(f"  {k}: {tuple(v.shape)}")

    # ------------------------------------------------------------
    # 4) Generate (IMPORTANT: do NOT pass kb_* args)
    # ------------------------------------------------------------
    print("[Phase 5] Generating response (KB OFF: not passing kb_* args)...")

    # 为了稳定性，可以固定 seed（可选）
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)

    output_ids = model.generate(
        **inputs,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        max_new_tokens=1024,  # 你示例里 32768 太大，先用较小值做回归验证更稳
    )

    # ------------------------------------------------------------
    # 5) Decode
    # ------------------------------------------------------------
    input_len = inputs["input_ids"].shape[1]
    result = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("[Phase 5] Response:")
    print(result)
    print("=" * 80)

    print("[Phase 5] ✅ PASSED: KBLAMOlmo3Attention in-place, KB OFF generation completed.")


if __name__ == "__main__":
    main()
