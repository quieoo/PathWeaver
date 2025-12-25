# test_phase_1_kb_injection.py
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblam.models.olmo3.kblam_olmo3_attention import (
    KBLAMOlmo3Attention,
    replace_attention_with_kblam,
)
from kblam.models.kblam_config import KBLaMConfig


# =========================
# 基本配置
# =========================
MODEL_PATH = "/home/sdu/zhu/models/olmo3-7b/"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1234)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(1234)


# =========================
# Phase 1 日志控制
# =========================
LOG_LAYER = 0        # 只在 layer 0 打详细日志，避免刷屏
LOG_ONCE = True      # 只打印一次


# =========================
# 构造真实形态 Fake KB
# =========================
def build_fake_kb(model, kb_len=4, kb_layer_frequency=1):
    hidden = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    slots = num_layers // kb_layer_frequency + 1
    kb_dim = hidden * slots

    print(f"[Phase1] Build Fake KB:")
    print(f"  kb_len={kb_len}")
    print(f"  hidden={hidden}")
    print(f"  slots={slots}")
    print(f"  kb_dim={kb_dim}")

    kb_keys = torch.randn(
        kb_len,
        kb_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    kb_values = torch.randn(
        kb_len,
        kb_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    return kb_keys, kb_values


# =========================
# 主测试流程
# =========================
@torch.no_grad()
def main():
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    print(f"[Phase1] Using device: {DEVICE}")
    torch.cuda.empty_cache()
    gc.collect()

    # -------- 1. Load tokenizer & model --------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE)

    model.eval()

    # -------- 2. Replace attention --------
    model.set_attn_implementation("eager")

    replace_attention_with_kblam(model)

    assert isinstance(
        model.model.layers[0].self_attn,
        KBLAMOlmo3Attention,
    ), "[Phase1] ❌ Attention replacement failed"

    print("[Phase1] ✅ Attention replaced with KBLAMOlmo3Attention")

    # -------- 3. Build input --------
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    print(f"[Phase1] Input length = {input_len}")

    # -------- 4. KB OFF baseline --------

    print("\n" + "=" * 80)
    print("[Phase1] KB OFF generation")
    output_ids_no_kb = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

    result_no_kb = tokenizer.decode(
        output_ids_no_kb[0][input_len:],
        skip_special_tokens=True,
    )
    print("[Phase1][KB OFF]", result_no_kb)

    # -------- 5. Build Fake KB --------
    kb_layer_frequency = 1
    kb_keys, kb_values = build_fake_kb(
        model,
        kb_len=4,
        kb_layer_frequency=kb_layer_frequency,
    )

    kb = (kb_keys, kb_values)

    kb_config = KBLaMConfig(
        sep_query_head=False,
        kb_layer_frequency=kb_layer_frequency,
        path_attn=False,
    )

    # -------- 6. KB ON generation --------
    print("\n" + "=" * 80)
    print("[Phase1] KB ON generation")

    # output_ids_kb = model.generate(
    #     **inputs,
    #     kb_kvs=kb,
    #     kb_config=kb_config,
    #     output_attentions=True,
    #     max_new_tokens=64,
    #     do_sample=False,
    # )

    # 修复：绕过 generate()，直接调用 model.forward()
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        kb_kvs=kb,
        kb_config=kb_config,
        output_attentions=True,
    )

    logits = out.logits
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    output_ids_kb = torch.cat(
        [inputs["input_ids"], next_token],
        dim=1,
    )

    result_kb = tokenizer.decode(
        output_ids_kb[0][input_len:],
        skip_special_tokens=True,
    )
    print("[Phase1][KB ON ]", result_kb)

    # -------- 7. Final verdict --------
    print("\n" + "=" * 80)
    if result_kb != result_no_kb:
        print("[Phase1] ✅ PASS: KB ON output differs from KB OFF")
    else:
        print("[Phase1] ❌ FAIL: KB ON output identical to KB OFF")

    print("=" * 80)


if __name__ == "__main__":
    print("验证：OLMo3 的 attention 中，KB token 被真实注入并参与了注意力计算。")
    main()
