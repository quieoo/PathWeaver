import torch
import gc
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Phase 5: Pure OLMo3 end-to-end generation (NO KB)
# ------------------------------------------------------------

MODEL_PATH = "/home/sdu/zhu/models/olmo3-7b/"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Phase 5] Using device: {device}")

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------
    print("[Phase 5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------
    # 2. Load model (IMPORTANT: no device_map here)
    # ------------------------------------------------------------
    print("[Phase 5] Loading model with bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )

    model.eval()
    model.to(device)

    # ------------------------------------------------------------
    # 3. Sanity check: ensure no KB is involved
    # ------------------------------------------------------------
    print("[Phase 5] Sanity check: KB disabled")
    assert not hasattr(model, "kb_config"), "Model unexpectedly has kb_config"
    assert not hasattr(model, "kb_kvs"), "Model unexpectedly has kb_kvs"

    # ------------------------------------------------------------
    # 4. Build a single chat input
    # ------------------------------------------------------------
    message = [
        {
            "role": "user",
            "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"
        }
    ]

    inputs = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move inputs to model device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("[Phase 5] Input shape:")
    for k, v in inputs.items():
        print(f"  {k}: {tuple(v.shape)}")

    # ------------------------------------------------------------
    # 5. Generate (pure HF generate, no KB args)
    # ------------------------------------------------------------
    print("[Phase 5] Generating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # ------------------------------------------------------------
    # 6. Decode only newly generated tokens
    # ------------------------------------------------------------
    input_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    )

    print("\n" + "=" * 80)
    print("[Phase 5] Response:")
    print(generated_text)
    print("=" * 80)

    print("[Phase 5] PASSED: pure OLMo3 generation completed successfully")


if __name__ == "__main__":
    main()
