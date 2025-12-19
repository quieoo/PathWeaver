"""
Minimal smoke test that uses the self-contained Olmo3 model to generate text
without requiring a newer transformers release.
"""

import os
import torch
from transformers import AutoTokenizer

from kblam.models.olmo3_model import Olmo3Config, Olmo3ForCausalLM


MODEL_DIR = "/home/sdu/zhu/models/olmo3-7b/"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    chat_template_path = os.path.join(MODEL_DIR, "chat_template.jinja")
    if os.path.exists(chat_template_path):
        with open(chat_template_path, "r") as f:
            tokenizer.chat_template = f.read()

    config = Olmo3Config.from_pretrained(MODEL_DIR)
    model = Olmo3ForCausalLM.from_pretrained(MODEL_DIR, config=config, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    prompt = "Who would win in a fight - a dinosaur or a cow named Moo Moo? Answer concisely."
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print("Response:\n", result)


if __name__ == "__main__":
    main()
