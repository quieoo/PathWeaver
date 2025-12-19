import os
import gc
import json
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LogitsProcessorList

from olmo3_standalone import (
    DeviceAwareSuppressTokensLogitsProcessor,
    HFWrapperForGeneration,
    Olmo3Config,
    Olmo3StandaloneModel,
)


# --------------------------------------------------
# device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()


# --------------------------------------------------
# model dir
# --------------------------------------------------
MODEL_DIR = "/home/sdu/zhu/models/olmo3-7b/"


# --------------------------------------------------
# load HF config.json
# --------------------------------------------------
print("Loading config.json...")
with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    hf_config = json.load(f)

config = Olmo3Config(
    vocab_size=hf_config["vocab_size"],
    hidden_size=hf_config["hidden_size"],
    intermediate_size=hf_config["intermediate_size"],
    num_hidden_layers=hf_config["num_hidden_layers"],
    num_attention_heads=hf_config["num_attention_heads"],
    num_key_value_heads=hf_config.get("num_key_value_heads"),
    max_position_embeddings=hf_config["max_position_embeddings"],
    rms_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
    rope_theta=hf_config.get("rope_theta", 10000.0),
    sliding_window=hf_config.get("sliding_window"),
    pad_token_id=hf_config.get("pad_token_id", 1),
)

print("Config loaded.")


# --------------------------------------------------
# build model
# --------------------------------------------------
print("Building OLMo3 standalone model...")
model = Olmo3StandaloneModel(config)


# --------------------------------------------------
# load safetensors shards via index.json
# --------------------------------------------------
print("Loading safetensors shards...")

index_file = os.path.join(MODEL_DIR, "model.safetensors.index.json")
with open(index_file, "r") as f:
    index = json.load(f)

weight_map = index["weight_map"]

state_dict = {}
for shard_file in sorted(set(weight_map.values())):
    shard_path = os.path.join(MODEL_DIR, shard_file)
    print(f"  loading {shard_file}")
    shard_state = load_file(shard_path, device="cpu")
    state_dict.update(shard_state)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
# print(f"Missing keys: {len(missing)}")
# print(f"Unexpected keys: {len(unexpected)}")

# 使用CPU和float32避免内存问题
model = model.to(device=device, dtype=torch.bfloat16)
model.eval()


# --------------------------------------------------
# tokenizer (with chat_template)
# --------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
chat_template_path = os.path.join(MODEL_DIR, "chat_template.jinja")
if os.path.exists(chat_template_path):
    with open(chat_template_path, "r") as f:
        tokenizer.chat_template = f.read()
    print("Chat template loaded from chat_template.jinja")
else:
    print("Warning: chat_template.jinja not found!")


def _build_banned_tokens(tokenizer: AutoTokenizer) -> list[int]:
    """Collect problematic special tokens to prevent corrupted generations."""

    banned_tokens: list[int] = []
    special_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|function|>",
        "<|function_calls|>",
    ]

    for tok in special_tokens:
        if tok in tokenizer.get_vocab():
            banned_tokens.append(tokenizer.convert_tokens_to_ids(tok))

    problematic_patterns = ["system", "user", "assistant", "function", "tool"]
    for pattern in problematic_patterns:
        for token_str, token_id in tokenizer.get_vocab().items():
            if pattern in token_str.lower() and len(token_str.strip()) < 10:
                banned_tokens.append(token_id)

    return sorted(set(banned_tokens))

# --------------------------------------------------
# minimal greedy generation (修复版本)
# --------------------------------------------------
@torch.no_grad()
def greedy_generate(
    model,
    input_ids,
    tokenizer,
    max_new_tokens=256,
):
    banned_tokens = _build_banned_tokens(tokenizer)

    for step in range(max_new_tokens):
        outputs = model(input_ids)
        # logits = outputs.last_hidden_state[:, -1, :]
        # next_token = torch.argmax(logits, dim=-1, keepdim=True)

        logits = outputs.logits[:, -1, :]   # (B, vocab_size)
        
        # ---- 禁用有问题的token ----
        if banned_tokens:
            logits[:, banned_tokens] = -1e9
        
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # 调试输出
        if step < 5:
            decoded = tokenizer.decode(next_token[0].item())
            print(f"Greedy 步骤 {step}: 生成 token {next_token[0].item()} = '{decoded}'")
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids
@torch.no_grad()
def sample_generate(
    model,
    input_ids,
    tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    eos_token_id = tokenizer.eos_token_id

    banned_tokens = _build_banned_tokens(tokenizer)

    print(f"禁止的token数量: {len(banned_tokens)}")

    for step in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # (B, vocab)

        # ---- 禁用 token（关键修复）----
        if banned_tokens:
            logits[:, banned_tokens] = -1e9

        # ---- temperature ----
        logits = logits / temperature

        # ---- softmax in fp32 ----
        probs = torch.softmax(logits.float(), dim=-1)

        # ---- top-p ----
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_probs[cumulative_probs > top_p] = 0.0
        sorted_probs[:, 0] = torch.where(
            sorted_probs[:, 0] == 0,
            torch.ones_like(sorted_probs[:, 0]),
            sorted_probs[:, 0],
        )
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token)

        # 调试输出
        if step < 5:  # 只打印前几步
            decoded = tokenizer.decode(next_token[0].item())
            print(f"步骤 {step}: 生成 token {next_token[0].item()} = '{decoded}'")

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return input_ids



# --------------------------------------------------
# test prompt (与你原始代码一致)
# --------------------------------------------------
# message = [
#     {"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}
# ]


# inputs = tokenizer.apply_chat_template(
#     message,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     return_dict=True,
# )
# inputs = {k: v.to(device) for k, v in inputs.items()}

prompt = "Who would win in a fight - a dinosaur or a cow named Moo Moo? Answer in a concise way."
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    return_attention_mask=True
).to(device)

banned_tokens = _build_banned_tokens(tokenizer)

print("Generating response...")
hf_model = HFWrapperForGeneration(model, tokenizer)
# hf_model.to(device)
hf_model.eval()



logits_processor = LogitsProcessorList()
if banned_tokens:
    logits_processor.append(
        DeviceAwareSuppressTokensLogitsProcessor(banned_tokens)
    )

with torch.no_grad():
    output_ids = hf_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
    )
    
result = tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print("\nResponse:")
print(result)
