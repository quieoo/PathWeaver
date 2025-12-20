import torch
import gc
from modelscope import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()

print("Loading model with bfloat16...")
olmo = AutoModelForCausalLM.from_pretrained(
    "/home/sdu/zhu/models/olmo3-7b/",
    torch_dtype=torch.bfloat16,  # 使用bf16
)
tokenizer = AutoTokenizer.from_pretrained("/home/sdu/zhu/models/olmo3-7b/")

olmo.to(device)

message = [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}]
inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt', return_dict=True)

if hasattr(olmo, 'device'):
    inputs = {k: v.to(olmo.device) for k, v in inputs.items()}
else:
    inputs = {k: v.to(device) for k, v in inputs.items()}

print("Generating response...")
response = olmo.generate(**inputs, temperature=0.6,
    top_p=0.95,
    max_new_tokens=32768,)

result = tokenizer.decode(response[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("Response:", result)

