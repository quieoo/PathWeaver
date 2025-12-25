import json

input_path = "/mnt/n0/yyl/KBLaM/datasets/wiki/memory_kb.json"
output_path = "/mnt/n0/yyl/KBLaM/datasets/wiki/memory_kb_fixed.json"

data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 已修复文件，输出到 {output_path}")
