import json

# 原始文件路径
input_file = "/mnt/n0/yyl/KBLaM/datasets/wiki/memory_kb_fixed.json"
# 输出文件路径
output_file = "/mnt/n0/yyl/KBLaM/datasets/wiki/memory_kb_extract.json"

# 读取原始 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 如果文件中是一个列表形式（如 [entry1, entry2, ...]）
# 提取前 10 条
sampled_data = data[:10]

# 保存到新文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=4)

print(f"✅ 已提取 {len(sampled_data)} 条数据，保存到 {output_file}")
