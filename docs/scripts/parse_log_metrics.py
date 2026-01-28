import re

# log_file="../experiments/train_at2qa_2wiki_4_2.log"

log_file="../experiments/train_at2qa_2wiki_4_9_2.log"


rouge1_values = []

with open(log_file, "r", encoding="utf-8") as f:
    text = f.read()

# 匹配 rouge1: 数值
pattern = re.compile(r"'rouge1'\s*:\s*([0-9]*\.?[0-9]+)")

for m in pattern.finditer(text):
    rouge1_values.append(float(m.group(1)))

for r in rouge1_values:
    print(r)

print(f"共解析到 {len(rouge1_values)} 条 rouge1")
print("平均 rouge1 =", sum(rouge1_values) / len(rouge1_values))
