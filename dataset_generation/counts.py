count = 0
with open("../datasets/wiki/wiki.json", "r", encoding="utf-8") as f:
    for _ in f:
        count += 1

print("行数（即 JSON 对象数）：", count)
