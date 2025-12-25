import json

# 输入与输出文件路径
input_file = "../datasets/extend/test_datasets.json"     # 原始数据文件
output_file = "../datasets/extend/test_datasets_converted.json"  # 输出结果文件

def convert_dataset(input_file, output_file):
    # 读取原始数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    # 遍历每个样本
    for sample in data:
        if "paragraphs" not in sample:
            continue
        
        for para in sample["paragraphs"]:
            triples = para.get("triples", [])
            
            for triple in triples:
                head = triple.get("Head", "").strip()
                relation = triple.get("Relation", "").strip()
                
                if not head or not relation:
                    continue
                
                # 构造描述、问题与键字符串
                desc = f"the {relation} of {head}"
                Q = f"What is the {relation} of {head}?"
                key_string = desc

                results.append({
                    # "description": desc,
                    "Q": Q,
                    "key_string": key_string
                })

    # 保存结果为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 已成功转换 {len(results)} 条三元组，并保存到：{output_file}")

# 执行转换
if __name__ == "__main__":
    convert_dataset(input_file, output_file)
