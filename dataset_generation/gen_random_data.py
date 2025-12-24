import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm

def generate_memory_based_kb(num_items=10000, embedding_dim=45056, save_path="../datasets/wiki/memory_kb.json"):
    """
    生成基于“随机内存块”的 KB, 每条记录的 key/value 都来源于伪内存内容
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(num_items), desc="生成内存随机KB"):
            # 在CPU上模拟一段随机内存数据（字节流）
            random_bytes = os.urandom(embedding_dim // 8)  # 每条KB约几KB大小
            # 将内存块转为16进制字符串（或base64，防止乱码）
            mem_text = random_bytes.hex()

            # 构造一条KB记录
            entry = {
                "key_string": mem_text,
                "description": mem_text
            }
            f.write(json.dumps(entry) + "\n")

    print(f"✅ 已生成 {num_items} 条基于内存数据的随机KB, 保存到 {save_path}")


if __name__ == "__main__":
    generate_memory_based_kb(num_items=10000)
