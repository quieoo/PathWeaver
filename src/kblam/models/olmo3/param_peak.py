import json
from safetensors.torch import load_file

MODEL_DIR = "/home/sdu/zhu/models/olmo3-7b/"

# 任选一个 shard 就够
sd = load_file(f"{MODEL_DIR}/model-00001-of-00003.safetensors")

# 打印前 100 个 key
for i, k in enumerate(sd.keys()):
    print(k)
    if i >= 100:
        break
