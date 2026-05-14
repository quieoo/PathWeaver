# LMCache Setup

````bash
# 准备实验环境
cd uv_envs/lmcache
uv venv --python 3.12
source .venv/bin/activate
uv pip install "vllm==0.10.0"
````

需要修改vllm代码：
In vllm/vllm/v1/worker/gpu_worker.py, comment out ensure_kv_transfer_initialized(vllm_config) in function def init_worker_distributed_environment.
In the same file, add
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.integration.vllm.utils import ENGINE_NAME
        
VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
ensure_kv_transfer_initialized(self.vllm_config)
at the end of the function def load_model.

从源码编译安装LMCache：
````bash
source uv_envs/lmcache/.venv/bin/activate

git clone https://github.com/LMCache/LMCache.git
cd LMCache
git checkout v0.3.9
uv pip install -v --no-build-isolation .

python - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)
import lmcache.c_ops
print("lmcache.c_ops import ok")
PY
````

简单测试：
````bash
source /mnt/n0/uv_envs/lmcache/.venv/bin/activate

cd experiments/lmcache

# 创建一个配置文件 blending.yaml
export CUDA_VISIBLE_DEVICES=2
python blend.py --model /mnt/n0/models/llama3_8B_instruct/
python blend.py --model /mnt/n0/models/qwen3-4B/

mkdir /mnt/n0/PathWeaver/experiments/lmcache/persisted_kv
python blend.py \
  --model /mnt/n0/models/qwen3-4B/ 


````

benchmarking测试：
````bash
LMCACHE_CONFIG_FILE=blending.yaml vllm serve /mnt/n0/models/qwen3-4B/ --gpu-memory-utilization 0.8 --port 8000 --no-enable-prefix-caching --kv-transfer-config --enforce-eager '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/n0/models/qwen3-4B/",
    "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts.  # #  Qwen3 is a state-of-the-art model that can handle complex tasks with high accuracy. ",
    "max_tokens": 100,
    "temperature": 0.7
  }'

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/n0/models/qwen3-4B/",
    "prompt": "Qwen3 is a state-of-the-art model that can handle complex tasks with high accuracy.  # #  Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
    "max_tokens": 100,
    "temperature": 0.7
  }'

````

# LMCache for RAG

````bash
source /mnt/n0/uv_envs/lmcache/.venv/bin/activate

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 64 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.8 \
  --use-lmcache \
  --lmcache-warmup-mode full \
  --recompute-ratios 0.05 \
  --index-path ../../experiments/vector_rag_index/2wiki_bge \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 64 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.8 \
  --use-lmcache \
  --recompute-ratios 0.3 \
  --index-path ../../experiments/vector_rag_index/hotpot_bge \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_dev.jsonl \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --gpu-memory-utilization 0.8 \
  --n-samples 100 \
  --similarity-top-k 64 \
  --max-model-len 65536 \
  --use-lmcache \
  --recompute-ratios 0.6 \
  --index-path ../../experiments/vector_rag_index/musique_bge \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
python3 ../../experiments/vector_rag.py   --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json   --model-path /mnt/n0/models/qwen3-4B-Instruct   --embedding-model /mnt/n0/models/bge-en-v1.5/   --n-samples 100   --similarity-top-k 64   --max-model-len 65536   --gpu-memory-utilization 0.8   --use-lmcache   --lmcache-warmup-mode reuse   --recompute-ratios 0.05   --index-path ../../experiments/vector_rag_index/2wiki_bge
python3 ../../experiments/vector_rag.py   --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json   --model-path /mnt/n0/models/qwen3-4B-Instruct   --embedding-model /mnt/n0/models/bge-en-v1.5/   --n-samples 100   --similarity-top-k 64   --max-model-len 65536   --gpu-memory-utilization 0.8   --use-lmcache   --lmcache-warmup-mode full   --recompute-ratios 0.8   --index-path ../../experiments/vector_rag_index/2wiki_bge
python3 ../../experiments/vector_rag.py   --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json   --model-path /mnt/n0/models/qwen3-4B-Instruct   --embedding-model /mnt/n0/models/bge-en-v1.5/   --n-samples 100   --similarity-top-k 64   --max-model-len 65536   --gpu-memory-utilization 0.8   --use-lmcache   --lmcache-warmup-mode full   --recompute-ratios 1.0   --index-path ../../experiments/vector_rag_index/2wiki_bge

````

# LMCache_jiawei
- /mnt/n0/uv_envs/lmcache/build_from_source/LMCache_jiawei/LMCache/
- /mnt/n0/uv_envs/lmcache/build_from_source/LMCache/