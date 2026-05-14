# SETUP

````bash
cd /mnt/n0/uv_envs
uv venv msa --python 3.12
source /mnt/n0/uv_envs/msa/bin/activate

git clone https://github.com/quieoo/MSA.git
cd MSA
uv pip install -r requirements.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

mkdir /mnt/n0/models/MSA-4B
huggingface-cli download --resume-download EverMind-AI/MSA-4B --local-dir /mnt/n0/models/MSA-4B

# Run inference on benchmarks
bash scripts/run_benchmarks.sh eval_benchmark

# Compute LLM-based scores
bash scripts/calculate_llm_score.sh eval_benchmark
````
# 与现有测试框架兼容
- 修复msa_service中的deserialize，保证存盘的Memory可以被捞回来。
- dataset.py, 数据处理
- metrics_evaluator.py，通用的评测接口
- msa.py, 按照MSA/benchmark.py相同的方式调用MSA

````bash

nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_dataset.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_queryset.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/popqa_doc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &

nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_train.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/squad_doc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &


nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/2wiki_10kdoc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &

nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/hotpot_doc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &

nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_dev.jsonl \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/musique_doc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &
````

## max-length
设置max_generate_tokens基本没什么影响
````bash
python PathWeaver/experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/MSA-4B \
  --memory-docs 9000 \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir /home/sdu/zhu/kblam/msa_cache/2wiki_10kdoc_cache \
  --max-batch-size 1 \
  --max-length 64 

````

# 与vector-rag对比
````bash

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_dataset.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_queryset.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/popqa_bge \
  >> overall_vector_rag_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_train.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/squad_bge \
  >> overall_vector_rag_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/2wiki_bge \
  >> overall_vector_rag_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/hotpot_bge \
  >> overall_vector_rag_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_dev.jsonl \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/musique_bge \
  >> overall_vector_rag_qwen4b_instruct_bge.log 2>&1 &


````


# Tail Knowledge
使用新的MultiHopRAG数据集

````bash
export CUDA_VISIBLE_DEVICES=1

source /mnt/n0/uv_envs/kblam-rag/bin/activate
nohup python3 PathWeaver/docs/experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/multi-hop/multihoprag/merged_queries.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path PathWeaver/docs/experiments/vector_rag_index/multi_hoprag_bge \
  >> overall_vector_rag_multi_hoprag_qwen4b_instruct_bge.log 2>&1 &


source /mnt/n0/uv_envs/msa/bin/activate
export CUDA_VISIBLE_DEVICES=1
python PathWeaver/experiments/msa.py \
  --dataset-path /mnt/n0/datasets/multi-hop/multihoprag/merged_queries.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/MSA-4B \
  --memory-docs 10000 \
  --n-samples 100 \
  --block-size 1024 \
  --memory-cache-dir /home/sdu/zhu/kblam/msa_cache/multi_hoprag_cache \
  --max-batch-size 1 \
  --max-length 64

````
