# TTFT vs Accuracy

配置API调用judge model: export DASHSCOPE_API_KEY=xxx

## w/o knowledge
````bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2

nohup python ../../experiments/vector_rag.py  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/  --n-samples 100  --similarity-top-k 16 --without-knowledge >> overall_wo_kb_2wiki_llama8b_bge.log 2>&1 &

nohup python ../../experiments/vector_rag.py  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_v1.json    --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/  --n-samples 100  --similarity-top-k 16 --without-knowledge > overall_wo_kb_hotpot_llama8b_bge.log 2>&1 &



````
### qwen2.5-72B-4bit

````bash
python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/qwen2.5-72B-4bit   --n-samples 100  --without-knowledge

````

## Oracle

````bash
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16
source /mnt/n0/uv_envs/kblam-rag/bin/activate
export CUDA_VISIBLE_DEVICES=1


nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_dataset.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_queryset.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/popqa_bge \
  >> overall_oracle_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_train.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/squad_bge \
  >> overall_oracle_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --index-path ../../experiments/vector_rag_index/2wiki_bge \
  >> overall_oracle_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_newQ/2wiki_dev_newqa_v4.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --index-path ../../experiments/vector_rag_index/2wiki_bge \
  >> overall_oracle_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --index-path ../../experiments/vector_rag_index/hotpot_bge \
  >> overall_oracle_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --index-path ../../experiments/vector_rag_index/musique_bge \
  >> overall_oracle_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/multi-hop/multihoprag/merged_queries.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/multi_hoprag_bge \
  >> overall_oracle_qwen4b_instruct_bge.log 2>&1 &


````

## vector-rag

RAG先构建embedding再建FAISS索引，保存在本地。
第一次先用CUDA装载embedding模型，保存索引。第二次再用CPU装载embedding模型，从索引中检索，避免显存溢出。

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
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_newQ/2wiki_dev_new_qa.json \
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

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/multi-hop/multihoprag/merged_queries.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 131072 \
  --index-path ../../experiments/vector_rag_index/multi_hoprag_bge \
  >> overall_vector_rag_qwen4b_instruct_bge.log 2>&1 &


````

## MSA
````bash
source /mnt/n0/uv_envs/msa/bin/activate

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
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_dev.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/squad_doc_cache \
  --max-batch-size 1 \
  --seed 2 \
  >> overall_msa.log 2>&1 &


nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/2wiki_10kdoc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &
# {'rouge1': 0.6650522695890562, 'rouge2': 0.4383809523809524, 'rougeL': 0.6643679198703291, 'rougeLsum': 0.6646646363554257, 'exact_match': 0.59, 'f1_overlap': 0.6515725960935376, 'faithfulness01': 0.64}

nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_newQ/2wiki_dev_first_samples.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/2wiki_10kdoc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &

# {'rouge1': 0.7355555555555555, 'rouge2': 0.2864285714285715, 'rougeL': 0.7349444444444444, 'rougeLsum': 0.7328888888888889, 'exact_match': 0.67, 'f1_overlap': 0.7286666666666668, 'faithfulness01': 0.69}

nohup python ../../experiments/msa.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_newQ/2wiki_dev_newqa_v4.json \
  --model-path /mnt/n0/models/MSA-4B \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir ../../experiments/msa_cache/2wiki_10kdoc_cache \
  --max-batch-size 1 \
  >> overall_msa.log 2>&1 &

# {'rouge1': 0.508631377926644, 'rouge2': 0.3510044563279857, 'rougeL': 0.509932942394667, 'rougeLsum': 0.5136349823068838, 'exact_match': 0.45, 'f1_overlap': 0.5123463575429661, 'faithfulness01': 0.49}



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

## LMCache
````bash
source /mnt/n0/uv_envs/lmcache/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

# # 测试
# python3 ../../experiments/vector_rag.py \
#   --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1_10.json \
#   --model-path /mnt/n0/models/qwen3-4B-Instruct \
#   --embedding-model /mnt/n0/models/bge-en-v1.5/ \
#   --n-samples 10 \
#   --similarity-top-k 16 \
#   --index-path ../../experiments/vector_rag_index/hotpot_clean_10_bge \
#   --max-model-len 32768 \
#   --use-lmcache \
#   --recompute-ratios 0.25 \
#   >> overall_vector_rag_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_dataset.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_queryset.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --use-lmcache \
  --recompute-ratios 0.15 \
  --lmcache-warmup-mode reuse \
  --index-path ../../experiments/vector_rag_index/popqa_bge \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_train.json \
  --dataset-type 2wiki \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --use-lmcache \
  --recompute-ratios 0.15 \
  --lmcache-warmup-mode reuse \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/squad_bge \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &


nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --lmcache-warmup-mode reuse \
  --index-path ../../experiments/vector_rag_index/2wiki_bge \
  --use-lmcache \
  --recompute-ratios 0.15 \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --lmcache-warmup-mode reuse \
  --index-path ../../experiments/vector_rag_index/hotpot_bge \
  --use-lmcache \
  --recompute-ratios 0.15 \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &

nohup python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_dev.jsonl \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path ../../experiments/vector_rag_index/musique_bge \
  --use-lmcache \
  --lmcache-warmup-mode reuse \
  --recompute-ratios 0.15 \
  >> overall_lmcache_qwen4b_instruct_bge.log 2>&1 &




````

## graph-rag

````bash
conda activate vllm-13
export CUDA_VISIBLE_DEVICES=2
python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/llama3_8B_instruct \
  --served-model-name llama_8b \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192


curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama_8b",
    "messages": [
      {"role": "user", "content": "你是什么模型"}
    ],
    "temperature": 0.7
  }'

conda activate autoschemakg
export CUDA_VISIBLE_DEVICES=3
nohup python  ../../../AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
  --kg-path /mnt/n0/KBLAM/AutoSchemaKG/example/generated/2wiki_dev/ \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --dataset-keyword 2wiki_dev.json \
  --encoder-model /home/sdu/zhu/models/bge-en-v1.5/ \
  --llm-model llama_8b \
  --test-samples 100 \
  --llm-endpoint 'http://127.0.0.1:8001/v1' \
  --topN 16 > overall_graph_rag_2wiki_llama8b_bge.log 2>&1 &

conda activate autoschemakg
export CUDA_VISIBLE_DEVICES=3
nohup python  ../../../AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
  --kg-path /mnt/n0/KBLAM/AutoSchemaKG/example/generated/hotpot_dev/ \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_v1.json\
  --dataset-keyword hotpot_dev.json \
  --encoder-model /home/sdu/zhu/models/bge-en-v1.5/ \
  --llm-model llama_8b \
  --test-samples 100 \
  --llm-endpoint 'http://127.0.0.1:8001/v1' \
  --topN 16 > overall_graph_rag_hotpot_llama8b_bge.log 2>&1 &
````
### qwen2.5-72B-4bit + bge-embedding

````bash
conda activate vllm-13
export CUDA_VISIBLE_DEVICES=2,3
nohup python -m vllm.entrypoints.openai.api_server \
  --model /mnt/n0/models/qwen2.5-72B-4bit \
  --served-model-name qwen_72b \
  --host 0.0.0.0 \
  --enforce-eager \
  --port 8001 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 > /mnt/n0/PathWeaver/experiments/qwen2.5-72B-4bit.log 2>&1 &

curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen_72b",
    "messages": [
      {"role": "user", "content": "你是什么模型"}
    ],
    "temperature": 0.7
  }'

conda activate autoschemakg
export CUDA_VISIBLE_DEVICES=3
nohup python  ../../../AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
  --kg-path /mnt/n0/KBLAM/AutoSchemaKG/example/generated/2wiki_dev/ \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --dataset-keyword 2wiki_dev.json \
  --llm-model qwen_72b \
  --encoder-model /home/sdu/zhu/models/bge-en-v1.5/ \
  --test-samples 100 \
  --llm-endpoint 'http://127.0.0.1:8001/v1' \
  --topN 16 > overall_graph_rag.log 2>&1 &
````


## PathWeaver

````bash
conda activate kblam_tf457

nohup python ../../experiments/eval_generation.py generation     --kb_size=10     --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3     --encoder_spec qwen-embedding-0.6B     --encoder_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800_encoder/encoder.pt     --model_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800     --kb_layer_frequency 3 --kb_scale_factor 1     --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples     --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json     --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy     --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy     --step 0 --t_step 8000     --dataset_type at2qa_2wiki --query_size 100 --seed 1 --path_attn >> overall_pathweaver_2wiki_llama8b_qwen.log 2>&1 &

nohup python experiments/eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir experiments/train/dag_kv_hotpot_v5.2.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7300_encoder/encoder.pt \
    --model_dir experiments/train/dag_kv_hotpot_v5.2.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7300 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv \
    --test_dataset hotpot_dev_dag_v5.2_cleaned_v1.json \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type dag --query_size 100 --seed 1 --path_attn > EXPs/overall_pathweaver_hotpot_llama8b_qwen.log 2>&1 &

````
### bge-embedding
````bash
python ../../experiments/eval_generation.py generation     --kb_size=10     --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3     --encoder_spec bge     --encoder_dir /home/sdu/zhu/kblam/train/atfb_2wiki_bge_v1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_bge_at2qa_2wiki_llama3_step_7400_encoder/encoder.pt     --model_dir /home/sdu/zhu/kblam/train/atfb_2wiki_bge_v1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_bge_at2qa_2wiki_llama3_step_7400     --kb_layer_frequency 3 --kb_scale_factor 1     --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples     --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json     --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_bge-en-v1.5_embd_key.npy     --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_bge-en-v1.5_embd_value.npy     --step 2400 --t_step 8000     --dataset_type at2qa_2wiki --query_size 100 --seed 1 --path_attn 

````


## KBLaM

````bash
# 关闭path_attn

nohup python ../../experiments/eval_generation.py generation     --kb_size=10     --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3     --encoder_spec qwen-embedding-0.6B     --encoder_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800_encoder/encoder.pt     --model_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800     --kb_layer_frequency 3 --kb_scale_factor 1     --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples     --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json     --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy     --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy     --step 0 --t_step 8000     --dataset_type at2qa_2wiki --query_size 100 --seed 1 >> overall_kblam_2wiki_llama8b_qwen.log 2>&1 &
nohup python experiments/eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir experiments/train/dag_kv_hotpot_v5.2.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7300_encoder/encoder.pt \
    --model_dir experiments/train/dag_kv_hotpot_v5.2.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7300 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv \
    --test_dataset hotpot_dev_dag_v5.2_cleaned_v1.json \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type dag --query_size 100 --seed 1 > EXPs/overall_kblam_hotpot_llama8b_qwen.log 2>&1 &
````



