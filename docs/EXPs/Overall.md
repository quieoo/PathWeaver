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
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2


nohup python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model all-MiniLM-L6-v2     --n-samples 100  --similarity-top-k 16 --oracle-retrieval > overall_oracle.log 2>&1 &

````

### qwen2.5-72B-4bit

````bash
python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/qwen2.5-72B-4bit   --n-samples 100  --oracle-retrieval

````

## vector-rag

RAG先构建embedding再建FAISS索引，保存在本地。
第一次先用CUDA装载embedding模型，保存索引。第二次再用CPU装载embedding模型，从索引中检索，避免显存溢出。

````bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2

nohup python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/     --n-samples 100  --similarity-top-k 16 --index-path ../../experiments/vector_rag_index/2wiki_bge >> overall_vector_rag_2wiki_llama8b_bge.log 2>&1 &


nohup python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/     --n-samples 100  --similarity-top-k 16 --index-path ../../experiments/vector_rag_index/hotpot_bge --embedding-device cuda >> overall_vector_rag_hotpot_llama8b_bge.log 2>&1 &
nohup python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_v1.json    --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/     --n-samples 100  --similarity-top-k 16 --index-path ../../experiments/vector_rag_index/hotpot_bge --embedding-device cpu >> overall_vector_rag_hotpot_llama8b_bge.log 2>&1 &
````

### qwen-72B-int4 + bge-embedding

````bash
export CUDA_VISIBLE_DEVICES=0,1

python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/qwen2.5-72B-4bit     --embedding-model all-MiniLM-L6-v2     --n-samples 100  --similarity-top-k 16 --index-path ../../experiments/vector_rag_index/2wiki_allmini


## qwen72B + BGE-embedding
python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/qwen2.5-72B-4bit     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/     --n-samples 100  --similarity-top-k 16 --index-path ../../experiments/vector_rag_index/2wiki_bge
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
export CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/qwen2.5-72B-4bit \
  --served-model-name qwen_72b \
  --host 0.0.0.0 \
  --enforce-eager \
  --port 8001 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95

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
export CUDA_VISIBLE_DEVICES=2
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