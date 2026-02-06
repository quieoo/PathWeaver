# TTFT vs Accuracy

## w/o knowledge
````bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2

nohup python ../../experiments/vector_rag.py  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model all-MiniLM-L6-v2     --n-samples 100  --similarity-top-k 16 --without-knowledge >> overall_wo_kb.log 2>&1 &

````


## Oracle

````bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2


nohup python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model all-MiniLM-L6-v2     --n-samples 100  --similarity-top-k 16 --oracle-retrieval > overall_oracle.log 2>&1 &

````

## vector-rag

````bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2


nohup python ../../experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model all-MiniLM-L6-v2     --n-samples 100  --similarity-top-k 16 >> overall_vector_rag.log 2>&1 &


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
  \
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
nohup python  ../../../AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
  --kg-path /mnt/n0/KBLAM/AutoSchemaKG/example/generated/2wiki_dev/ \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --dataset-keyword 2wiki_dev.json \
  --llm-model llama_8b \
  --test-samples 100 \
  --llm-endpoint 'http://127.0.0.1:8001/v1' \
  --topN 16 > overall_graph_rag.log 2>&1 &
````
## PathWeaver

````bash
conda activate kblam_tf457

python ../../experiments/eval_generation.py generation     --kb_size=10     --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3     --encoder_spec qwen-embedding-0.6B     --encoder_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800_encoder/encoder.pt     --model_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800     --kb_layer_frequency 3 --kb_scale_factor 1     --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples     --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json     --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy     --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy     --step 2400 --t_step 8000     --dataset_type at2qa_2wiki --query_size 100 --seed 1 --path_attn 

````

## KBLaM

````bash
# 关闭path_attn

nohup python ../../experiments/eval_generation.py generation     --kb_size=10     --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3     --encoder_spec qwen-embedding-0.6B     --encoder_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800_encoder/encoder.pt     --model_dir /home/sdu/zhu/kblam/train/atfb_2wiki_v3/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7800     --kb_layer_frequency 3 --kb_scale_factor 1     --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples     --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json     --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy     --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy     --step 2400 --t_step 8000     --dataset_type at2qa_2wiki --query_size 100 --seed 1 >> overall_kblam.log 2>&1 &

````