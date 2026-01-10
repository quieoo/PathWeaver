# 向量RAG检索

````bash
python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/merge_data/merged_dev.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10

nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/hotpot_2hop_test.json     --dataset-type hotpot     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_hotpot_2hop_1.1.log 2>&1 &

nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_prepare.json     --dataset-type musique     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_musique_2hop_1.1.log 2>&1 &

````


# 图增强的RAG检索

````bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=1
# export HF_ENDPOINT=https://hf-mirror.com
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16

nohup python graph_rag.py \
  --n-samples 100 \
  --model-path /home/sdu/zhu/models/llama3_8B_instruct \
  --similarity-top-k 10 \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/graph_rag \
  --test_dataset 2wiki.json \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_hnsw \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B >> graph_rag.log 2>&1 &

nohup python graph_rag.py \
  --n-samples 100 \
  --model-path /home/sdu/zhu/models/llama3_8B_instruct \
  --similarity-top-k 10 \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/graph_rag \
  --test_dataset hotpot_2hop.json \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/hotpot_2hop_hnsw \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B >> graph_rag.log 2>&1 &

nohup python graph_rag.py \
  --n-samples 100 \
  --model-path /home/sdu/zhu/models/llama3_8B_instruct \
  --similarity-top-k 10 \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/graph_rag \
  --test_dataset musique_2hop.json \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/musique_2hop_hnsw \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B >> graph_rag.log 2>&1 &

````

# PathWeaver
````bash
conda activate kblam_tf457
export CUDA_VISIBLE_DEVICES=1
# export HF_ENDPOINT=https://hf-mirror.com
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16


nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_2wiki_1.1.log 2>&1 &

nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/hotpot_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_12999_encoder/encoder.pt  \
    --model_dir ./train/hotpot_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_12999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop \
    --test_dataset test_datasets.jsonl \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_hotpot_2hop_1.1.log 2>&1 &

nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299_encoder/encoder.pt  \
    --model_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop \
    --test_dataset test_datasets_triples.jsonl \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_musique_2hop_1.1.log 2>&1 &
````