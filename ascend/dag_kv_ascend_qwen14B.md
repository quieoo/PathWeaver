# DAG-KV Train

````bash
python experiments/train.py \
  --seed 1 --B 1 --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-14B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 10 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa \
  --save_period 200 --keep_top_k_ckpt 5
````
## test batch size (B=2✅)

````bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B1
nohup python experiments/train.py \
  --seed 1 --B 1 --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-14B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B1 \
  --save_period 200 --keep_top_k_ckpt 5 \
  > experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B1/training_log.txt 2>&1 &

export CUDA_VISIBLE_DEVICES=2
mkdir -p experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2
nohup python experiments/train.py \
  --seed 1 --B 2 --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-14B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2 \
  --save_period 200 --keep_top_k_ckpt 5 \
  > experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2/training_log.txt 2>&1 &

export CUDA_VISIBLE_DEVICES=3
mkdir -p experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B4
nohup python experiments/train.py \
  --seed 1 --B 4 --lr 5e-4 --use_lr_decay --gradient_accm_step 10 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-14B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B4 \
  --save_period 200 --keep_top_k_ckpt 5 \
  > experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B4/training_log.txt 2>&1 &
````

## B=2, more step
````bash
export CUDA_VISIBLE_DEVICES=2
mkdir -p experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1
nohup python experiments/train.py \
  --seed 1 --B 2 --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-14B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 16000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1 \
  --save_period 200 --keep_top_k_ckpt 5 \
  > experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1/training_log.txt 2>&1 &
````

# Test

## DAG-KV inference on Ascend NPU

Run the following setup and function inside the already running Docker
container. Choose an idle physical NPU; it becomes logical `npu:0` in Python.

```bash
export ASCEND_RT_VISIBLE_DEVICES=7
export DAG_KV_DEVICE=npu
export PYTHONPATH=/workspace/dag_kv_ascend/PathWeaver/src:${PYTHONPATH:-}

DATASET_DIR=/workspace/dag_kv_ascend/datasets/test_set
MODEL_DIR=/workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800
ENCODER_DIR=/workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800_encoder/encoder.pt
LLM_BASE_DIR=/workspace/dag_kv_ascend/models/qwen3-14B-Instruct

run_dag_kv_eval() {
  local dataset="$1"
  local embedding_prefix="$2"

  python /workspace/dag_kv_ascend/PathWeaver/experiments/eval_generation.py generation \
    --dataset_dir "${DATASET_DIR}" \
    --test_dataset "${dataset}" \
    --model_dir "${MODEL_DIR}" \
    --encoder_dir "${ENCODER_DIR}" \
    --encoder_spec qwen-embedding-0.6B \
    --llm_base_dir "${LLM_BASE_DIR}" \
    --llm_type qwen3 \
    --dataset_type dag \
    --precomputed_embed_keys_path "${DATASET_DIR}/${embedding_prefix}_qwen-embedding-0.6B_embd_key.npy" \
    --precomputed_embed_values_path "${DATASET_DIR}/${embedding_prefix}_qwen-embedding-0.6B_embd_value.npy" \
    --kb_layer_frequency 3 \
    --dag_kb_size 1 \
    --query_size 100 \
    --path_attn \
    --step 15800 \
    --t_step 16000 \
    --kb_scale_factor 4 \
    --no-full_eval
}
```

Run one dataset at a time:

```bash
# PopQA
run_dag_kv_eval popqa.jsonl popqa

# SQuAD v4
run_dag_kv_eval squad_v4.jsonl squad_v4

# 2WikiMultiHopQA
run_dag_kv_eval \
  2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl \
  2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa

# HotpotQA
run_dag_kv_eval \
  hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl \
  hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa

# MuSiQue
run_dag_kv_eval \
  musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl \
  musique_dev_tripled_v5-qwen3.5-27B_dag_aa

# MintQA (2-hop)
run_dag_kv_eval \
  mintqa_pruned64_hop2_dag_aa.jsonl \
  mintqa_pruned64_hop2_dag_aa
```

The commands run in the foreground. The `query_size` value can be increased
only after verifying available NPU HBM.
