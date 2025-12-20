cd /mnt/n0/KBLAM/KBLaM/experiments/
export CUDA_VISIBLE_DEVICES=0


python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 4000 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --model_save_dir ./train/2wiki1_1.0 

nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000   --N 9999999   --model_dir_to_resume ./train/2wiki1_1.0/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_3999 \
  --model_save_dir ./train/2wiki1_1.0 >> ./train/2wiki_1.0.log 2>&1 &
