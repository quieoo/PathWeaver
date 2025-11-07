## evaluate while train

````bash
# 训练Synthetic观察效果

# 使用Synthetic3的训练脚本
nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 4800 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path ../datasets/synthetic_embd/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path ../datasets/synthetic_embd/train_datasets_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path ../datasets/synthetic_embd/train_datasets_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/debug_train_evaluate \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  --test_data_path ../datasets/synthetic_embd/test_synthetic.json \
  --test_precomputed_embed_keys_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 20 \
  >> train_evaluate.log 2>&1 &


# 开启验证之后模型Loss趋势改变，同时验证精度始终较低。关闭验证后，模型Loss趋势恢复正常。
  nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 4800 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path ../datasets/synthetic_embd/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path ../datasets/synthetic_embd/train_datasets_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path ../datasets/synthetic_embd/train_datasets_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/debug_train_evaluate \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  >> train_evaluate.log 2>&1 &

# 正常推理下得分：'rouge1': 0.6742802404134849
nohup python eval.py generation \
    --eval_mode kb --kb_size=10 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/debug_train_evaluate/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4799_encoder/encoder.pt \
    --model_dir ./train/debug_train_evaluate/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4799 \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir ../datasets/synthetic_embd \
    --test_dataset test_synthetic.json \
    --precomputed_embed_keys_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed 1 >> eval_kblam.log 2>&1 &



````

## squad 数据集
准备好squad数据集，转换为synthetic格式

创建embedding，为了控制变量，先使用旧版的KBLaM创建方法和all-MiniLM-L6-v2模型
````bash
python generate_kb_embeddings.py \
    --model_name all-MiniLM-L6-v2 \
    --dataset_name squad_train --dataset_path /mnt/n0/datasets/squad/train_datasets.json \
    --output_path /mnt/n0/datasets/squad/

python generate_kb_embeddings.py \
    --model_name all-MiniLM-L6-v2 \
    --dataset_name squad_test --dataset_path /mnt/n0/datasets/squad/test_datasets.json \
    --output_path /mnt/n0/datasets/squad/

````


训练
````bash
nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 4800 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/debug_squad \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  >> train_squad.log 2>&1 &


# 推理测试,推理得分基本上和训练时的验证得分一致
nohup python eval.py generation \
    --eval_mode kb --kb_size=10 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/debug_squad/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4799_encoder/encoder.pt  \
    --model_dir ./train/debug_squad/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4799  \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed 1 >> eval_squad.log 2>&1 &

# 验证直接训练
nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 5000 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/debug_squad \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  >> train_squad.log 2>&1 &
````