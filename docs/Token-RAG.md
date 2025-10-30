
# 准备
准备阶段，相关代码在“tools”目录下。

## 生成三元组
````bash
# 生成前6000个样本的三元组
nohup python triples_gen.py \
    --dataset_type musique \
    --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl \
    --model_path /mnt/n0/models/llama3_8B_instruct \
    --batch_size 20 --num_sample 6000 \
    > triple.log 2>&1 &

# 生成从6000开始的所有样本的三元组，每一千条样本保存一次结果
nohup python triples_gen.py \
    --dataset_type musique \ 
    --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl \
    --model_path /mnt/n0/models/llama3_8B_instruct \
    --batch_size 20 --start_from 6000 --save_every 1000 \
    > triple_v2.log 2>&1 &

# 创建测试集三元组
export CUDA_VISIBLE_DEVICES=1
python triples_gen.py \
    --dataset_type musique \
    --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl \
    --model_path /mnt/n0/models/llama3_8B_instruct \
    --batch_size 20 --num_sample 100


````
生成位于dataset_path同目录下的新文件，文件名包含"triples"


## 生成数据集
````bash
python datasets_gen.py -t musique -p1 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl -p2 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train_triple_llama3_8B_instruct_inst6_num_sample6000.json -p3 ../datasets/musique_train_6000/train_datasets.json

python datasets_gen.py -t musique -p1 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl -p2 /mnt/n0/datasets/MuSiQue/musique_train_triple_19938.json -p3 ../datasets/musique_train_19938/train_datasets.json

# 测试数据集
python datasets_gen.py -t musique -p1 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl -p2 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev_triple_llama3_8B_instruct_inst6_0_100.json -p3 ../datasets/musique_dev_100/test_datasets.json

````
功能：将三元组内容合并到原始数据集，生成新的训练用数据集
参数说明：
    -p1: 原始数据集路径
    -p2: 三元组文件路径
    -p3: 输出数据集路径


## 生成KB-Embedding
````bash
python embedding.py \
    --dataset_type musique \
    --model_name all-MiniLM-L6-v2 \
    --dataset_path ../datasets/musique_train_6000/train_datasets.json

python embedding.py \
    --dataset_type musique \
    --model_name all-MiniLM-L6-v2 \
    --dataset_path ../datasets/musique_dev_100/test_datasets.json


# 这个版本更快
python embedding_v2.py     --dataset_type musique     --model_name all-MiniLM-L6-v2     --dataset_path ../datasets/musique_train_19938/train_datasets.json --batch_size 1024

# 保持测试集一致
python embedding_v2.py \
    --dataset_type musique \
    --model_name all-MiniLM-L6-v2 \
    --dataset_path ../datasets/musique_dev_100/test_datasets.json \
    --batch_size 1024

# 使用qwen3模型#qwen需要更高级别的transformers库

conda activate qwen-embedding 
# 开启并行处理，避免警告
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type musique \
  --dataset_path ../datasets/musique_dev_100/test_datasets.json \
  --batch_size 1024

nohup python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type musique \
  --dataset_path ../datasets/musique_train_19938/train_datasets.json \
  --batch_size 2048 > embed.log 2>&1 &
````
功能：
    - 生成KB-Embedding，存储在dataset_path同目录下
    - 在原始数据集中添加每个样本、每个段落的三元组偏移量（训练时候需要）

# 训练

训练脚本：experiments/train.py

````bash
nohup python train.py   --seed 1 --B 1 --lr 5e-4   --sep_query_head --use_cached_embd --use_lr_decay   --kb_token_layer_frequency 1   --encoder_spec all-MiniLM-L6-v2   --key_embd_src key   --dataset_dir ../datasets/musique_train_6000/   --dataset_type musique   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/   --llm_type llama3   --model_save_dir ./train/musique_kbsize1_600_nooutlier   --gradient_accm_step 10   --save_period 100   --total_steps 600   --kb_size 1   --outlier_num -9999   --verbose > train_musique_kbsize1_600_nooutlier.log 2>&1 &


nohup python train.py   --seed 1 --B 10 --lr 5e-4   --sep_query_head --use_cached_embd --use_lr_decay   --kb_token_layer_frequency 1   --encoder_spec all-MiniLM-L6-v2   --key_embd_src key   --dataset_dir ../datasets/musique_train_19938/   --dataset_type musique   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/   --llm_type llama3   --model_save_dir ./train/musique_kbsize1_19938_nooutlier   --gradient_accm_step 10   --save_period 100   --total_steps 199   --kb_size 1   --outlier_num -9999   --verbose > train_musique_kbsize1_19938_nooutlier.log 2>&1 &


# 使用新的embedding训练
nohup python train.py   --seed 1 --B 10 --lr 5e-4   --sep_query_head --use_cached_embd --use_lr_decay   --kb_token_layer_frequency 1   --encoder_spec qwen-embedding-0.6B  --key_embd_src key   --dataset_dir ../datasets/musique_train_19938/   --dataset_type musique   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/   --llm_type llama3   --model_save_dir ./train/musique_kbsize1_19938_qwenembedding   --gradient_accm_step 10   --save_period 100   --total_steps 199   --kb_size 1   --outlier_num -9999   --verbose > train_musique_kbsize1_19938_qwenembedding.log 2>&1 &

# 设置动态增长的kb_size
# 降低学习率
# 提高梯度累积步数
# 降低KB注入频率
# 增大B
nohup python train.py   --seed 1 --B 16 --lr 1e-4   --sep_query_head --use_cached_embd --use_lr_decay   --kb_token_layer_frequency 3   --encoder_spec qwen-embedding-0.6B  --key_embd_src key   --dataset_dir ../datasets/musique_train_19938/   --dataset_type musique   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/   --llm_type llama3   --model_save_dir ./train/musique_kbsize10_14000step_qemb   --gradient_accm_step 30   --save_period 500   --total_steps 5000   --kb_size 10   --outlier_num -9999   --verbose --debug_level 1 > train_musique_kbsize10_14000step_qemb.log  2>&1 &

nohup python train.py   --seed 1 --B 16 --lr 1e-4   --sep_query_head --use_cached_embd --use_lr_decay   --kb_token_layer_frequency 1   --encoder_spec qwen-embedding-0.6B  --key_embd_src key   --dataset_dir ../datasets/musique_train_19938/   --dataset_type musique   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/   --llm_type llama3   --model_save_dir ./train/musique_kbsize20_14000step_qemb   --gradient_accm_step 30   --save_period 500   --total_steps 5000   --kb_size 20   --outlier_num -9999   --verbose --debug_level 1 > train_musique_kbsize20_14000step_qemb.log  2>&1 &



````



关键参数：
    - dataset_dir: 存储训练数据集和KB-Embedding的目录
    - dataset_type: 数据集类型
    - kb_size: 在训练时如何包含KB Tokens, 例如，在数据集类型为musique时，kb_size=-1，包含当前样本的所有段落的KB Tokens；kb_size=1，包含当前样本中与答案相关的段落的KB Tokens

# 推理

## 传统RAG

````bash

export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16

# 传统RAG，单跳检索
python llama_rag_v2.py \
    --dataset-path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl \
    --model-path /mnt/n0/models/llama3_8B_instruct \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --n-samples 100 \
    --similarity-top-k 3

# Oracle-RAG (Ground Truth Context)
python llama_rag_v2.py \
    --dataset-path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl \
    --model-path /mnt/n0/models/llama3_8B_instruct \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --n-samples 100 \
    --oracle-retrieval

# DeepSeek-R1-Distill-Qwen-32B 四卡运行
export CUDA_VISIBLE_DEVICES=2,3,4,5

python llama_rag_v2.py \
    --dataset-path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl \
    --model-path /mnt/n0/models/deepseek-r1-distill-qwen-32B \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --n-samples 100 \
    --oracle-retrieval

export CUDA_VISIBLE_DEVICES=2,3
python llama_rag_v2.py \
    --dataset-path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl \
    --model-path /mnt/n0/models/deepseek-r1-distill-qwen-14B \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --n-samples 100 \
    --oracle-retrieval

python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl     --model-
path /mnt/n0/models/qwen2.5-14B-Instruct/     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100     --oracle-retrieval


# 多跳检索
# 需要python 3.11 环境
conda activate llama311


````
## Token-RAG
````bash
python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --dataset_type musique \
    --encoder_dir ./train/musique_kbsize5_6000_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize5SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_599_encoder/encoder.pt \
    --model_dir ./train/musique_kbsize5_6000_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize5SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_599 \
    --kb_layer_frequency 1 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed -1 \
    --kb_scale_factor 1.0 --kb_size 1


python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --dataset_type musique \
    --encoder_dir ./train/musique_kbsize1_19938_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_200_encoder/encoder.pt \
    --model_dir ./train/musique_kbsize1_19938_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_200 \
    --kb_layer_frequency 1 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed -1 \
    --kb_scale_factor 0.125 --kb_size 1

# 注意kb_layer_frequency 3, 应该和训练时一样
python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --dataset_type musique \
    --encoder_dir ./train/musique_kbsize10_14000step_qemb/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_4999_encoder/encoder.pt \
    --model_dir ./train/musique_kbsize10_14000step_qemb/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_4999 \
    --kb_layer_frequency 3 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 100 --seed -1 \
    --kb_scale_factor 1 --kb_size 1


python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --dataset_type musique \
    --encoder_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_700_encoder/encoder.pt \
    --model_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_700 \
    --kb_layer_frequency 3 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 100 --seed -1 \
    --kb_scale_factor 0.55 --kb_size 1

````

# 三元组评价

````bash

# 使用阿里API
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16
python triple_quality_score.py \
  --input ../datasets/musique_dev_100/test_datasets.json \
  --output ./musique_llm_reachability.jsonl \
  --use-qwen \
  --qwen-model qwen3-max


````

# DEBUG

````bash
nohup python train.py \
  --seed 1 --B 1 --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay \
  --kb_token_layer_frequency 1 \
  --encoder_spec all-MiniLM-L6-v2 \
  --key_embd_src key \
  --dataset_dir ../datasets/musique_train_6000/ \
  --dataset_type musique \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ \
  --llm_type llama3 \
  --model_save_dir ./train/musique_debug \
  --gradient_accm_step 10 \
  --save_period 1000 \
  --total_steps 100 \
  --kb_size 1 \
  --outlier_num -9999 \
  --verbose > train_debug.log 2>&1 &

nohup python train.py \
  --seed 1 --B 1 --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay \
  --kb_token_layer_frequency 1 \
  --encoder_spec all-MiniLM-L6-v2 \
  --key_embd_src key \
  --dataset_dir ../datasets/musique_train_6000/ \
  --dataset_type musique \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ \
  --llm_type llama3 \
  --model_save_dir ./train/musique_debug \
  --gradient_accm_step 10 \
  --save_period 1000 \
  --total_steps 100 \
  --kb_size 1 \
  --outlier_num -9999 \
  --model_dir_to_resume ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4 \
  --verbose > train_debug.log 2>&1 &




python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --dataset_type musique \
    --encoder_dir ./train/musique_debug/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_99_encoder/encoder.pt \
    --model_dir ./train/musique_debug/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_99 \
    --kb_layer_frequency 3 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 1 --seed -1 \
    --kb_scale_factor 1 --kb_size 1 \
    --save_dir ./gen_output_debug 


````

## 打印 Attention Weights

````bash

python ../tools/show_attn_weights.py \
    --attn_dir ./attn_weights/ \
    --keyword "debug_kbscale1" \
    --output_dir ./attn_heatmaps

````


## 训练

````bash
nohup python train.py \
  --seed 1 --B 1 --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay \
  --kb_token_layer_frequency 3 \
  --encoder_spec all-MiniLM-L6-v2 \
  --key_embd_src key \
  --dataset_dir ../datasets/musique_train_6000/ \
  --dataset_type musique \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ \
  --llm_type llama3 \
  --model_save_dir ./train/musique_kbsize1_6000_nooutlier_debug \
  --gradient_accm_step 10 \
  --save_period 100 \
  --total_steps 550 \
  --kb_size 1 \
  --outlier_num -9999 \
  --verbose > train_musique_kbsize1_6000_nooutlier.log 2>&1 &
````

## 相同的参数下输出精度变化


````bash
nohup python eval.py generation     --eval_mode kb     --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3     --encoder_spec all-MiniLM-L6-v2     --dataset_type musique     --encoder_dir ./train/musique_kbsize1_19938_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_200_encoder/encoder.pt     --model_dir ./train/musique_kbsize1_19938_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_200     --kb_layer_frequency 1     --dataset_dir ../datasets/musique_dev_100     --test_dataset test_datasets.json     --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_key.npy     --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_value.npy     --query_size 100 --seed -1     --kb_scale_factor 0.125 --kb_size 1 >> debug_eval_consistency_output.log 2>&1 &

````

需要在模型推理时固定以下参数：
- `do_sample=False`
- `top_p=None`
- `temperature=0.0`

使用answer_question_deterministic代替answer_question


## 查看 Attention Weights

输出正确的一个sample：22, id: 2hop__821197_368148

````bash
rm -rf ./attn_weights/* ./attn_heatmaps/*

nohup python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --dataset_type musique \
    --encoder_dir ./train/musique_kbsize10_14000step_qemb/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_4999_encoder/encoder.pt \
    --model_dir ./train/musique_kbsize10_14000step_qemb/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_4999 \
    --kb_layer_frequency 3 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 1 --seed -1 \
    --kb_scale_factor 1 --kb_size 1 --debug_flag >> debug_eval_attention_weights.log 2>&1 &




python ../tools/show_attn_weights.py \
    --attn_dir ./attn_weights/ \
    --keyword "debug" \
    --output_dir ./attn_heatmaps \
    --show average



python ../tools/show_attn_weights.py \
    --attn_dir ./attn_weights/ \
    --keyword "2hop__481349_302087" \
    --output_dir ./attn_heatmaps \
    --show max

# 样本三：2hop__702496_430061
python ../tools/show_attn_weights.py \
    --attn_dir ./attn_weights/ \
    --keyword "2hop__702496_430061" \
    --output_dir ./attn_heatmaps \
    --show average

# 样本三切换不同的kb scale factor
# 0.125, 0.25, 0.5, 1, 2, 4
python ../tools/show_attn_weights.py \
    --attn_dir ./attn_weights/ \
    --keyword "2hop__702496_430061" "14.npy" \
    --output_dir ./attn_heatmaps \
    --show max

````

## 手动修剪三元组

````bash

# 手动修剪样本2hop__702496_430061中的正确三元组，保留关键信息

# 重新创建测试集embedding
python embedding_v2.py \
    --dataset_type musique \
    --model_name all-MiniLM-L6-v2 \
    --dataset_path ../datasets/musique_dev_100/test_datasets.json \
    --batch_size 1024

# 重新执行测试
nohup python eval.py generation --eval_mode kb --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 --encoder_spec all-MiniLM-L6-v2 --dataset_type musique --encoder_dir ./train/musique_kbsize1_19938_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_200_encoder/encoder.pt --model_dir ./train/musique_kbsize1_19938_nooutlier/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-9999KBSize1SepQueryHeadKeyFromkey_all-MiniLM-L6-v2_musique_llama3_step_200 --kb_layer_frequency 1 --dataset_dir ../datasets/musique_dev_100 --test_dataset test_datasets.json --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_key.npy --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_all-MiniLM-L6-v2_embd_value.npy --query_size 1 --seed -1 --kb_scale_factor 1 --kb_size 1 >> debug_eval_attention_weights.log 2>&1 &
````

## 精选三元组

````bash
# 在特定的样本下，手动选择最需要的两条三元组
nohup python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --dataset_type musique \
    --encoder_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_800_encoder/encoder.pt \
    --model_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_800 \
    --kb_layer_frequency 3 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 100 --seed -1 \
    --kb_scale_factor 0.5 --kb_size 1 --debug_flag >> debug_eval_attention_weights.log 2>&1 &


````


## 训练调试


````bash
nohup python train.py   --seed 1 --B 16 --lr 1e-4   --sep_query_head --use_cached_embd --use_lr_decay   --kb_token_layer_frequency 1   --encoder_spec qwen-embedding-0.6B  --key_embd_src key   --dataset_dir ../datasets/musique_train_19938/   --dataset_type musique   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/   --llm_type llama3   --model_save_dir ./train/debug_400   --gradient_accm_step 30   --save_period 100   --total_steps 400   --kb_size 10   --outlier_num -9999   --verbose --debug_level 1 > train.debug_400.log  2>&1 &


python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --dataset_type musique \
    --encoder_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq1UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_2000_encoder/encoder.pt \
    --model_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq1UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_2000 \
    --kb_layer_frequency 1 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 100 --seed -1 \
    --kb_scale_factor 0.5 --kb_size 1

    nohup python eval.py generation \
    --eval_mode kb \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --dataset_type musique \
    --encoder_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq1UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_2500_encoder/encoder.pt \
    --model_dir ./train/debug/stage1_lr_0.0001KBTokenLayerFreq1UseOutlier-9999KBSize10SepQueryHeadKeyFromkey_qwen-embedding-0.6B_musique_llama3_step_2500 \
    --kb_layer_frequency 1 \
    --dataset_dir ../datasets/musique_dev_100 \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path ../datasets/musique_dev_100/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 100 --seed -1 \
    --kb_scale_factor 0.5 --kb_size 1 --debug_flag >> debug_eval_attention_weights.log 2>&1 &

python ../tools/show_attn_weights.py     --attn_dir ./attn_weights/     --keyword "821197"     --output_dir ./attn_heatmaps/     --show max --kb_len 10
````