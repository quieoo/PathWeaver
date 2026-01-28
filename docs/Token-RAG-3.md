
# env setup

````bash
conda activate autoschemakg

pip install -e .
````

# Vector RAG

````bash
cd experiments

python vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100  --similarity-top-k 16

python vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100  --oracle-retrieval

````

# Run through the AutoSchemaKG

## vllm server
````bash

conda activate vllm-13
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/qwen2_7 \
  --served-model-name qwen2_7 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code


export CUDA_VISIBLE_DEVICES=2,3,4,5
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/qwen2.5-14B-Instruct \
  --served-model-name qwen_14 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --trust-remote-code > qwen_14_server.log 2>&1 &


# 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2_7",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "temperature": 0.7
  }'

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen_14",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "temperature": 0.7
  }'
````
## prepare datasets
````bash
python ../atlas_rag/kg_construction/prepare_datasets.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --dataset_type 2wiki \
  --output ../example/example_data/2wiki_dev.json
head -n 5 ../example/example_data/2wiki_dev.json


# 训练集太大（167454样本，1674540个段落）,只取前10k
python ../atlas_rag/kg_construction/prepare_datasets.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_train.json \
  --dataset_type 2wiki \
  --output ../example/example_data/2wiki_train_10k.json \
  --limit 10000
head -n 5 ../example/example_data/2wiki_train_10k.json
````

## generate the KG
````bash
# 清理之前的生成文件
rm -rf ../example/generated/2wiki_dev/*

conda activate autoschemakg

cd scripts
nohup python 1.create_kg_2wiki.py > create_kg_2wiki_train.log 2>&1 &


````

## retrieve and generate response

````bash
conda activate vllm-13
export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/llama3_8B_instruct \
  --served-model-name llama3_8B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code


# 另一个终端
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3_8B",
    "messages": [
      {"role": "user", "content": "你是什么模型"}
    ],
    "temperature": 0.7
  }'


conda activate autoschemakg

# 跑完整的benchmark
cd scripts
python 2.kg_benchmark.py \
  --kg-path ../../example/generated/2wiki_dev \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --dataset-keyword 2wiki_dev.json \
  --llm-model llama3_8B \
  --test-samples 100 \
  --topN 3 \
  --output-path ./graph_rag_bench_2wiki_test100.jsonl


python graph_rag_benchmark.py \
  --kg-path ../../example/generated/2wiki_dev \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --dataset-keyword 2wiki_dev.json \
  --llm-model llama3_8B \
  --test-samples 100 \
  --topN-list 1,2,4,8,16,32,64

python 2.1.atlas_mutihopqa.py
````


# AutoSchemaKG to KBLaM
## Prepare Dataset
Current build KG: 
  - 2wiki_dev
  - 2wiki_train_10k

````bash
# 测试
python 4.kg_to_kblam_dataset.py \
  --encoder-path /home/sdu/zhu/models/qwen-embedding-0.6B \
  --kg-directory ../../example/generated/2wiki_dev \
  --key-word 2wiki_dev.json \
  --question-file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --output-file ../../example/generated/2wiki_dev/kblam_train_datasets.json \
  --max-hops 3 \
  --num-samples 1

# 处理2wiki_dev
export CUDA_VISIBLE_DEVICES=1
nohup python 4.kg_to_kblam_dataset.py \
  --encoder-path /home/sdu/zhu/models/qwen-embedding-0.6B \
  --kg-directory ../../example/generated/2wiki_dev \
  --key-word 2wiki_dev.json \
  --question-file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --output-file ../../example/generated/2wiki_dev/kblam_train_datasets.json > kg_to_kblam_dataset_2wiki_dev.log 2>&1 &

# 处理2wiki_train_10k
export CUDA_VISIBLE_DEVICES=2
nohup python 4.kg_to_kblam_dataset.py \
  --encoder-path /home/sdu/zhu/models/qwen-embedding-0.6B \
  --kg-directory ../../example/generated/2wiki_train_10k \
  --key-word 2wiki_train_10k.json \
  --question-file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_train.json \
  --output-file ../../example/generated/2wiki_train_10k/kblam_train_datasets.json \
  --num-samples 10000 > kg_to_kblam_dataset_2wiki_train_10k.log 2>&1 &
````

创建数据集：合并两个数据集，并切出来100个样本作为测试集

````bash
python 4.1.dataset_split.py
python 4.1.1dataset_format.py
````

## Train

````bash
# 预计算embedding
export CUDA_VISIBLE_DEVICES=1
conda activate kblam_tf457
python tools/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type autoschemakg_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/train_datasets.json \
  --batch_size 1024

python tools/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type autoschemakg_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets.json \
  --batch_size 1024

export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/train_datasets.json --dataset_type autoschemakg_2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 --path_attn \
  --model_save_dir ./train/autoschemakg_2wiki_1.0 \
  >> train_autoschemakg2kblam_2wiki_1.0.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/train_datasets.json --dataset_type autoschemakg_2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/autoschemakg_2wiki_1.1 \
  >> train_autoschemakg2kblam_2wiki_1.1.log 2>&1 &

# ------------------------------------
# 没开path_attn验证精度0.2,开了之后精度反而下降到0.13
# ------------------------------------

# DEBUG
nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/autoschemakg_2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_autoschemakg_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/autoschemakg_2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_autoschemakg_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type autoschemakg_2wiki --query_size 100 --seed 1 --save_dir ./gen_tmp >> eval_autoschemakg2kblam_2wiki_1.1.log 2>&1 &


nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/autoschemakg_2wiki_1.0/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_autoschemakg_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/autoschemakg_2wiki_1.0/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_autoschemakg_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type autoschemakg_2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_autoschemakg2kblam_2wiki_1.0.log 2>&1 &

# ----------- 基于AutoSchemaKG生成的数据集训练模型，噪音太多，导致模型训练效果很差----------------------------------------

# 将模型换成原来训练的，开启path_attn精度0.27
nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/autoschemakg2kblam/2wiki/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type autoschemakg_2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp > eval_autoschemakg2kblam_2wiki_1.0.log 2>&1 &

````


# PathWeaver-AllTriple
## KG Construction with out QA
遍历原始样本，对于每一个段落，提取其中所有实体和三元组。
提取属性三元组和关系三元组。
为关系三元组取一个属性别名。

<!-- 指向示例文件的超链接 -->
[示例文件](../tools/peek.json)

训练集文件：
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_train_2hop_compositional.jsonl
  - Num triples: 4830974
测试集文件：
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl
  - Num triples: 347488
## Naive Retrieve

直接离线处理, 输入all_triples数据集文件，输出一个kblam-qa格式的数据集.
AllTriple数据集格式：
{
  "question":str,
  "answer":str,
  "context": [
    {
      "title": str,
      "triple_list": [
        {
          "type": str,
          "name": str,
          "description_type": str,
          "description": str,
          "key_string": str
        }
      ]
    }
  ]
}
QA数据集格式：
{
  "Q": str,
  "A": str,
  "triple_lists": [
    {
      "name": str,
      “description_type": str,
      "description": str,
      "key_string": str
    },
    {
      "name": str,
      "description_type": str,
      "description": str,
      "key_string": str
    }
  ]
}
处理方法：
<!-- 第一版 -->
- 遍历每一个样本
- 每一个AT样本中包含“supporting_facts”字段，该字段是一个列表（长度为2，表明两跳样本），每一个列表内的第一个字符串是关键context的title。基于这一信息过滤其它不想管的context。
- 对于目标context（应该是两个），提取所有三元组并构建一个子图。三元组是其中的节点，如果三元组T1的description等于三元组T2的name，那么就存在一条边T1->T2。
- 遍历所有2-hop路径，拼接key_string（使用kb_retriever中的build_path_string），计算向量
- 计算和Q的向量相似度，取top-1

<!-- 第二版 -->
不再使用数据集中的key_stirng判断三元组的类型，而是将每个三元组都同时看成属性三元组和关系三元组。
构建一个新的build_path_string方法，接受2个三元组，但是应该返回4个可能的path string。
最后统一经过向量化和距离排序
对于最后的Top-1 Path，根据两个三元组的类别重新生成key_string(属性三元组: "the <description_type> of <name>", 关系三元组: "<name> <description_type>")

````bash

python scripts/AT2QA.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100
# 检索时间：0.0513

python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100


# 创建embedding
cd ../tools
python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional.json \
  --batch_size 1024

# Saved embeddings → /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_key.npy / /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_value.npy

# 在AT2QA生成的数据集上跑Llama3-8B-PW-2wiki
cd ../experiments
python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset AT2QA_2wiki_dev_2hop_compositional.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn 



````

## hard_negs retraining

````bash
# 生成新的训练集：包含一个样本包含最多16个路径，其中包含hard_negative路径，也就是语义很像，但是答案不对的路径
nohup python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_train_2hop_compositional.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --k 16 > at.log 2>&1 &

python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_top16.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100 --k 16

# 合并gold_path
  # 不能按照Q来合并，因为QA数据集中Q也优化过，得按照id合并
python scripts/extract_gold_path.py
# Extract gold path from /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json -> /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional.json, result saved to /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json


# 执行at2qa训练
#-------------------------------v1----------------------------------
# B=1
# 降LR: 5e-4 -> 5e-6
# 总步数：5k(续训练)
# use_cached_embd: 关
# precomputed: 关
# base_embeder_path: /home/sdu/zhu/models/qwen-embedding-0.6B
# test_data_path: 关，使用train数据集的最后100条
# dataset_type: at2qa_2wiki(关键)， at2qa激活正确的get_embeddings， 2wiki激活hop_num=2
# 恢复训练：model_dir_to_resume ./train/2wiki_1.1
#-------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 1  --lr 5e-6 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_1 --save_period 100 \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_at2qa_2wiki_1.log 2>&1 &

#-------------------------------v2----------------------------------
# 在v1的基础上：
# warmup_ratio: 0.1
# 梯度裁剪： grad_clip=1.0
#------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=3
nohup python train.py \
  --seed 1 --B 1  --lr 5e-6 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --warmup_ratio 0.1 --grad_clip 1.0 \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_2 --save_period 100 \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_at2qa_2wiki_2.log 2>&1 &


#-------------------------------v3----------------------------------
# 在v1的基础上恢复学习率
# lr: 5e-4
#------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=4
nohup python train.py \
  --seed 1 --B 1  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_3 --save_period 100 \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_at2qa_2wiki_3.log 2>&1 &

#-------------------------------v4----------------------------------
# 在v3的基础上:
 # random_shuffle
 # 增加一个stage-4: 没有gold_path
#------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=2
nohup python train.py \
  --seed 1 --B 1  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4 --save_period 100 \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_at2qa_2wiki_4.log 2>&1 &

# v1: 5000步，loss 0.25, 精度0.13
#   感觉是学习率太低
# v2: 1689步，loss 0.27
#   同样的问题
# v3: 1849步，loss 0.58
# v4: 2337步，loss 0.04

#--------------------v4---------------------------
# 把v4挪到L40上重新训练
# Stage-1 (2000步): 1 * gold path + 16 * random path
# Stage-2 (2000步): 1 * gold path + 8 * negative path + 8 * random path
# Stage-3 (2000步)：1 * gold path + 16 * negative path 
# Stage-4 (2000步)：16 * negative path 

export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 1  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4 --save_period 100 \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_at2qa_2wiki_4_1.log 2>&1 &


# 解析训练日志
python scripts/parse_log_metrics.py 

#-------------------------------------
# S1: 0.1 -> 0.8
# S2: 0.2 -> 0.8
# S3: 0.8 -> 0.9
# S4: 0.5 -> 0.6
#-------------------------------------


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#这个版本存在问题，只保留了原来数据集中question和QA数据集中的question相同的样本
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

````
## Q-align & hard_negs

Stage-1 (2000步): gold_Q -- 1 * gold path + 16 * random path
Stage-2 (2000步): Q -- 1 * gold path + 16 * random path
Stage-3 (2000步): Q -- 1 * gold path + 8 * negative path + 8 * random path
Stage-4 (2000步)：Q -- 1 * gold path + 16 * negative path 
Stage-5 (2000步)：Q -- 16 * negative path 

````bash
export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 1  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 10000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.2 --save_period 100 \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_at2qa_2wiki_4_2.log 2>&1 &


# 增大batch
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 10000 --N 9999999 \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --model_save_dir ./train/at2qa_2wiki_4.3 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_3.log 2>&1 &
````

### v4.3过程中stage-0的精度问题

stage-1按理来说gold_Q + gold path应该是很稳的，但是训练过程中（前2000步）精度先上升（0.5）再下降（0），最后才恢复到0.3不到。

验证是否是数据集问题：
````bash
# QA数据集 + QA_2wiki_8000step 模型
export CUDA_VISIBLE_DEVICES=1
nohup python eval_generation.py generation \
    --kb_size=16 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 1 --path_attn >> at2qa_stage0_debug.log 2>&1 &

  #--  精度没问题： 0.86

# AT2QA数据集 + QA_2wiki_8000step 模型，但是stage-0

# 修改current step为0
nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --query_size 1 --path_attn \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --step 0 --t_step 1 \
    --dataset_type at2qa_2wiki >> at2qa_stage0_debug.log 2>&1 &

#-- 精度是崩溃的：0.05

# Test3: 使用at2qa的数据集处理方法加载at数据集
nohup python eval_generation.py generation \
    --kb_size=16 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --dataset_type 2wiki --query_size 1 --path_attn >> at2qa_stage0_debug.log 2>&1 &

# --找不到原因，反正就是base_embeder出来就是不一样，但是不知道什么原因（已经尽可能对齐offline的计算方法）

# Test4：使用offline的方法处理at2qa数据集
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --batch_size 1024

# ✅ Saved embeddings → /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy / /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy

nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --query_size 1 --path_attn \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --step 0 --t_step 1 \
    --dataset_type at2qa_2wiki >> at2qa_stage0_debug.log 2>&1 &

# 离线处理是有用的，精度恢复正常

````
### embedding对齐问题

````bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 10000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.5 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_5.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 20000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.6 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_6.log 2>&1 &
````

#=========
重训练行不通，精度完全崩溃，判断肯定还是数据集的问题
#=========
也做过最小测试，只要输入是相同的，那么embedding后的向量肯定也是相同的，不存在离线和在线偏移的问题


### AT2QA 数据集离线处理

````bash
# 测试集
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --batch_size 1024

# 训练集
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --batch_size 1024

export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 1  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.7 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_7.log  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 4000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.8 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_8.log  2>&1 &

````

## 两阶段 T1： Q-align; T2: hard_negs
````bash
# 使用get_question_type_sampled_T1，get_triple_ids_T1
# B=5
# steps: 4000 (3.5h)
export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 4000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.9.1 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_9_1.log  2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --model_dir_to_resume ./train/at2qa_2wiki_4.9.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_3999 \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.9.2 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_9_2.log  2>&1 &
````

=============结论================
T1阶段很顺利，精度很快上升到0.8并保持问题
T2阶段面临问题：精度随着gold path的比例降低而降低，最后只有0.2

加一组4_9_2的实验，换成evaluate一直只用negative path，观察精度是否变化
````bash
export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \`
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --model_dir_to_resume ./train/at2qa_2wiki_4.9.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_3999 \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_4.9.2.1 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_9_2.1.log  2>&1 &

````


## AT-From-Base
不在QA数据集上续训练（已经形成了对gold path的依赖），直接从base模型上开始训练，随着训练可以逐渐增加negative path的数量，但是全程使用question和negative path




## testing

### top-1
生成数据集：top-1 (暂时只有top-1)
````bash
python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100

# AT2QA默认生成path列表，完全转变成2wiki数据集格式需要提出path列表中的第一个path中的triple
python scripts/path_extract.py


python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional.json \
  --batch_size 1024


````

````bash
# 测试v4(2300-step)): 0.26
export CUDA_VISIBLE_DEVICES=2
python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/at2qa_2wiki_4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_2300_encoder/encoder.pt \
    --model_dir ./train/at2qa_2wiki_4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_2300 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset AT2QA_2wiki_dev_2hop_compositional.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn 

# 测试v4(4200-step)): 0.26
python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/at2qa_2wiki_4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_4200_encoder/encoder.pt \
    --model_dir ./train/at2qa_2wiki_4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_4200 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset AT2QA_2wiki_dev_2hop_compositional.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn

````

### top-16

````bash
python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_top16.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100 --k 16

python scripts/extract_gold_path.py

cd ../experiments
python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/at2qa_2wiki_4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/at2qa_2wiki_4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset AT2QA_2wiki_test_2hop_compositional_gold.json \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --dataset_type at2qa_2wiki --query_size 100 --seed 1 --path_attn

````