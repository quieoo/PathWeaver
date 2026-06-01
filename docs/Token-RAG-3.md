# env setup

```bash
conda activate autoschemakg

pip install -e .
```

# Vector RAG

```bash
cd experiments

python vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100  --similarity-top-k 16

python vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100  --oracle-retrieval

```

# Run through the AutoSchemaKG

## vllm server

```bash

conda activate vllm-13
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/qwen2_7 \
  --served-model-name qwen2_7 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code

export CUDA_VISIBLE_DEVICES=0,1
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/qwen2.5-14B-Instruct \
  --served-model-name qwen_14 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --trust-remote-code > qwen_14_server.log 2>&1 &


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
```

## prepare datasets

```bash
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
```

## generate the KG

```bash
# 清理之前的生成文件
rm -rf ../example/generated/2wiki_dev/*

conda activate autoschemakg

cd scripts
nohup python 1.create_kg_2wiki.py > create_kg_2wiki_train.log 2>&1 &


```

## retrieve and generate response

```bash
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
```

## Hotpot

```bash
# 0.prepare data
conda activate autoschemakg
cd AutoSchemaKG/docs

python ../atlas_rag/kg_construction/prepare_datasets.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --dataset_type hotpot \
  --output ../example/example_data/hotpot_dev.json
head -n 5 ../example/example_data/hotpot_dev.json

# 修改DATA_DIRECTORY和DATA_NAME
cd scripts
nohup python 1.create_kg_2wiki.py > create_kg_2wiki_train.log 2>&1 &

```

## Musique

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/qwen2.5-14B-Instruct \
  --served-model-name qwen_14 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --trust-remote-code > qwen_14_server.log 2>&1 &

conda activate autoschemakg
cd AutoSchemaKG/docs

python ../atlas_rag/kg_construction/prepare_datasets.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_dev.jsonl \
  --dataset_type musique \
  --output ../example/example_data/musique_dev.json
head -n 5 ../example/example_data/musique_dev.json

# 修改DATA_DIRECTORY和DATA_NAME
cd scripts
nohup python 1.create_kg_2wiki.py > create_kg_musique_dev.log 2>&1 &
```

# AutoSchemaKG to KBLaM

## Prepare Dataset

Current build KG:

- 2wiki_dev
- 2wiki_train_10k

```bash
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
```

创建数据集：合并两个数据集，并切出来100个样本作为测试集

```bash
python 4.1.dataset_split.py
python 4.1.1dataset_format.py
```

## Train

```bash
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

```

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

```bash

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



```

## hard_negs retraining

```bash
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

```

## Q-align & hard_negs

Stage-1 (2000步): gold*Q -- 1 * gold path + 16 _ random path
Stage-2 (2000步): Q -- 1 _ gold path + 16 _ random path
Stage-3 (2000步): Q -- 1 _ gold path + 8 _ negative path + 8 _ random path
Stage-4 (2000步)：Q -- 1 _ gold path + 16 _ negative path
Stage-5 (2000步)：Q -- 16 \_ negative path

```bash
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
```

### v4.3过程中stage-0的精度问题

stage-1按理来说gold_Q + gold path应该是很稳的，但是训练过程中（前2000步）精度先上升（0.5）再下降（0），最后才恢复到0.3不到。

验证是否是数据集问题：

```bash
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

```

### embedding对齐问题

```bash
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
```

#=========
重训练行不通，精度完全崩溃，判断肯定还是数据集的问题
#=========
也做过最小测试，只要输入是相同的，那么embedding后的向量肯定也是相同的，不存在离线和在线偏移的问题

### AT2QA 数据集离线处理

```bash
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

```

## 两阶段 T1： Q-align; T2: hard_negs

```bash
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
```

=============结论================
T1阶段很顺利，精度很快上升到0.8并保持问题
T2阶段面临问题：精度随着gold path的比例降低而降低，最后只有0.2

加一组4_9_2的实验，换成evaluate一直只用negative path，观察精度是否变化

```bash
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
  --model_save_dir ./train/at2qa_2wiki_4.9.2.1 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_4_9_2.1.log  2>&1 &

```

## AT-From-Base

不在QA数据集上续训练（已经形成了对gold path的依赖），直接从base模型上开始训练，随着训练可以逐渐增加negative path的数量，但是全程使用question和negative path

```bash
# get_embeddings_at2qa_from_precompute_batch ——》 get_triple_ids_ATFB
# get_question_type_sampled_T2

# 四个阶段提高negative path的数量
# 1. 1
# 2. 4
# 3. 8
# 4. 16

export CUDA_VISIBLE_DEVICES=0
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
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_5 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_5.log  2>&1 &


```

### 提取silver path

silver path是指negative path中指向正确答案的路径
确保在每个阶段中都有silver path

- extract_silver_path.py, 把数据集中的silver path移动到第一位
- 在kb_retriever.py中，从头开始读入negative path之后通过random.shuffle(path_triple_base)打乱顺序

提取silver path

```bash
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/extract_siver_path.py

# 在脚本里直接改文件名
```

```
Extract silver path from AT2QA_2wiki_test_2hop_compositional_gold.json to ATFB_2wiki_test_2hop_compositional_silver.json
Total sample: 100
Exact match ratio: 0.44
Contain ratio: 0.8

Extract silver path from AT2QA_2wiki_train_2hop_compositional_gold.json to ATFB_2wiki_train_2hop_compositional_silver.json
Total sample: 74988
Exact match ratio: 0.5989091587987412
Contain ratio: 0.8886488504827439
```

重新计算embedding

```bash
export CUDA_VISIBLE_DEVICES=2
# 测试集
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --batch_size 1024

# 训练集
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --batch_size 1024

export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --disable_random_sample \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_5.1.1 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_5.1.1.log  2>&1 &

# 1*silver + (n-1)*random negative
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_5.1.2 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_5.1.2.log  2>&1 &


```

### 训练坍塌问题

```
[15:17:24] INFO     step: 3654 , loss: 0.013303414644906298        train.py:1348
3655-8000-73100: [56108 56109 56110 56111 56112 56113 56114 56115 56116 56117]
3655-8000-73101: [56118 56119 56120 56121 56122 56123 56124 56125 56126 56127]
3655-8000-73102: [56128 56129 56130 56131 56132 56133 56134 56135 56136 56137]
3655-8000-73103: [56138 56139 56140 56141 56142 56143 56144 56145 56146 56147]
3655-8000-73104: [56148 56149 56150 56151 56152 56153 56154 56155 56156 56157]
3655-8000-73105: [56158 56159 56160 56161 56162 56163 56164 56165 56166 56167]
3655-8000-73106: [56168 56169 56170 56171 56172 56173 56174 56175 56176 56177]
3655-8000-73107: [56178 56179 56180 56181 56182 56183 56184 56185 56186 56187]
3655-8000-73108: [56188 56189 56190 56191 56192 56193 56194 56195 56196 56197]
3655-8000-73109: [56198 56199 56200 56201 56202 56203 56204 56205 56206 56207]
3655-8000-73110: [56208 56209 56210 56211 56212 56213 56214 56215 56216 56217]
3655-8000-73111: [56218 56219 56220 56221 56222 56223 56224 56225 56226 56227]
3655-8000-73112: [56228 56229 56230 56231 56232 56233 56234 56235 56236 56237]
3655-8000-73113: [56238 56239 56240 56241 56242 56243 56244 56245 56246 56247]
3655-8000-73114: [56248 56249 56250 56251 56252 56253 56254 56255 56256 56257]
3655-8000-73115: [56258 56259 56260 56261 56262 56263 56264 56265 56266 56267]
3655-8000-73116: [56268 56269 56270 56271 56272 56273 56274 56275 56276 56277]
3655-8000-73117: [56278 56279 56280 56281 56282 56283 56284 56285 56286 56287]
3655-8000-73118: [56288 56289 56290 56291 56292 56293 56294 56295 56296 56297]
3655-8000-73119: [56298 56299 56300 56301 56302 56303 56304 56305 56306 56307]
[15:17:27] INFO     step: 3655 , loss: 0.021708486590068788        train.py:1348
3656-8000-73120: [56308 56309 56310 56311 56312 56313 56314 56315 56316 56317]
3656-8000-73121: [56318 56319 56320 56321 56322 56323 56324 56325 56326 56327]
3656-8000-73122: [56328 56329 56330 56331 56332 56333 56334 56335 56336 56337]
3656-8000-73123: [56338 56339 56340 56341 56342 56343 56344 56345 56346 56347]
3656-8000-73124: [56348 56349 56350 56351 56352 56353 56354 56355 56356 56357]
3656-8000-73125: [56358 56359 56360 56361 56362 56363 56364 56365 56366 56367]
3656-8000-73126: [56368 56369 56370 56371 56372 56373 56374 56375 56376 56377]
3656-8000-73127: [56378 56379 56380 56381 56382 56383 56384 56385 56386 56387]
3656-8000-73128: [56388 56389 56390 56391 56392 56393 56394 56395 56396 56397]
3656-8000-73129: [56398 56399 56400 56401 56402 56403 56404 56405 56406 56407]
3656-8000-73130: [56408 56409 56410 56411 56412 56413 56414 56415 56416 56417]
3656-8000-73131: [56418 56419 56420 56421 56422 56423 56424 56425 56426 56427]
3656-8000-73132: [56428 56429 56430 56431 56432 56433 56434 56435 56436 56437]
3656-8000-73133: [56438 56439 56440 56441 56442 56443 56444 56445 56446 56447]
3656-8000-73134: [56448 56449 56450 56451 56452 56453 56454 56455 56456 56457]
3656-8000-73135: [56458 56459 56460 56461 56462 56463 56464 56465 56466 56467]
3656-8000-73136: [56468 56469 56470 56471 56472 56473 56474 56475 56476 56477]
3656-8000-73137: [56478 56479 56480 56481 56482 56483 56484 56485 56486 56487]
3656-8000-73138: [56488 56489 56490 56491 56492 56493 56494 56495 56496 56497]
3656-8000-73139: [56498 56499 56500 56501 56502 56503 56504 56505 56506 56507]
[15:17:30] INFO     step: 3656 , loss: 2.1545043233782053          train.py:1348
```

目前来看似乎是数据集切换到silver之后出的问题。

- 数据集变化
- batch size 5 -> 10

### B=5

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/at2qa_2wiki_5.1.2 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_at2qa_2wiki_5.1.2.log  2>&1 &

python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/at2qa_2wiki_5.1.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/at2qa_2wiki_5.1.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset ATFB_2wiki_test_2hop_compositional_silver.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type at2qa_2wiki --query_size 100 --seed 1 --path_attn

# 1 + 1 -> 1 + 15
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_v2 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_v2.log  2>&1 &


python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/atfb_2wiki_v2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/atfb_2wiki_v2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_at2qa_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
    --test_dataset ATFB_2wiki_test_2hop_compositional_silver.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type at2qa_2wiki --query_size 100 --seed 1 --path_attn
```

### keep_top_k_ckpt=10

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_v3 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_v3.log  2>&1 &


# 跑一组 2-4-8-16作对比
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
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_v3.1 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_v3.1.log  2>&1 &


  # 跑一组 2-4作对比
  # 使用w/o siliver数据集作为验证输入
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
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_v3.2 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_v3.2.log  2>&1 &
```

#### top-k ckpt的测试

```bash

# 批量测试

nohup python scripts/batch_inference.py > batch_inference.log 2>&1 &

```

### silver+random; 2,4 path

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_v4 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_v4.log  2>&1 &

```

### retrieve; 2-4-8 path

```bash
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
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_v5 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_v5.log  2>&1 &

export DASHSCOPE_API_KEY=sk-xxxx
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/batch_inference.py > batch_inference_v5.log 2>&1 &
```

### bge-embedding

```bash
export CUDA_VISIBLE_DEVICES=1
conda activate kblam_tf457

python scripts/embedding_v2.py \
  --model_name bge \
  --local_model_path /home/sdu/zhu/models/bge-en-v1.5/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --batch_size 1024

python scripts/embedding_v2.py \
  --model_name bge \
  --local_model_path /home/sdu/zhu/models/bge-en-v1.5/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --batch_size 1024

cd experiments
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec bge --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_bge-en-v1.5_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_bge-en-v1.5_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_bge-en-v1.5_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_bge-en-v1.5_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_bge_v1 --keep_top_k_ckpt 10 --save_period 100 \
  > train_atfb_2wiki_bge_v1.log  2>&1 &


```

NOTE：不确定是过程出的错还是bge-embedidng本身在这类任务上性能更差，导致训练出来的模型精度降低大约10%.

### 复现qwen-embedding

同时做一个小修改:

- 训练过程中的验证集使用和推理时一模一样的配置：2条、无silver

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_qwen_embedding_v1 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_qwen_embedding_v1.log  2>&1 &

```

- 训练路径：8或16条，1 silver + n random
- 验证路径：和训练一样

```bash

# 4阶段
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
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 12000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_qwen_embedding_v3 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_qwen_embedding_v3.log  2>&1 &


# 一组一模一样配置的
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 12000 --N 9999999 \
  --model_save_dir ./train/atfb_2wiki_qwen_embedding_v3.1 --keep_top_k_ckpt 10 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  > train_atfb_2wiki_qwen_embedding_v3.1.log  2>&1 &
```

## 修复训练过程中不稳定的崩溃问题

在训练PathWeaver模型过程中出现精度崩溃的现象，在训练一定步数后loss突然大增，之后的训练虽然可以使得loss继续降低，但是模型的推理精度会受到不可恢复的影响。
这种现象并不总是会发生，但是发现随着训练的批量大小增加发生概率会变大。

### 问题复现

跑两组，不同的B值

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/debug_spikes_v1 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_debug_spikes_v1.log  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 20  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/debug_spikes_v2 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_debug_spikes_v2.log  2>&1 &

```

### 解决1：训练集过滤

filter_valid_sample.py: 将路径为空，或者不包含正确答案的样本过滤掉

```
Original samples: 74988
Valid samples: 66752
Saved valid samples to ATFB_2wiki_train_2hop_compositional_silver_filtered.json
```

重新生成embedding:

```bash
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered.json \
  --batch_size 1024
```

重新训练：

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/debug_spikes_v3 --save_period 100 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_debug_spikes_v3.log  2>&1 &

```

效果：似乎稳定了很多，Training loss不再波动

进一步测试：

- 修改路径选择，训练时候：silver+random -> silver+sequencial, 验证时候：2条逐渐增加到8条 -> 固定2条(固定current_step=0)
- keep_top_k_ckpt=10

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/debug_spikes_v4 --save_period 100 --keep_top_k_ckpt 10 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_debug_spikes_v4.log  2>&1 &

```

### 解决2： 降低batch size

```bash
export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
  --seed 1 --B 2  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/debug_spikes_v5 --save_period 100 --keep_top_k_ckpt 10 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_debug_spikes_v5.log  2>&1 &

```

结论：减少batch size仍旧会出现崩溃现象

### 解决3：不要分阶段，直接固定路径数量

修改kb_retriever.py中的get_triple_ids_ATFB：

```python
    step_ratio=[1.0]
    stage_num=len(step_ratio)
    negative_path_num=[max_kb_paths]
```

```bash
export CUDA_VISIBLE_DEVICES=2
nohup python train.py \
  --seed 1 --B 2  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type at2qa_2wiki \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir ./train/debug_spikes_v6 --save_period 100 --keep_top_k_ckpt 10 \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
  >> train_debug_spikes_v6.log  2>&1 &
```

结论：这不行，模型完全不收敛

## Hotpot

一些新的挑战：

- Hotpot上答案可能出现在head、relation位置，而不是一定在tail
- 答案和上下文可能存在相同意思，但是拼写不同的情况 （数据集过滤时候需要注意）

### 原版AT2QA_v2

```bash
export CUDA_VISIBLE_DEVICES=2
nohup python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_hotpot_paths_train.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000 \
  --k 16 >> at_hotpot.log 2>&1 &

nohup python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_hotpot_paths_dev.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100 \
  --k 16 >> at_hotpot.log 2>&1 &
```

输出：

```
100条
Average pick time: 0.1253s
Graph recall: 0.4900
Answer recall: 0.4500

1000条
Average pick time: 0.1141s
Graph recall: 0.5470
Answer recall: 0.4930
```

### 原版 + bge-embedding

```bash
export CUDA_VISIBLE_DEVICES=2
nohup python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_hotpot_paths_train.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 256 \
  --keep_score \
  --limit 100 \
  --k 16 >> at_hotpot.log 2>&1 &

nohup python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_hotpot_paths_dev.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 256 \
  --keep_score \
  --limit 100 \
  --k 16 >> at_hotpot.log 2>&1 &
```

输出：

```
Average pick time: 0.0693s
Graph recall: 0.4900
Answer recall: 0.4200
```

结论：bge更快，但是性能比qwen-embedding-0.6B略微慢点

### 开启verbose debug

```bash
export CUDA_VISIBLE_DEVICES=2
nohup python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_hotpot_paths_train_debug.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100 \
  --k 16 > at_hotpot.log 2>&1 &
```

## testing

### top-1

生成数据集：top-1 (暂时只有top-1)

```bash
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


```

```bash
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

```

### top-16

```bash
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

```

# DAG-KV

DAG-KV问题描述：

- 有一张知识图谱，其中包含若干实体，实体与实体之间拥有关系，从而产生关系三元组，每个实体又有属性从而产生属性三元组。实体和属性值是节点，关系和属性类型是边。
- 每个三元组可以表达成两个键值对（关系和属性类型作为双向边）
- 对于一个问题首先判断它与哪些实体有关，这些实体之间必须是有关系的。从实体图中找到一个最小子图。目标是限制搜索范围，避免包含噪音三元组（比如不同实体但是拥有相似的属性）
- 在最小实体子图中继续筛选和问题有关的属性，获得一个最小实体属性图。最小实体属性图应该只包含单向边，不应该拥有环。
- 将最小实体属性图中的每一条边转换为键值对，保存邻接矩阵代表键值对之间的关系。

## 知识图谱提取(LLM-based-triple-extractor)

### 提取规则

1. 实体抽取
   - 抽取所有显式出现的 named entities（人/组织/地点/作品/事件等）
   - 日期/数值/职业/国籍/描述短语不当作实体节点

2. 关系抽取 (entity-entity)
   - 仅抽取文本中明确陈述的实体-实体关系
   - 关系名为短动词/介词短语，不包含数值/日期

3. 属性抽取 (entity-attribute)
   - 抽取实体的显式属性：时间、地点、数量、身份、类别、头衔、描述等
   - 属性名短且一致

4. KV 生成规则
   - 对每个属性事实(e,a,v)输出两对KV:
     - key=natural_forward_string(e,a), value=v
     - key=natural_reverse_string(v,a), value=e
   - 对每个关系事实(e1,r,e2)输出两对KV:
     - key=natural_forward_string(e1,r), value=e2
     - key=natural_reverse_string(e2,r), value=e1
   - natural_reverse_string必须是对同一事实在反向视角下的等价描述，不能引入新条件
   - 例如：
     - 例如，属性事实(India,location,South Asia)，输出 ('the location of India', 'South Asia'), ('South Asia is the location of', 'India')
     - 例如，关系事实(Bill Gates, founded, Microsoft), 输出 ('Bill Gates founded', 'Microsoft'), ('Microsoft is founded by', 'Bill Gates')

5. 关系派生属性
   - 对每个关系事实(e1,r,e2)派生出一个等价的属性事实 (e1,a,e2), a是派生属性名，再额外生成两对KV:
     - key=natural_forward_string(e1,a), value=e2
     - key=natural_reverse_string(e2,a), value=e1
     - 例如：
       - 关系事实 (Albert Einstein, was born in, Ulm), 派生出属性名 'birthplace'，因此额外输出两个派生属性KV('the birthplace of Albert Einstein', 'Ulm'), ('the person born in Ulm', 'Albert Einstein')

### Prompt模板

```python
Prompt_prefix = (
  """
  You are a precise knowledge extraction system designed to build a clean bidirectional KV-based knowledge graph.

Your task is to extract factual knowledge from ONE given context and output structured key/value pairs.

============================================================
I. ENTITY EXTRACTION
============================================================

1. Extract ALL explicitly mentioned named entities:
   - Persons
   - Organizations
   - Locations
   - Works (books, films, songs, etc.)
   - Events
   - Clearly named real-world entities

2. The following MUST NOT be treated as entities:
   - Dates
   - Numbers
   - Quantities
   - Job titles
   - Nationalities
   - Descriptive phrases
   - Categories

If unsure, treat it as a VALUE, not an entity.

============================================================
II. RELATION EXTRACTION (entity → entity)
============================================================

1. Extract only explicitly stated entity-to-entity relations.
2. Do NOT infer or create multi-hop relations.
3. Relation names must:
   - Be short verb phrases or prepositional phrases
   - NOT contain dates or numbers
   - Reflect the wording of the text

Example:
(Bill Gates, founded, Microsoft)

============================================================
III. ATTRIBUTE EXTRACTION (entity → value)
============================================================

Extract explicit attributes of entities, including:
- Time
- Location
- Quantity
- Identity
- Category
- Title
- Description

Attribute names must:
- Be short
- Be consistent
- Not include value information inside the attribute name

Example:
(Albert Einstein, date of birth, 14 March 1879)

============================================================
IV. KV GENERATION RULES (BIDIRECTIONAL)
============================================================

For EACH extracted fact, generate TWO key/value pairs.

A. For attribute fact (e, a, v):

1) Forward KV:
   key = natural_forward_string(e, a)
   value = v

2) Reverse KV:
   key = natural_reverse_string(v, a)
   value = e

B. For relation fact (e1, r, e2):

1) Forward KV:
   key = natural_forward_string(e1, r)
   value = e2

2) Reverse KV:
   key = natural_reverse_string(e2, r)
   value = e1

------------------------------------------------------------
Reverse Equivalence Constraint (CRITICAL)
------------------------------------------------------------

natural_reverse_string MUST:
- Be a logically equivalent restatement of the SAME fact
- Only reverse perspective
- NOT introduce new assumptions
- NOT introduce new type labels
- Use neutral wording if needed

If strict equivalence cannot be achieved, use a safe neutral reverse phrasing.

------------------------------------------------------------
Examples
------------------------------------------------------------

Attribute fact:
(India, location, South Asia)

Output:
('the location of India', 'South Asia')
('South Asia is the location of', 'India')

Relation fact:
(Bill Gates, founded, Microsoft)

Output:
('Bill Gates founded', 'Microsoft')
('Microsoft is founded by', 'Bill Gates')

============================================================
V. RELATION-DERIVED ATTRIBUTE
============================================================

For each relation (e1, r, e2), derive ONE equivalent attribute fact (e1, a, e2)

Rules:
- The derived attribute must not change meaning.
- It must not introduce new type information.
- It must be interchangeable with the relation in question-answer form.

For each valid derived attribute, generate TWO additional KV pairs:

1) Forward:
   key = natural_forward_string(e1, a)
   value = e2

2) Reverse:
   key = natural_reverse_string(e2, a)
   value = e1

------------------------------------------------------------
Example (SAFE)
------------------------------------------------------------

Relation:
(Albert Einstein, was born in, Ulm)

Derived attribute:
(Albert Einstein, birthplace, Ulm)

Additional KV:
('the birthplace of Albert Einstein', 'Ulm')
('the person born in Ulm', 'Albert Einstein')

============================================================
VI. OUTPUT FORMAT (STRICT)
============================================================

Output plain text only.
Each KV pair must be written on one line:

key | value

Do NOT output JSON.
Do NOT output explanations.
Do NOT infer missing facts.
Do NOT merge contexts.
Preserve surface names exactly as written.

============================================================
Convert the following context:
  """
)
```

## 图谱检索

```bash
export CUDA_VISIBLE_DEVICES=2
nohup python scripts/DAG_KV_Retriever_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 3 --max_sinks 3 --use_seededge_beam \
  --verbose  > dag_hotpot.log 2>&1 &

nohup python scripts/DAG_KV_Retriever_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 3 --max_sinks 3  --use_seededge_beam > dag_hotpot.log 2>&1 &


```

Answer recall: 0.269
增加entity normalization： 0.278

增加字符级的相似度作为得分计算

```bash
nohup python scripts/DAG_KV_Retriever_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 3 --max_sinks 3 --use_seededge_beam \
  --verbose  >> dag_hotpot.log 2>&1 &

```

直接清理sink

```bash

nohup python scripts/DAG_KV_Retriever_v3.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 10 --use_seededge_beam --pred_weight 1 \
  --sink_prune_qrel \
  --promote_topk_nodes 32 \
  --promote_margin 0.015 \
  --promote_drop_th 0.18 \
  --protect_high_out 0.45 >> dag_hotpot.log 2>&1 &
```

回到原版

```bash
nohup python scripts/DAG_KV_Retriever_v3.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 10 --use_seededge_beam --pred_weight 1 \
  --keep_one_attr_direction \
  --verbose >> dag_hotpot.log 2>&1 &

```

### Related work-1: PropRAG

### Related work-2: HippoRAG

```bash
export CUDA_VISIBLE_DEVICES=2
nohup python scripts/DAG_KV_HippoRAG.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 3 --max_sinks 3 \
  --score_mode ppr --verbose > dag_hotpot_hipporag.log 2>&1 &

nohup python scripts/DAG_KV_HippoRAG.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000  > dag_hotpot_hipporag.log 2>&1 &

nohup python scripts/DAG_KV_HippoRAG.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000  \
    --max_hops 4 \
  --beam_width 6 \
  --max_nodes 80 \
  --max_edges 140 \
  --max_sinks 12 \
  --seed_top_m 60 \
  --hippo_topk_evidence 200 \
  --hippo_iters 50 \
  --hippo_mix_beta 0.35  > dag_hotpot_hipporag.log 2>&1 &
```

### Related work-3: Multi-hop Dense Retrieval

## DAG-KV 检索相关工作

问题建模：
输入：
一个由实体节点和属性/实体值节点组成的有向候选图，边来自
head + rel -> tail 和 rel + tail -> head。

目标：
找到一个子图满足：

- 尽量小（少边、少噪声）；
- 覆盖问题中的锚点与约束；
- 连通且无环；
- 内部节点更像“中间实体/桥接实体”，叶子更像“属性值/答案候选”；
- 不用答案监督建图，但答案在评测时更容易出现在 sink。

这本质上就是一个带方向偏置与叶子偏置的 Prize-Collecting Steiner Arborescence 问题。

Prompt：
我现在已经提取出来一些三元组，并构建一个实体-属性图，其中实体和属性都是图上的节点，图上的边基于三元组生成“head+rel -> tail”和"rel + tail -> head"，我现在需要在这个图上生成一个最小DAG图，同时保证答案尽可能出现在DAG图的末梢（我虽然有答案但是只能用它来检查结果是否正确，不能用来建图）。我的想法是这样的：先找一些SOTA的这领域相关工作，把他们直接用到我的这个问题上，然后比较一下效果，如果有效果好的，我就直接套用，如果效果不好的话我再自己动手实现一个新的方法。这样子这个过程以及结果放在我的论文里头会更加具有说服力。目前我已经确定了5个相关工作：IRCoT、HippoRAG、G-Retriever、UNIQORN、SubgraphRAG。接下来给我一版基于IRCoT的新代码，和现有的DAG_KV_Retriever_v3.py保持一致的输入、输出格式，统计graph recall、answer recall和none-sink recall。其它代码可以随便改，目标是尽可能符合IRCoT的思想，如果需要的话我可以去下载找来相关的论文或者开源代码给你参考。

### 现有方法 IRCoT

IRCoT-style iterative graph retrieval
一次性检索改为 IRCoT-style 迭代检索

- 第 0 步先用 question 做 seed retrieval。
- 后续每一步根据当前 frontier、已选 relation、已覆盖 clue，构造一个 structured thought，再用这个 thought 去重排候选边。
- 本质上对应 IRCoT 的 “retrieve → reason → retrieve → reason”。

把 CoT 句子替换成结构化 thought state

- 因为你这里的知识源不是自然语言段落，而是 KV-edge 图。
- 所以我没有强行引入一个 LLM 去生成自然语言 CoT，而是用frontier entities + recent relations + unresolved question tokens组成下一轮检索查询。这样更贴近你的任务，也更容易做 ablation。

显式偏向“答案落在末梢”

- 在 iterative ranking 里保留了对 attribute/value-like edge 的 leaf_bonus。
- 最后继续做 DAG 去环和 max_sinks 约束，尽量把终点压缩成少量 sinks。

和原始 IRCoT 不同的地方：没有真的调用 LLM 生成自然语言 CoT；现在是 deterministic 的 graph-native thought。

```bash
python scripts/graph_gen/DAG_KV_IRCoT.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5 \
  --supporting_only \
  --seed_top_m 6 \
  --max_steps 4 \
  --edges_per_step 4 \
  --beam_width 2 \
  --max_nodes 32 \
  --max_edges 44 \
  --max_sinks 2 \
  --leaf_bonus 0.08 \
  --limit 100 \
  --keep_score

# graph_recall	path_recall	answer_recall
# 0.96	0.51	0.27

cd scripts
nohup ./run_sweep_ir_cot.sh >> dag_hotpot_ircot.log 2>&1 &
cd ..

```

结果：

- answer recall: 0.24 ~ 0.4

### 现有方法 HippoRAG

在 DAG_KV_HippoRAG.py 中，我们将原始 HippoRAG/HippoRAG2 面向文本证据的检索流程适配到结构化三元组图场景：首先将问题与图中实体节点进行语义匹配与显式提及匹配，选取得分最高的实体作为 query seeds；随后不再采用局部 beam 搜索，而是按照 HippoRAG 的核心思想，在图上执行以这些 seeds 为个性化重启分布的 Personalized PageRank（PPR）传播，从而获得与问题相关的全局实体重要性分数。对于 HippoRAG 版本，PPR 直接在实体图上进行；对于 HippoRAG2 版本，则进一步将每条 KV 边视为一个 memory node，构建“实体—记忆”二部图，在实体节点与 KV 证据节点之间联合传播，使传播结果能够同时刻画实体相关性与证据相关性。最后，根据实体 PPR 分数及 memory node 分数对 KV 边进行排序，并在保留连通性的前提下截取得到规模受限的子图，再通过 DAG 化与 sink 数量约束，使答案更倾向于落在子图末梢。这样，HippoRAG/HippoRAG2 的“query concept linking—graph propagation—evidence readout”框架就被自然迁移到了本文的实体—属性 DAG 检索任务中。

```bash

python scripts/graph_gen/DAG_KV_HippoRAG.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --variant hipporag2 \
  --supporting_only \
  --seed_top_m 8 \
  --ppr_alpha 0.15 \
  --ppr_iters 50 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --leaf_bonus 0.08 \
  --limit 100 \
  --keep_score

cd scripts
nohup ./run_sweep_hipporag.sh >> dag_hotpot_hipporag.log 2>&1 &
cd ..

```

- HippoRAG 0.32~0.44
- HippoRAG2 0.33~0.45

### 现有方法-1 G-Retriever：PCST 子图检索

这版是**“G-Retriever 风格 / PCST-like”**，不是逐行复刻官方实现。原因是你现在这个任务不是论文里的“文本图 + LLM 生成”，而是“已有 DAG-KV 候选图上做最小 DAG 检索”；另外当前环境里没有 pcst_fast 这类专用 PCST 求解器，所以我采用的是：

问题感知节点 prize
用 node-question 相似度、问题显式 mention、最佳 incident edge 分数、桥接度来构造节点奖励。

PCST-like 终端选择
从 anchor 节点出发，按 prize - connect_cost 贪心扩展 terminal。

加权 Steiner tree 连通
用 networkx 的加权 Steiner tree 把这些 terminal 连成一个小骨架。

从 root 向外定向成 DAG
再用 leaf bonus 把 value-like 节点更倾向放到末梢，尽量提高 sink answer recall。

```bash
pip install pcst_fast
python scripts/graph_gen/DAG_KV_GRetriever_PCST_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 256 \
  --topk_node_prize 6 \
  --topk_edge_prize 12 \
  --pcst_cost_e 0.5 \
  --max_nodes 30 \
  --max_edges 40 \
  --limit 100

cd scripts
nohup ./batch_run_pcst_v2.sh >> dag_hotpot_pcst.log 2>&1 &
cd ..
```

结论：
将 G-Retriever 的 PCST 子图检索思想迁移到 DAG-KV 图后，可以稳定保持较高的 graph recall（约 0.96），说明 PCST 风格的全局子图选择能够有效检索到答案相关区域。然而，answer recall 仅在 0.40–0.47 间小幅波动，表明现有 PCST 检索虽然能够找回答案相关子图，但缺乏将答案节点进一步组织为 DAG 末梢的结构归纳偏置。因此，仅依赖 PCST 风格的相关子图选择不足以满足 DAG-KV 的 answer-terminalization 需求，还需进一步设计面向末梢答案节点的定向与剪枝机制。

### 现有方法-2 UNIQORN：Group Steiner Tree

UNIQORN 是一种面向知识图谱与文本联合问答的统一推理框架，其核心思想是在问题相关证据构建的上下文图（context graph）上，通过问题线索（question cues）生成若干 anchor groups，并利用 Group Steiner Tree (GST) 在图中寻找能够同时连接这些锚点的最小子图，从而得到支持答案推理的证据结构。该机制能够在复杂图结构中自动发现连接多跳证据的关键路径，因此与本文需要从实体–属性图中检索最小推理子图的任务具有天然的契合性。

在本工作中，我们将已抽取的三元组构建为实体–属性图，并将其作为 UNIQORN 的上下文图输入。具体而言，我们首先根据问题文本与节点标签之间的语义相似度生成若干锚点集合，并将其组织为 anchor groups；随后在图上执行 Group Steiner Tree 搜索以获得若干候选证据子图，并根据节点在多棵 GST 中的出现频次进行排序。最终，我们将得到的无向子图定向为一个最小 DAG，使推理路径由问题相关实体逐步收敛至潜在答案节点，从而使答案尽可能位于 DAG 的末梢节点。通过这种方式，我们能够在保持 UNIQORN 推理思想的同时，使其适配本文的 DAG 子图检索任务。

```bash
python scripts/graph_gen/DAG_KV_UNIQORN.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --supporting_only \
  --limit 100 \
  --keep_score

cd scripts
./run_sweep_uniqorn.sh
cd ..

```

answer recall: 0.35~0.4

### 现有方法-3 SubgraphRAG：all-at-once 子图检索

我们选择 SubgraphRAG 作为 DAG-KV 检索任务的重要对比方法，主要因为它与本文问题在检索目标和结构形式上具有较高一致性：本文需要从由实体节点、属性节点及定向 KV 边构成的大图中，直接检索出一个规模受控、结构紧凑且尽可能包含答案证据的子图，而 SubgraphRAG 的核心思想正是对图中候选三元组进行并行打分，并在此基础上一次性选出与问题最相关的子图，避免了传统多跳逐步扩展方法中容易出现的路径误差累积与早期决策偏置问题。因此，本文将 SubgraphRAG 适配到 DAG-KV 场景中：首先将每个三元组对应的两条 KV 边视为有向图中的候选检索单元；然后结合问题与 key、value、relation 语义相似度，以及从问题锚点实体出发的方向性结构特征，对候选边进行统一评分；接着选取得分最高的一组边并进行连通性补全与环路消解，构造满足节点数和边数约束的最小化 DAG 子图；最后以答案是否出现在 DAG 末梢节点对应的 value 中来评估其检索效果。这样的适配既较好保留了 SubgraphRAG“并行子图检索”的方法优势，也使其能够自然服务于本文“答案尽可能落在 DAG 末梢”的任务目标。

```bash
python scripts/DAG_KV_SubgraphRAG.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 256 \
  --keep_score \
  --supporting_only \
  --max_nodes 30 \
  --max_edges 40 \
  --limit 100 \
  --max_sinks 8

cd scripts
nohup ./run_sweep_subgraphrag.sh >> dag_hotpot_subgraphrag.log 2>&1 &
cd ..
```

直接硬推：answer recall: 0.21~0.42

实现2：trainable MLP scorer

trainable MLP scorer：先为每条候选 KV edge 构造 question / src / relation / dst / key / value 的语义表示，再拼上 topic entity indicator 和 DDE 风格方向结构特征，最后用 MLP 学一个 relevance score

先训练，再推理

```bash
python scripts/DAG_KV_SubgraphRAG_trainable.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 10 \
  --lr 1e-3 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache.pkl \
  --hidden_dim 512

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/hotpot_dev_subgraphrag_trainable.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --seed_edge_topk 18 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score --cpu

# batch run
cd scripts
nohup ./run_sweep_subgrapn_train_infer.sh >> dag_hotpot_subgraphrag_trainable.log 2>&1 &
cd ..

```

answer recall: 0.61 ~ 0.73

mlp模型的迁移能力，Hotpot上训练，2wiki上推理

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/2wiki_train.jsonl \
  --output /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/2wiki_train_subgraphrag_trainable.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/sweep_train_infer/checkpoints/train_08_hd768_lr5e-4_neg6.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --seed_edge_topk 18 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score --cpu

```

### 现有方法最优

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/hotpot_dev_subgraphrag_trainable.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/sweep_train_infer/checkpoints/train_08_hd768_lr5e-4_neg6.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score --cpu



```

Answer recall: 0.7000
Graph recall: 0.9600
None-sink recall: 0.8800

#### parameter-free最优

```bash
python scripts/graph_gen/DAG_KV_HippoRAG.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --variant hipporag2 \
  --supporting_only \
  --seed_top_m 8 \
  --ppr_alpha 0.15 \
  --max_sinks 4 \
  --leaf_bonus 0.08 \
  --limit 100 \
  --keep_score
```

Answer recall: 0.4500
Graph recall: 0.9500
None-sink recall: 0.6900

### Next: Terminal-Aware DAG Pruning

首先确定：答案未成为 sink 的样本里，有多少是因为“答案节点还有出边”导致的。

```
Answer recall: 0.4500
Graph  recall: 0.9500
None-sink recall: 0.6900
Miss sink but in graph: 24
Among sink-miss samples, due to answer node having outgoing edges: 24 / 24 = 1.0000
Among sink-miss samples, other reasons: 0 / 24 = 0.0000

Answer recall: 0.7000
Graph  recall: 0.9600
None-sink recall: 0.8800
Miss sink but in graph: 18
Among sink-miss samples, due to answer node having outgoing edges: 18 / 18 = 1.0000
Among sink-miss samples, other reasons: 0 / 18 = 0.0000

```

两种情况下都是100%
结论：

- 现有方法的主要失败模式是 answer terminalization failure。
- 在失败样本中，答案节点未成为 sink 的原因 100% 是其仍有出边。
- 这些出边既包括 ATTRIBUTE，也包括 RELATION，说明问题不是局部属性噪声，而是缺少对“终点性”的全局建模

很多系统把重点放在“下一步扩什么”，却没有把“什么时候该停”建模成独立控制变量。2026 年一篇专门综述多跳 QA 检索—推理流程的工作，直接把 stop/continue criteria 列为四大设计轴之一，认为这是影响效果、效率和证据忠实度的关键环节 [Retrieval–Reasoning Processes for Multi-hop Question Answering: A Four-Axis Design Framework and Empirical Trends]

## 在SubGraphRAG上继续优化

- 现有方法的主要失败模式是 answer terminalization failure。
- 在失败样本中，答案节点未成为 sink 的原因 100% 是其仍有出边。
- 这些出边既包括 ATTRIBUTE，也包括 RELATION，说明问题不是局部属性噪声，而是缺少对“终点性”的全局建模

### answer-terminalization 后处理

对于最终子图中每个仍有出边的内部节点 v：
看它的最佳入边 best_in
如果 best_in 已经很强，且这个节点相对其前驱更“贴近问题”
同时它的后继边整体明显更弱
那么认为这个节点更像“应该停下来的终点”，就删除它的弱后继边
如果所有后继都弱，则直接把它变成 sink

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/hotpot_dev_subgraphrag_trainable.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/sweep_train_infer/checkpoints/train_08_hd768_lr5e-4_neg6.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --answer_terminalization \
  --keep_score --cpu
```

结果：
Answer recall: 0.6900
Graph recall: 0.9600
None-sink recall: 0.8600

纯后处理、又不依赖 answer 的 stop-expansion 偏置，已经开始伤到 coverage 了

### 增强hard negatives

把“答案节点的出边”强行加入负样本池，让模型学会“到答案后不要继续扩展”

重新collect_training_examples，再训练。

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v3.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 5e-4 \
  --neg_pos_ratio 6 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v2.pkl \
  --rebuild_train_cache \
  --hard_neg_ratio 0.25 \
  --hard_neg_min 1 \
  --hidden_dim 768

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v3.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/hotpot_dev_subgraphrag_trainable.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score --cpu
```

训练参数 hard_neg_ratio:

- 0.5: Answer recall: 0.6900 Graph recall: 0.9600 None-sink recall: 0.8100
- 0.25: Answer recall: 0.7400 Graph recall: 0.9600 None-sink recall: 0.8800

扫参数：

```bash
cd scripts/graph_gen
nohup ./run_sweep_subgrapn_train_infer_v3.sh > sweep_train_infer_v3.log  2>&1 &
cd ..
```

### 增加end_node model

把 edge scorer 改成 node-ending aware 的两阶段打分。

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v4.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v4.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 5e-4 \
  --neg_pos_ratio 6 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v4.pkl \
  --hidden_dim 768 \
  --end_hidden_dim 768 \
  --end_neg_pos_ratio 8 \
  --end_lr 5e-4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  --patience 10   --rebuild_train_cache

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v4.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/hotpot_dev_subgraphrag_trainable.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v4.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score --cpu

```

Answer recall: 0.7800
Graph recall: 0.9600
None-sink recall: 0.8600

### 共享编码器-联合训练

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v4.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3


python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/hotpot_dev_subgraphrag_trainable.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score --cpu


python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 4096 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --answer_aware \
  --keep_score

```

Answer recall: 0.8200
Graph recall: 0.9600
None-sink recall: 0.8800

## Train

1. 准备数据集

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_tmp.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --answer_aware \
  --keep_score --answerable_only

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 1000 \
  --keep_score --cpu --answer_aware --answerable_only
```

2. offline embedding

```bash
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --batch_size 1024

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --batch_size 1024

```

✅ Computed 427710 key embeddings and 427710 value embeddings.
✅ Saved embeddings → /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_key.npy / /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_value.npy

✅ Computed 11318 key embeddings and 11318 value embeddings.
✅ Saved embeddings → /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_key.npy / /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_value.npy

3. 训练

### train_dag_kv

```bash
export CUDA_VISIBLE_DEVICES=0

nohup python experiments/train_dag_kv.py \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --val_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --output_dir experiments/train/dag_kv_hotpot_v1 \
  --llm_type llama3 \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct \
  --encoder_spec qwen-embedding-0.6B \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_value.npy \
  --val_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_key.npy \
  --val_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_value.npy \
  --batch_size 1 \
  --lr 5e-4 \
  --use_lr_decay \
  --total_steps 8000 \
  --grad_accum_steps 20 \
  --eval_every 100 \
  --eval_samples 100 \
  --save_every 100 \
  --keep_top_k_ckpt 10 \
  --debug_print_every 10 \
  --path_attn \
  --use_multihop_adj \
  --max_hops 10 \
  --dynamic_hops_by_longest_path \
  --save_full_model > experiments/train/dag_kv_hotpot_v1/training_log.txt 2>&1 &


# 短实验：验证也用训练集，查看验证输出
python experiments/train_dag_kv.py \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --val_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --output_dir experiments/train/dag_kv_hotpot_v1 \
  --llm_type llama3 \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct \
  --encoder_spec qwen-embedding-0.6B \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_value.npy \
  --val_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_key.npy \
  --val_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_value.npy \
  --batch_size 1 \
  --lr 5e-4 \
  --use_lr_decay \
  --total_steps 8000 \
  --grad_accum_steps 20 \
  --eval_every 10 \
  --eval_samples 10 \
  --save_every 100 \
  --keep_top_k_ckpt 10 \
  --debug_print_every 10 \
  --path_attn \
  --use_multihop_adj \
  --max_hops 10 \
  --dynamic_hops_by_longest_path \
  --save_full_model
```

################################
V1: 虽然简化了，但是代码表现有点奇怪，训练精度太高，验证完全没有精度
################################

### train

使用原来的训练脚本

```bash
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v2 --save_period 100 --keep_top_k_ckpt 10  >> experiments/train/dag_kv_hotpot_v2/training_log.txt  2>&1 &
```

################################
V2: loss波动下降，验证代码有问题没有跑起来
################################

################################
V2.1
################################
优化：

- answerable_only: 只保留能够回答的样本做训练
- fix：空图跳过、验证问题

```bash
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v2.1 --save_period 100 --keep_top_k_ckpt 10  > experiments/train/dag_kv_hotpot_v2.1/training_log.txt  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python experiments/train.py \
  --seed 1 --B 1  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v2.2 --save_period 100 --keep_top_k_ckpt 10  > experiments/train/dag_kv_hotpot_v2.2/training_log.txt  2>&1 &
```

5-batch:

- 初始 loss 约 2.1，非常高。在前 几十到一百 step 内迅速下降到 ≈0.25 左右。
- Step 100–1500, loss 从 ≈0.25 缓慢下降到 ≈0.15
- Step 1500–4000 loss 从 ≈0.15 继续下降到 ≈0.06–0.08 左右
- 仍然存在一些 尖峰（spikes），但总体趋势稳定下降
- 然而训练过程中的验证发现，在最开始600步内精度从0.09提升到0.27，但是之后精度不再提升而是一直波动在0.25左右

### 训练精度优化

虽然answer recall 高，但是训练精度差
打印出来可以看到虽然正确答案在sink中，但是可能只是一个孤立节点，没有包含证据

先调小max_sinks:

```bash

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v4.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 100 \
  --keep_score

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v6.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 100 \
  --keep_score \
  --metrics_output experiments/subgraph_mlp/retrieval_metrics.json
```

### 问题分析

```bash

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score

python scripts/graph_gen/sink_distribution_analysis.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl  \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --output experiments/subgraph_mlp/sink_distribution_qkey.json \
  --edge_score_mode q_key \
  --plot_dir experiments/subgraph_mlp/plots \
  --plot_prefix dev --answer_only



python scripts/vialize_dag.py /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl --isfilter
```

结果表明有60%的answer sink没有任何入边，存在证据链断裂的问题。当Answer recall有0.8200时，Answer + supported sink ratio只有0.4800。
打印出来的可视化样本也证明了这个问题，正确答案虽然在sink中，但是是一个孤立节点，没有任何证据连接到它。而有些错误的sink虽然不是答案，但是有证据链连接到它们。在最终模型生成时反而有可能选择了这些错误的sink。

原因：

- edge 的弱监督定义太偏“答案值命中”。现在的 weak_label_edge 基本只把 value_matches_answer(e.value, answer) 的边标成正样本，而对“通向答案、但 value 本身不是答案”的桥接边没有奖励。这会直接带来一个偏差：模型更容易学会识别“答案边”，但不容易学会识别“支持答案的前驱边”，结果就是答案节点能进图，甚至能当 sink，但它前面的 support chain 没被保下来。
- subgraph 选择逻辑本身偏局部高分边，不偏“闭合到答案”。select_subgraph_edges 的策略本质上还是： 先按 edge score 全局 top-k 再做 topic-centered expansion 再做 connectivity patch。 这套流程没有任何一项是在显式优化“某个高 answer-like 节点至少保留一条入边”。所以只要答案前一跳的边分数不够高，它就会在早期直接掉出去。
- 当前 node-end scorer 也没真正监督“supported terminalization” 你现在的 node 监督是：只要某节点 incoming edge 中有 value 匹配 answer，就把它视为正 end-node。 这还是在学“答案节点像不像终点”，不是在学“答案终点是否应该被支持”。而且 mine_node_end_examples 主要压制的是答案节点后继和高 endness 的错误点，不是在提升答案入边保留率。

新的目标：answer on supported sink
把训练和推理目标改成下面这个优先级：

- 答案节点进图
- 答案节点成为 sink
- 答案 sink 至少有 1 条高质量入边
- 错误 sink 不要比正确 sink 拥有更强的支持链

#### 问题分析-进一步

细分成这几个指标：

A. pre-selection answer inbound edge recall
在原始图里，找出所有 dst 是答案节点的边，统计在 select_subgraph_edges 之后保留了多少。

B. post-cycle answer inbound edge recall
看这些边是不是在 break_cycles_to_dag 里被去掉了。

C. final supported answer sink ratio
现在已有的指标：answer recall vs answer + supported sink ratio。

```bash

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.1.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 8 \
  --limit 100 \
  --keep_score

```

NOTE：代码不是很完善，统计不准

#### 优化：edge supervision

增加正样本类型 2：support-to-answer edge 凡是满足下面任一条件的边，也标成正或弱正： e.dst 是答案节点 或 e.dst 可以一跳到答案节点 或 e.src -> ... -> answer 在 gold / oracle / weak path 里出现 最小实现甚至可以先做成： 先找所有 answer node 把所有 dst in answer_nodes 的边也标成正样本 但权重低于 direct answer edge 这样模型就会开始学习： “不是只有答案值本身重要，通向答案的最后一跳也重要。” 这是当前代码里最缺的信号。

对每个高 answer-like node 先找它的候选入边集合，按 edge score 排序； 如果这个节点本身分高，但当前没有入边被选中，就强制补入 1 条最优入边。

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v6.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v6_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v6.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  --edge_support_label 0.65 \
  --edge_predecessor_label 0.35 \
  --edge_pos_threshold 0.5

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v6.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v6_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 100 \
  --protected_inbound \
  --protected_inbound_topk 1 \
  --protected_inbound_min_score 0.55
```

Answer recall: 0.7500
Graph recall: 0.9600
None-sink recall: 0.8300
Answer + supported sink ratio: 0.4300
Sink relevance rank stats (merged across final sink counts):
all_samples=100 answer_sink_samples=75 top-1=0.8667 top-2=0.9595 top-3=1.0000

#### 优化：直接答案找入边

```bash

# 使用闭源模型API提取三元组
# /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_tripled_v4.3.jsonl
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_tripled_v4.3.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_tripled_v4.3_dag_v5.2.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 100 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4


python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 100 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4 \
  --verbose

#########################
python scripts/graph_gen/sink_distribution_analysis.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl  \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --output experiments/subgraph_mlp/sink_distribution_qkey.json \
  --edge_score_mode q_key \
  --plot_dir experiments/subgraph_mlp/plots \
  --plot_prefix dev --answer_only
python scripts/vialize_dag.py /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl --isfilter
#########################


python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4


python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl \
  --batch_size 1024

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --batch_size 1024

nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v5.2.1 --save_period 100 --keep_top_k_ckpt 10  > experiments/train/dag_kv_hotpot_v5.2.1/training_log.txt  2>&1 &


```

#### 优化edge scorer的标签 （v5 -> v5.3）

现在 v5 的 weak_label_edge 确实是“答案值边=1，其它=0”，这会天然压制中间支撑边。
可以改成：

- 先找每个训练的最优子图，把最优子图上的边打1
- 最优子图找法：如果一条边最终能： 指向答案节点，或者 位于从 topic 到答案节点的 gold 支撑路径上， 那就把它纳入正样本。

待实现：

- dedup_edges: 遍历kvedges，如果有多条边(src, dst)相同，则只保留其中score最大的那条

```bash

# 跑几组训练，看是否能正确挑出最优子图
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.3.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5.3.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 3 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v5.3.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  --rebuild_train_cache \
  --verbose

# 正式训练
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.3.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5.3.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 3 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v5.3.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  --rebuild_train_cache

# 用训练后的模型生成数据集
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.3.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5.3.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 1000 \
  --keep_score


python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.3.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5.3.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score --answerable_only

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3.jsonl \
  --batch_size 1024

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3.jsonl \
  --batch_size 1024

nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v5.3.1 --save_period 100 --keep_top_k_ckpt 5 > experiments/train/dag_kv_hotpot_v5.3.1/training_log.txt  2>&1 &

nohup bash /mnt/n0/KBLAM/KBLaM/docs/scripts/graph_gen/run_pipeline_subgraphrag_v5.3.sh > ./run_pipeline_v5.3.log 2>&1 &

```

优化edge-scorer首先会略微降低answer recall (0.8 -> 0.7)，同时answer+supported sink ratio也没有明显的提升

## Baseline

```bash
export CUDA_VISIBLE_DEVICES=2

python scripts/graph_gen/DAG_KV_Retriever_v3.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_retrieverv3.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 10 --use_seededge_beam --pred_weight 1 \
  --max_sinks 3

python scripts/graph_gen/DAG_KV_Retriever_v3.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_train_retrieverv3.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 10 --use_seededge_beam --pred_weight 1 \
  --answer_terminalization --max_sinks 3

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_retrieverv3.json \
  --batch_size 1024

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_train_retrieverv3.json \
  --batch_size 1024

export CUDA_VISIBLE_DEVICES=0
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_train_retrieverv3.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_train_retrieverv3_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_train_retrieverv3_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_retrieverv3.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_retrieverv3_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_retrieverv3_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_retrieverv3 --save_period 100 --keep_top_k_ckpt 10  > experiments/train/dag_kv_hotpot_retrieverv3/training_log.txt  2>&1 &

```

# 数据集清洗

## hotpot_dev_1k

- w/o knowledge distillation
  把没有知识注入的情况下也能回答正确的样本挑出去，保留faith score01为0的样本
  跑w/o knowledge

```bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16

python experiments/vector_rag.py  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/  --n-samples 100  --similarity-top-k 16 --without-knowledge --dis_out_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_without_knowledge.json

```

- answer + supported distillation
  把v5.2(找answer recall + 反向找证据)中answer recall但是没有支撑边的挑出来，保留其它样本

```bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 1000 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4 \
  --dis_out_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_without_islet.json

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 1000 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4 \
  --dis_out_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_supported.json
```

- 合并出来两版hotpot_dev
  - v1: w/o knowledge无法回答 + 没有孤岛
  - v2: w/o knowledge无法回答 + answer sink且有支撑

```bash
# baseline数据集
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/gen_clean_dataset.py \
  --source_file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --id_file_1 /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_without_knowledge.json \
  --id_file_2 /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_supported.json \
  --output_file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json

# 我们用数据集
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/gen_clean_dataset.py \
  --source_file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2.jsonl \
  --id_file_1 /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_without_knowledge.json \
  --id_file_2 /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_supported.json \
  --output_file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1.json

```

使用5.2.1模型直接跑测试：

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1.json \
  --batch_size 1024

python experiments/eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir experiments/train/dag_kv_hotpot_v5.2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7500_encoder/encoder.pt \
    --model_dir experiments/train/dag_kv_hotpot_v5.2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7500 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv \
    --test_dataset hotpot_dev_dag_v5.2_cleaned_v1.json \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type dag --query_size 100 --seed 1 --path_attn

# {'rouge1': 0.5012210309411118, 'rouge2': 0.22266666666666665, 'rougeL': 0.4930676293505906, 'rougeLsum': 0.49362068160597555, 'exact_match': 0.3, 'f1_overlap': 0.4823994708994709, 'bert_score_precision': 0.785520613193512, 'bert_score_recall': 0.7779142260551453, 'bert_score_f1': 0.7794387340545654, 'faithfulness': 0.45, 'faithfulness01': 0.45}

```

baseline重跑v1数据集：

```bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=2
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16

# wo knowledge
python experiments/vector_rag.py  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/  --n-samples 100  --similarity-top-k 16 --without-knowledge

# {'rouge1': 0.09067063492063493, 'rouge2': 0.014666666666666668, 'rougeL': 0.08931385281385282, 'rougeLsum': 0.0894484126984127, 'exact_match': 0.02, 'f1_overlap': 0.08662698412698411, 'bert_score_precision': 0.707589328289032, 'bert_score_recall': 0.6929397583007812, 'bert_score_f1': 0.6990845203399658, 'faithfulness': 0.088, 'faithfulness01': 0.08}


# vector rag
## 使用旧的索引，因为原来的索引里头应该是有所有样本
python experiments/vector_rag.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json    --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model /home/sdu/zhu/models/bge-en-v1.5/     --n-samples 100  --similarity-top-k 16 --index-path experiments/vector_rag_index/hotpot_bge --embedding-device cpu
# {'rouge1': 0.5733160800552104, 'rouge2': 0.3560714285714286, 'rougeL': 0.5694799861973774, 'rougeLsum': 0.5714152864044167, 'exact_match': 0.43, 'f1_overlap': 0.5701956521739131, 'bert_score_precision': 0.8240012526512146, 'bert_score_recall': 0.8197939991950989, 'bert_score_f1': 0.8200425505638123, 'faithfulness': 0.576, 'faithfulness01': 0.52}

# graph rag
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
conda activate autoschemakg
export CUDA_VISIBLE_DEVICES=4
python  ../../AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
  --kg-path /mnt/n0/KBLAM/AutoSchemaKG/example/generated/hotpot_dev/ \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --dataset-keyword hotpot_dev.json \
  --encoder-model /home/sdu/zhu/models/bge-en-v1.5/ \
  --llm-model llama_8b \
  --test-samples 100 \
  --llm-endpoint 'http://127.0.0.1:8001/v1' \
  --topN 16
# {'rouge1': 0.5653693340972752, 'rouge2': 0.3184122807017544, 'rougeL': 0.5654768270944741, 'rougeLsum': 0.5667426470588234, 'exact_match': 0.44, 'f1_overlap': 0.562335688820983, 'bert_score_precision': 0.7742792367935181, 'bert_score_recall': 0.7932407855987549, 'bert_score_f1': 0.7811219692230225, 'faithfulness': 0.554, 'faithfulness01': 0.53}

```

## 重新训练一版5.2.2 (✅️)

```bash
# 重构训练集
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --batch_size 1024



nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v5.2.2 --save_period 100 --keep_top_k_ckpt 5  > experiments/train/dag_kv_hotpot_v5.2.2/training_log.txt  2>&1 &


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

# {'rouge1': 0.5490039971485202, 'rouge2': 0.2851111111111111, 'rougeL': 0.5460418541716715, 'rougeLsum': 0.5479309106489632, 'exact_match': 0.37, 'f1_overlap': 0.5317086691086691, 'bert_score_precision': 0.8157083988189697, 'bert_score_recall': 0.8178564310073853, 'bert_score_f1': 0.8146867156028748, 'faithfulness': 0.531, 'faithfulness01': 0.52}




```

## 减小B(5.2.3❌️)

- 4090可以用B=3跑
  看是否有变化

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python experiments/train.py \
  --seed 1 --B 3  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v5.2.3 --save_period 100 --keep_top_k_ckpt 5  > experiments/train/dag_kv_hotpot_v5.2.3/training_log.txt  2>&1 &
```

训练了四千多步，目前看来差不大多

---

acc=0.53

---

## raw triple(DAG-KV-Retriever v7❌️)

修改DAG-KV-Retriever, 使用程序化生成的kv对取代原来大模型生成的kv对：

- aggressive mode:
  forward key: "{type*prefix} | {subject} | {relation}"
  backward key: "{type_prefix} | {object} | reverse*{relation}"
- templated mode:
  属性三元组:
  正向: "the {relation} of {subject} is" -> value = object
  反向: "the entity whose {relation} is {object} is" -> value = subject
  关系三元组:
  正向: "{subject} {relation}" -> value = object
  反向: "the entity that {relation} {object} is" -> value = subject

---

精度下降：acc=0.4844

---

```bash
# 重新训练
export CUDA_VISIBLE_DEVICES=1
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v7.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v7.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v7.pkl \
  --hidden_dim 1024 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  --program_kv_mode templated


python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v7.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v7.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 1000 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4 \
  --dis_out_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_v7_supported.json
# Answer recall: 0.7180
# Graph  recall: 0.9180
# None-sink recall: 0.7840
# Answer + supported sink ratio: 0.5360
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=1000 answer_sink_samples=718 top-1=0.7563 top-2=0.9313 top-3=1.0000
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/gen_clean_dataset.py \
  --source_file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7.jsonl \
  --id_file_1 /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_without_knowledge.json \
  --id_file_2 /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_supported.json \
  --output_file /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1.json

python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v7.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v7.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1.json \
  --batch_size 1024

python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7.jsonl \
  --batch_size 1024


nohup python experiments/train.py \
  --seed 1 --B 3  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v7 --save_period 100 --keep_top_k_ckpt 5  > experiments/train/dag_kv_hotpot_v7/training_log.txt  2>&1 &

# 顺便测一下参数
export CUDA_VISIBLE_DEVICES=3
nohup python experiments/train.py \
  --seed 1 --B 3  --lr 3e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v7.1 --save_period 100 --keep_top_k_ckpt 5  > experiments/train/dag_kv_hotpot_v7.1/training_log.txt  2>&1 &
## 偶发性的OOM崩溃
nohup python experiments/train.py \
  --seed 1 --B 3  --lr 3e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v7_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v7_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_dir_to_resume experiments/train/dag_kv_hotpot_v7.1/stage1_lr_0.0003KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_4200 \
  --model_save_dir experiments/train/dag_kv_hotpot_v7.1 --save_period 100 --keep_top_k_ckpt 5  >> experiments/train/dag_kv_hotpot_v7.1/training_log.txt  2>&1 &

```

## GPT三元组抽取 (❌️)

逐条优化精度

- 修复模型输出后处理
- 找推理出错的样本，分析错误原因，重新提取三元组和建DAG图

```bash
export CUDA_VISIBLE_DEVICES=1
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16

python experiments/eval_generation.py generation \
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
    --dataset_type dag --query_size 100 --seed 1 --path_attn \
    --save_dir experiments/gen_output_debug/triple 
```
示例样本：
5adf4ba65542992d7e9f931c
GT: Beauty and the Beast	PRED: The Lion King

1. 尝试直接修改dag图(hotpot_dev_dag_v5.2_cleaned_v1.json)
````bash
#1.1 复制
cp /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1.json /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1.json 

#1.2 修改polished1，使用chatgpt生成的替换“dag”

#1.3 重新计算embedding
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1.json \
  --batch_size 1024

python experiments/eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir experiments/train/dag_kv_hotpot_v5.2.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7300_encoder/encoder.pt \
    --model_dir experiments/train/dag_kv_hotpot_v5.2.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_llama3_step_7300 \
    --kb_layer_frequency 3 --kb_scale_factor 5 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv \
    --test_dataset hotpot_dev_dag_v5.2_cleaned_v1_polished1.json \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type dag --query_size 100 --seed 1 --path_attn \
    --save_dir experiments/gen_output_debug/triple_polished1
````
直接改没用，模型的输出不会变。
调整kb_scale_factor（比如增大到5）可以让他输出正确答案，但是其它样本也可能出错。

2. 重新生成三元组-DAGRetriever-dag图

## 调整kb_scale_factor重新训练(5.2.4✅️)

- 增加训练时候的kb_scale_factor到5
- 验证也到5
- 使用polished验证集

````bash
export CUDA_VISIBLE_DEVICES=1
nohup python experiments/train.py \
  --seed 1 --B 3  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --kb_scale_factor 5 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 5 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v5.2.4 --save_period 100 --keep_top_k_ckpt 5  > experiments/train/dag_kv_hotpot_v5.2.4/training_log.txt  2>&1 &

````

## 关闭seperate_query_head (5.2.5)

````bash
export CUDA_VISIBLE_DEVICES=2
nohup python experiments/train.py \
  --seed 1 --B 3  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --kb_scale_factor 5 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_polished1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 5 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_v5.2.5 --save_period 100 --keep_top_k_ckpt 5  > experiments/train/dag_kv_hotpot_v5.2.5/training_log.txt  2>&1 &

````


# Qwen3

训练一次Hotpot测试qwen模型的性能：
````bash
uv venv kblam --python 3.12
source /mnt/n0/uv_envs/kblam/bin/activate

nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.2_qwen-embedding-0.6B_embd_value.npy \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_hotpot_qwen3_4B_v1 --save_period 100 --keep_top_k_ckpt 5  > experiments/train/dag_kv_hotpot_qwen3_4B_v1/training_log.txt  2>&1 &

# 扫scale_factor_kb
source /mnt/n0/uv_envs/kblam/bin/activate
export CUDA_VISIBLE_DEVICES=0
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv \
  --test_dataset hotpot_dev_dag_v5.2_cleaned_v1.json \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_hotpot_qwen3_4B_v1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_4200_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_hotpot_qwen3_4B_v1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_4200 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.2_cleaned_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 4200 \
  --t_step 8000 \
  --kb_scale_factor_range 0.5 8.0 \
  --exp_config_name qwen3_step4200_kbsf_range \
  --save_dir /mnt/n0/PathWeaver/experiments/eval_results \
  --seed 1

````

## multi-hop training set

### triple_gen

training_set:
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl (不同版本)
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/hotpot_train_tripled_v5-qwen2.5-72B_4bit.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl

合并数据集:
  合并多个数据集文件
  默认只保留 answer_sufficient == True 的样本
  统一 id 和 _id 为 dataset_sourceid
  保留 dataset 和 source_id 字段，方便回溯来源
  先按合并后的 id 去重
  再按样本内容去重，避免跨文件或源内重复内容残留
  遇到重复时优先保留“质量更高”的版本：
    answer_sufficient=True
    triple_list 更长
    context 更多
    revision_notes 更多
    整体 payload 更大
  额外参数：--allow-answer-insufficient

````bash
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/merge_tripled_datasets.py \
  --dataset hotpot=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/hotpot_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --dataset 2wiki=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --dataset musique=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit.jsonl \
  --stats-output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit.stats.json
````

test_set:
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json
    包含：{'compositional': 5234, 'comparison': 3022, 'inference': 1549}
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json
    包含：{'bridge': 350}
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl
    包含：answerable 2417

提取测试集的三元组：
````bash
# setup 4ul40
export CUDA_VISIBLE_DEVICES=0,1
source /mnt/n0/uv_envs/kblam-rag/bin/activate
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen2.5-72B-4bit/ \
  --served-model-name Qwen2.5-72B-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code > Qwen2.5-72B-4bit.log 2>&1 &
nohup bash -c '
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 128 \
  --skip-comparison \
  --limit 300 \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/2wiki_dev \
  --sample-retries 2 \
  --answer-aware;
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 128 \
  --skip-comparison \
  --limit 300 \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/hotpot_dev \
  --sample-retries 2 \
  --answer-aware;
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 128 \
  --skip-comparison \
  --limit 300 \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/musique_dev \
  --sample-retries 2 \
  --answer-aware
' > dev_tripled_v5-qwen2.5-72B_4bit.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0,1
source /mnt/n0/uv_envs/kblam-rag/bin/activate
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 16384  \
  --kv-cache-dtype auto \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}'  \
  --trust-remote-code > Qwen3.5-27B.log 2>&1 &
nohup bash -c '
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 64 \
  --skip-comparison \
  --limit 300 \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/2wiki_dev_qwen3.5_27B \
  --sample-retries 2 \
  --answer-aware;
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 64 \
  --skip-comparison \
  --limit 300 \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/hotpot_dev_qwen3.5_27B \
  --sample-retries 2 \
  --answer-aware;
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 64 \
  --skip-comparison \
  --limit 300 \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/musique_dev_qwen3.5_27B \
  --sample-retries 2 \
  --answer-aware
' >> dev_tripled_v5-qwen3.5-27B.log 2>&1 &

# setup 8u4090
export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate triple
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen2.5-72B-4bit/ \
  --served-model-name Qwen2.5-72B-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code > Qwen2.5-72B-4bit.log 2>&1 &

nohup python build_knowledge_graph_v5.py \
  --input /home/zhchen/zwb/datasets/dev_set/2wiki_dev_2hop.json \
  --output /home/zhchen/zwb/datasets/2wiki_dev_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 128 \
  --skip-comparison \
  --limit 300 \
  --resume \
  --stage-cache-dir /home/zhchen/zwb/datasets/v5_cache/2wiki_dev \
  --sample-retries 2 \
  --answer-aware > 2wiki_dev_tripled_v5-qwen2.5-72B_4bit.log 2>&1 &
````

### graph_gen

````bash

nohup python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --train_batch_size 1024 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_merged_v1.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  > /mnt/n0/PathWeaver/docs/scripts/graph_gen/train_subgraph.log 2>&1 &

nohup python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit_dag_v1.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4 > /mnt/n0/PathWeaver/docs/scripts/graph_gen/infer_subgraph.log 2>&1 &
# Answer recall: 0.9291
# Graph  recall: 0.9820
# None-sink recall: 0.9291
# Answer + supported sink ratio: 0.8504
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=159126 answer_sink_samples=159126 top-1=0.7961 top-2=0.9490 top-3=1.0000
# [DONE] input=171262 output=159126 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit_dag_v1.jsonl



python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
# Answer recall: 0.8565
# Graph  recall: 0.9662
# None-sink recall: 0.8565
# Answer + supported sink ratio: 0.8481
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=203 answer_sink_samples=203 top-1=0.7340 top-2=0.9212 top-3=1.0000
# [DONE] input=237 output=203 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
# Answer recall: 0.8833
# Graph  recall: 0.9933
# None-sink recall: 0.8833
# Answer + supported sink ratio: 0.8167
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=265 answer_sink_samples=265 top-1=0.8302 top-2=0.9692 top-3=1.0000
# [DONE] input=300 output=265 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
# Answer recall: 0.6833
# Graph  recall: 0.8967
# None-sink recall: 0.6833
# Answer + supported sink ratio: 0.6067
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=205 answer_sink_samples=205 top-1=0.7171 top-2=0.9064 top-3=1.0000
# [DONE] input=300 output=205 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl

````

### create_embeddings

````bash
export CUDA_VISIBLE_DEVICES=1

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit_dag_v1.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --batch_size 1024
````

### merged training

````bash
source /mnt/n0/uv_envs/kblam/bin/activate
export CUDA_VISIBLE_DEVICES=1

nohup python experiments/train.py \
  --seed 1 --B 15  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit_dag_v1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 2 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_multihop_qwen3_4B_v1 --save_period 100 --keep_top_k_ckpt 5  >> experiments/train/dag_kv_merged_multihop_qwen3_4B_v1/training_log.txt  2>&1 &

````

结论：
 - batch size 5/15: 精度上没有区别，5的速度更快
 - kb_scale_factor(训练和验证统一) 4>2>1

### Inference

model_chkp:
- /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/
- /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt

test_set_anwer_aware:
- /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl 
- /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl 
- /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl

test_set_no_aware:
````bash
source /mnt/n0/uv_envs/kblam/bin/activate
export CUDA_VISIBLE_DEVICES=3

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
# Answer recall: 0.8439
# Graph  recall: 0.9662
# None-sink recall: 0.8650
# Answer + supported sink ratio: 0.8397
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=237 answer_sink_samples=200 top-1=0.7400 top-2=0.9250 top-3=1.0000
# [DONE] input=237 output=237 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
# Answer recall: 0.8233
# Graph  recall: 0.9933
# None-sink recall: 0.8867
# Answer + supported sink ratio: 0.7700
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=300 answer_sink_samples=247 top-1=0.8300 top-2=0.9711 top-3=1.0000
# [DONE] input=300 output=300 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
# Answer recall: 0.5900
# Graph  recall: 0.8967
# None-sink recall: 0.7133
# Answer + supported sink ratio: 0.5500
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=300 answer_sink_samples=177 top-1=0.7627 top-2=0.9432 top-3=1.0000
# [DONE] input=300 output=300 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --batch_size 1024

````

检索开销：
````bash
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/retrieval_speed_test.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1 \
  --infer_batch_size 1 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4 \
  --limit 10 --profile_online_latency

# samples=1 question_encode=0.4107s feature_build=0.0099s model_scoring=0.0468s dag_postprocess=0.0016s export=0.0002s online_total=0.4691s
````
question ecode时间占大头，但是这部分时间在原来的Vector RAG中也有
剩下的时间包括打分、dag图构建的额外时间不超过0.06s



````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset 2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
 # QPS: 3.41
 # ---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.5005225885225886, 'rouge2': 0.14866666666666667, 'rougeL': 0.5028831168831169, 'rougeLsum': 0.5009446664446664, 'exact_match': 0.37, 'f1_overlap': 0.4944163059163059, 'faithfulness01': 0.42}
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset 2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_no_aware_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_no_aware_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 

python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
 # QPS: 3.41
 # ---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.5005225885225886, 'rouge2': 0.14866666666666667, 'rougeL': 0.5028831168831169, 'rougeLsum': 0.5009446664446664, 'exact_match': 0.37, 'f1_overlap': 0.4944163059163059, 'faithfulness01': 0.42}
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
  # QPS: 2.76
  # ---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.4701825396825397, 'rouge2': 0.1725, 'rougeL': 0.46938095238095234, 'rougeLsum': 0.4680396825396824, 'exact_match': 0.27, 'f1_overlap': 0.46112698412698416, 'faithfulness01': 0.34}

python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 

# QPS: 2.80
# ---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.4625455655455654, 'rouge2': 0.29529365079365083, 'rougeL': 0.45640437340437334, 'rougeLsum': 0.46147363747363734, 'exact_match': 0.26, 'f1_overlap': 0.4472585192585193, 'faithfulness01': 0.29}
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset musique_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_multihop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_no_aware_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
````

## multi/single-hops

数据集准备：
  - popqa
  - squad: tripled版本（v4）中的问题polish过，替换成原版的问题(v4.1)
````bash
python /mnt/n0/datasets/popqa/popqa2dag.py \
  --input /mnt/n0/datasets/popqa/test.tsv \
  --output /mnt/n0/datasets/popqa/test_dag_singlehop.jsonl
# [DONE] converted=14267 saved_to=/mnt/n0/datasets/popqa/test_dag_singlehop.jsonl
python /mnt/n0/datasets/popqa/popsplit.py
# [DONE] saved 14167 samples to /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/popqa.jsonl
# [DONE] saved 100 samples to /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl

python /mnt/n0/datasets/squad/v4/squad_v4_to_dag.py \
  --input /mnt/n0/datasets/squad/v4/train_datasets_validation_questions.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/squad_v4.1.jsonl

python /mnt/n0/datasets/squad/v4/squad_v4_to_dag.py \
  --input /mnt/n0/datasets/squad/v4/test_datasets_validation_questions.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1.jsonl
````

合并training_set
````bash
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merge_tripled_datasets.py \
  --dataset popqa=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/popqa.jsonl \
  --dataset squad=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/squad_v4.1.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_single_hop_train_tripled_v5.1.jsonl \
  --stats-output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_single_hop_train_tripled_v5.1.stats.json
# [DONE] wrote 101668 samples to /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_single_hop_train_tripled_v5.1.jsonl

python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merge_tripled_datasets.py \
  --dataset multihop=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit_dag_v1.jsonl \
  --dataset singlehop=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_single_hop_train_tripled_v5.1.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.jsonl \
  --stats-output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.stats.json
# [DONE] wrote 260794 samples to /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.jsonl

````

创建训练/测试集embedding
````bash
nohup python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.jsonl \
  --batch_size 1024 > /mnt/n0/PathWeaver/docs/scripts/embedding.log 2>&1 &
nohup python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.jsonl \
  --batch_size 1024 > /mnt/n0/PathWeaver/docs/scripts/embedding.log 2>&1 &

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1.jsonl \
  --batch_size 1024
````

### Hybrid Training

````bash
export CUDA_VISIBLE_DEVICES=0
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4 --save_period 100 --keep_top_k_ckpt 5  >> experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/training_log.txt  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 8 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb8 --save_period 100 --keep_top_k_ckpt 5  >> experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb8/training_log.txt  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 16 \
  --eval_step 100 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb16 --save_period 100 --keep_top_k_ckpt 5  >> experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb16/training_log.txt  2>&1 &
````

- v1.1
squad数据集调整问题(v4.1)

````bash
export CUDA_VISIBLE_DEVICES=1
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1 --save_period 200 --keep_top_k_ckpt 5  >> experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1/training_log.txt  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 12000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1.1 --save_period 200 --keep_top_k_ckpt 5  >> experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1.1/training_log.txt  2>&1 &
````


解析训练日志：
````bash
source /mnt/n0/uv_envs/kblam/bin/activate
python /mnt/n0/PathWeaver/experiments/train/show_training_score.py \
  --f /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/training_log.txt


````

结论：
 - kb 4>8>16

### Hybrid Inference

checkpoint: /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1.1/stage1_lr_0.005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_11000
- musique_dev_tripled_v5-qwen3.5-27B_dag_v1.json              
                    l, num_samples: 205, scale_factor: 4.0, FA01                
                    Score: 0.39 
- hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl              
                    , num_samples: 265, scale_factor: 4.0, FA01                 
                    Score: 0.5     
- 2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.j              
                    sonl, num_samples: 203, scale_factor: 4.0,                  
                    FA01 Score: 0.45 
- popqa.jsonl,train.py:1297
                    num_samples: 100, scale_factor: 4.0, FA01                   
                    Score: 0.84    

- popqa
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset popqa.jsonl  \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
````

- squad
  v4.1测试集有问题
  v4.0的前100条精度太高，可以开seed随机选
QPS: 2.91
---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.7472254991711647, 'rouge2': 0.3684237764224976, 'rougeL': 0.7338296653581484, 'rougeLsum': 0.736851076397516, 'exact_match': 0.55, 'f1_overlap': 0.7075176033460864, 'faithfulness01': 0.65}
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset squad_v4.jsonl  \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_11000 \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_11000_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 --seed 2

````

- 2wiki
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset 2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
````


- hotpot
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
````

- musique
QPS: 2.50
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999/ \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1_kb4/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7999_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 7999 \
  --t_step 8000 \
  --kb_scale_factor 4 
````


## MintQA

### 1.转格式

````bash
cd /mnt/n0/datasets/multi-hop/MintQA/
python convert_to_vector_rag.py test \
--limit 1 \
--output /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag.jsonl
````

### 2.剪枝
MintQA原图太大，做剪枝:
````bash
# 3分钟
python3 /mnt/n0/datasets/multi-hop/MintQA/convert_to_vector_rag.py \
  test \
  --max-triples-per-sample 256 \
  --read-workers 8 \
  --read-batch-size 16384 \
  --io-batch-size 128 \
  --num-workers 16 \
  --output /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256.jsonl

# 多线程会爆
nohup python3 /mnt/n0/datasets/multi-hop/MintQA/convert_to_vector_rag.py \
  train \
  --max-triples-per-sample 256 \
  --read-workers 16 \
  --read-batch-size 16384 \
  --io-batch-size 128 \
  --num-workers 1 \
  --output /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_256.jsonl >> /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_256.log 2>&1 &
````

剪枝到64
````bash
python3 /mnt/n0/datasets/multi-hop/MintQA/convert_to_vector_rag.py \
  test \
  --max-triples-per-sample 64 \
  --read-workers 8 \
  --read-batch-size 16384 \
  --io-batch-size 128 \
  --num-workers 16 \
  --output /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64.jsonl
  # Hop distribution (shortest supporting path length):
  #   1-hop: 882
  #   2-hop: 673
  #   3-hop: 133
  #   4-hop: 2
  #   no-path: 284

python3 /mnt/n0/datasets/multi-hop/MintQA/convert_to_vector_rag.py \
  train \
  --max-triples-per-sample 64 \
  --read-workers 16 \
  --read-batch-size 16384 \
  --io-batch-size 128 \
  --num-workers 1 \
  --output /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64.jsonl
  # Hop distribution (shortest supporting path length):
  #   1-hop: 3513
  #   2-hop: 2771
  #   3-hop: 479
  #   4-hop: 2
  #   no-path: 1123
````

### 3. 过滤多跳样本

````bash
python3 /mnt/n0/datasets/multi-hop/MintQA/filter_multi_hop.py \
 --input-file /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64.jsonl \
 --output-file /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2.jsonl \
 --min-hop 2
  # Load 1974 samples from /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64.jsonl
  # Filter 808 samples to /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2.jsonl
python3 /mnt/n0/datasets/multi-hop/MintQA/filter_multi_hop.py \
 --input-file /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64.jsonl \
 --output-file /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_3.jsonl \
 --min-hop 3
  # Filter 135 samples to /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_3.jsonl

python3 /mnt/n0/datasets/multi-hop/MintQA/filter_multi_hop.py \
 --input-file /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64.jsonl \
 --output-file /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_2.jsonl \
 --min-hop 2
  # Load 7888 samples from /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64.jsonl
  # Filter 3252 samples to /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_2.jsonl

python3 /mnt/n0/datasets/multi-hop/MintQA/filter_multi_hop.py \
 --input-file /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64.jsonl \
 --output-file /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_1.jsonl \
 --min-hop 1
  # Load 7888 samples from /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64.jsonl
  # Filter 6765 samples to /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_1.jsonl

````

vector-rag推理：
````bash
python3 /mnt/n0/PathWeaver/experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256.jsonl \
  --dataset-type mintqa \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --max-model-len 32768 \
  --mintqa_min_hop 2 \
  --index-path /mnt/n0/PathWeaver/experiments/vector_rag_index/mintqa_dev_bge_256

# {'rouge1': 0.8834848484848485, 'rouge2': 0.6222222222222222, 'rougeL': 0.8836363636363638, 'rougeLsum': 0.8810101010101008, 'exact_match': 0.87, 'f1_overlap': 0.8820808080808081, 'faithfulness01': 0.87}

python3 /mnt/n0/PathWeaver/experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256.jsonl \
  --dataset-type mintqa \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --mintqa_min_hop 2 \
  --index-path /mnt/n0/PathWeaver/experiments/vector_rag_index/mintqa_dev_bge_256
  # {'rouge1': 0.33491916416916406, 'rouge2': 0.19285714285714284, 'rougeL': 0.33516288286876517, 'rougeLsum': 0.33220739391327625, 'exact_match': 0.28, 'f1_overlap': 0.3362784241901889, 'faithfulness01': 0.32}


python /mnt/n0/PathWeaver/docs/experiments/msa.py \
  --dataset-path /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256.jsonl \
  --dataset-type mintqa \
  --mintqa_min_hop 2 \
  --n-samples 100 \
  --block-size 2048 \
  --memory-cache-dir /mnt/n0/PathWeaver/experiments/msa_cache/mintqa_test_pruned_256 \
  --max-batch-size 1 \
  --model-path /mnt/n0/models/MSA-4B

# ========== Latency ==========
# Per-question answer time: mean=3.1695s, p50=2.5817s, p95=4.7068s
# Average generate count per request: 3.0300
# Total input tokens per request: mean=298.7700, p50=252.0000, p95=599.0000
# Total output tokens per request: mean=487.0700, p50=410.0000, p95=929.0000
# Retrieval recall: mean=0.0000, p50=0.0000, p95=0.0000
# Retrieval hit@16: mean=0.0000
# Retrieval all-support-hit@16: mean=0.0000
# Throughput (QPS): 0.32
# =============================
# {'rouge1': 0.22285867204259013, 'rouge2': 0.07213852813852813, 'rougeL': 0.22314523474409242, 'rougeLsum': 0.2245690490243535, 'exact_match': 0.14, 'f1_overlap': 0.22226666856951888, 'faithfulness01': 0.24}

````

### Tripled-Format & Graph Gen

````bash
python /mnt/n0/datasets/multi-hop/MintQA/convert_to_tripled.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256.jsonl \
  --output /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256_tripled.jsonl
python /mnt/n0/datasets/multi-hop/MintQA/convert_to_tripled.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_256.jsonl \
  --output /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_256_tripled.jsonl

# zero-shot, 使用 merged_multihop训练的mlp模型推
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_256_tripled_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

# Answer recall: 0.6575
# Graph  recall: 0.8708
# None-sink recall: 0.6575
# Answer + supported sink ratio: 0.4752

# 在mintqa上训+推
nohup python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode train \
  --input /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_256_tripled.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v1_joint_mintqa_only.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --train_batch_size 1024 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_merged_v1_mintqa_only.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  > /mnt/n0/PathWeaver/docs/scripts/graph_gen/train_subgraph.log 2>&1 &

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_256_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/mintqa_pruned_256_tripled_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v1_joint_mintqa_only.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

# Answer recall: 0.8354
# Graph  recall: 0.8695
# None-sink recall: 0.8354
# Answer + supported sink ratio: 0.6471

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_256_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_256_tripled_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v1_joint_mintqa_only.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
  # Answer recall: 0.7882
  # Graph  recall: 0.8708
  # None-sink recall: 0.7882
  # Answer + supported sink ratio: 0.5760
  # Sink relevance rank stats (merged across final sink counts):
  #   all_samples=1556 answer_sink_samples=1556 top-1=0.9165 top-2=0.9715 top-3=1.0000

````

#### pruned_64
````bash
python /mnt/n0/datasets/multi-hop/MintQA/convert_to_tripled.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64.jsonl \
  --output /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_tripled.jsonl
python /mnt/n0/datasets/multi-hop/MintQA/convert_to_tripled.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64.jsonl \
  --output /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_tripled.jsonl

python /mnt/n0/datasets/multi-hop/MintQA/convert_to_tripled.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2.jsonl \
  --output /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2_tripled.jsonl
python /mnt/n0/datasets/multi-hop/MintQA/convert_to_tripled.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_3.jsonl \
  --output /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_3_tripled.jsonl
python /mnt/n0/datasets/multi-hop/MintQA/convert_to_tripled.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_1.jsonl \
  --output /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_1_tripled.jsonl

# zero-shot：
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_64_tripled_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
  # Answer recall: 0.7690
  # Graph  recall: 0.8597
  # None-sink recall: 0.7690
  # Answer + supported sink ratio: 0.4569
  # Sink relevance rank stats (merged across final sink counts):
  #   all_samples=1518 answer_sink_samples=1518 top-1=0.8676 top-2=0.9483 top-3=1.0000

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_64_tripled_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v1_joint_mintqa_only.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
  # Answer recall: 0.8105
  # Graph  recall: 0.8597
  # None-sink recall: 0.8105
  # Answer + supported sink ratio: 0.4210
  # Sink relevance rank stats (merged across final sink counts):
  #   all_samples=1600 answer_sink_samples=1600 top-1=0.9400 top-2=0.9822 top-3=1.0000

# 训推
nohup bash -c '
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode train \
  --input /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_tripled.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v1_joint_mintqa_only_pruned_64.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --train_batch_size 1024 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_merged_v1_mintqa_only_pruned_64.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/mintqa_pruned_64_tripled_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v1_joint_mintqa_only_pruned_64.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
' > /mnt/n0/PathWeaver/docs/scripts/graph_gen/train_subgraph.log 2>&1 &

# Answer recall: 0.8370
# Graph  recall: 0.8626
# None-sink recall: 0.8370
# Answer + supported sink ratio: 0.4855
# Sink relevance rank stats (merged across final sink counts):
#   all_samples=6602 answer_sink_samples=6602 top-1=0.9440 top-2=0.9850 top-3=1.0000
````

### Merged Hybrid V2

创建混合训练集：
````bash
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merge_tripled_datasets.py \
  --dataset mergedv1=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.jsonl \
  --dataset mintqa=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/mintqa_pruned_256_tripled_dag.jsonl \
  --output  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.jsonl \
  --stats-output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.stats.json \
  --allow-answer-insufficient

python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merge_tripled_datasets.py \
  --dataset mergedv1=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_hop_train_dag_v1.1.jsonl \
  --dataset mintqa=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/mintqa_pruned_64_tripled_dag.jsonl \
  --output  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.1.jsonl \
  --stats-output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.1.stats.json \
  --allow-answer-insufficient
````



创建embedding：
````bash
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.jsonl \
  --batch_size 1024

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_256_tripled_dag.jsonl \
  --batch_size 1024

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.1.jsonl \
  --batch_size 1024

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_64_tripled_dag.jsonl \
  --batch_size 1024

````

训练
````bash

nohup python experiments/train.py \
  --seed 1 --B 5  --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --use_cached_embd \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.1.jsonl \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.1_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_v2.1_qwen-embedding-0.6B_embd_value.npy \
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_64_tripled_dag.jsonl\
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_64_tripled_dag_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_v1_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned_64_tripled_dag_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 10000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_qwen3_4B_v2 --save_period 200 --keep_top_k_ckpt 5  > experiments/train/dag_kv_merged_hybrid_qwen3_4B_v2/training_log.txt  2>&1 &

````

### Merged Hybrid V2.1

#### Graph Gen
纯单跳(已经有了dag图: [1])：
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/popqa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/squad_v4.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl

对多跳数据集先合并一版训练集，然后训练mlp模型
训练集:
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/hotpot_train_tripled_v5-qwen2.5-72B_4bit.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl
  - /mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_1_tripled.jsonl

多跳数据集的测试集
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B.jsonl
  - /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2_tripled.jsonl


轻量版的合并脚本，ID级去重，只有样本里存在 answer_sufficient 且它明确为 False 时才过滤
````bash
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merge_tripled_datasets_lightweight.py \
  --dataset hotpot=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/hotpot_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --dataset 2wiki=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --dataset musique=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --dataset mintqa=/mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_1_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_multihop_training_v2.1.jsonl \
  --stats-output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_multihop_training_v2.1.stats
````
输出：
```
{
  "input_datasets": [
    {
      "dataset": "hotpot",
      "path": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/hotpot_train_tripled_v5-qwen2.5-72B_4bit.jsonl",
      "loaded": 72991,
      "dropped_answer_sufficient_false": 482,
      "duplicate_ids_dropped": 0,
      "final_kept": 72509
    },
    {
      "dataset": "2wiki",
      "path": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl",
      "loaded": 84130,
      "dropped_answer_sufficient_false": 911,
      "duplicate_ids_dropped": 3353,
      "final_kept": 79866
    },
    {
      "dataset": "musique",
      "path": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl",
      "loaded": 19874,
      "dropped_answer_sufficient_false": 987,
      "duplicate_ids_dropped": 0,
      "final_kept": 18887
    },
    {
      "dataset": "mintqa",
      "path": "/mnt/n0/datasets/multi-hop/MintQA/train_vector_rag_pruned_64_hop_1_tripled.jsonl",
      "loaded": 6765,
      "dropped_answer_sufficient_false": 0,
      "duplicate_ids_dropped": 0,
      "final_kept": 6765
    }
  ],
  "input_samples": 183760,
  "dropped_answer_sufficient_false": 2380,
  "duplicate_ids_dropped": 3353,
  "final_samples": 178027,
  "final_dataset_counts": {
    "2wiki": 79866,
    "hotpot": 72509,
    "mintqa": 6765,
    "musique": 18887
  }
}
```

在训练集上训练+训练集推理+测试集推理(answer-aware开或关)
````bash
nohup bash -c '
set -euo pipefail

python -u /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_multihop_training_v2.1.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --train_batch_size 2048 \
  --num_workers 0 \
  --disable_pin_memory \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_merged_v2.1.pkl \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_multihop_training_v2.1.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_multihop_training_v2.1_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4

' > /mnt/n0/PathWeaver/docs/scripts/graph_gen/train_gen_subgraph.log 2>&1 &
````

output:
```
Answer recall: 0.9246
Graph  recall: 0.9827
None-sink recall: 0.9246
Answer + supported sink ratio: 0.8310
Sink relevance rank stats (merged across final sink counts):
  all_samples=164601 answer_sink_samples=164601 top-1=0.7888 top-2=0.9455 top-3=1.0000
[DONE] input=178027 output=164601 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_multihop_training_v2.1_dag.jsonl


Answer recall: 0.8776
Graph  recall: 0.9662
None-sink recall: 0.8776
Answer + supported sink ratio: 0.8523
Sink relevance rank stats (merged across final sink counts):
  all_samples=208 answer_sink_samples=208 top-1=0.7212 top-2=0.9375 top-3=1.0000
[DONE] input=237 output=208 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl
Answer recall: 0.8523
Graph  recall: 0.9662
None-sink recall: 0.8819
Answer + supported sink ratio: 0.8312
Sink relevance rank stats (merged across final sink counts):
  all_samples=237 answer_sink_samples=202 top-1=0.7277 top-2=0.9356 top-3=1.0000
[DONE] input=237 output=237 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa.jsonl

Answer recall: 0.8800
Graph  recall: 0.9933
None-sink recall: 0.8800
Answer + supported sink ratio: 0.8033
Sink relevance rank stats (merged across final sink counts):
  all_samples=264 answer_sink_samples=264 top-1=0.7955 top-2=0.9697 top-3=1.0000
[DONE] input=300 output=264 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
Answer recall: 0.8200
Graph  recall: 0.9933
None-sink recall: 0.8900
Answer + supported sink ratio: 0.7667
Sink relevance rank stats (merged across final sink counts):
  all_samples=300 answer_sink_samples=246 top-1=0.7967 top-2=0.9675 top-3=1.0000
[DONE] input=300 output=300 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl

Answer recall: 0.6333
Graph  recall: 0.8967
None-sink recall: 0.6333
Answer + supported sink ratio: 0.5667
Sink relevance rank stats (merged across final sink counts):
  all_samples=190 answer_sink_samples=190 top-1=0.7211 top-2=0.9091 top-3=1.0000
[DONE] input=300 output=190 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
Answer recall: 0.5833
Graph  recall: 0.8967
None-sink recall: 0.6900
Answer + supported sink ratio: 0.5467
Sink relevance rank stats (merged across final sink counts):
  all_samples=300 answer_sink_samples=175 top-1=0.7657 top-2=0.9302 top-3=1.0000
[DONE] input=300 output=300 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl

Answer recall: 0.9257
Graph  recall: 1.0000
None-sink recall: 0.9257
Answer + supported sink ratio: 0.6015
Sink relevance rank stats (merged across final sink counts):
  all_samples=748 answer_sink_samples=748 top-1=0.8997 top-2=0.9677 top-3=1.0000
[DONE] input=808 output=748 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl
Answer recall: 0.8911
Graph  recall: 1.0000
None-sink recall: 0.9282
Answer + supported sink ratio: 0.5941
Sink relevance rank stats (merged across final sink counts):
  all_samples=808 answer_sink_samples=720 top-1=0.9014 top-2=0.9692 top-3=1.0000
[DONE] input=808 output=808 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa.jsonl
```

#### 训练集+测试集准备

训练集：合并单跳+多跳数据集
````bash
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merge_tripled_datasets_lightweight.py \
  --dataset singlehop_pop=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/popqa.jsonl \
  --dataset singlehop_squad=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/squad_v4.jsonl \
  --dataset multihop=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_multihop_training_v2.1_dag.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl \
  --stats-output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.stats
````
生成embedding
训练集: /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl
测试集(多跳有aa/naa两版)：
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl
  - /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa.jsonl
````bash
nohup bash -c '
set -euo pipefail

FILES=(
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_hybrid_training_v2.1.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_aa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa.jsonl
)

for file in "${FILES[@]}"; do
  echo "[$(date -Iseconds)] start: $file"
  python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
    --model_name qwen3-embedding-0.6B \
    --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
    --dataset_type dag \
    --dataset_path "$file" \
    --batch_size 1024 \
    --progress
  echo "[$(date -Iseconds)] done:  $file"
done
' > /mnt/n0/PathWeaver/docs/scripts/embedding_v2.log 2>&1 &
````

#### 训练


aa版测试集
````bash
export CUDA_VISIBLE_DEVICES=2
mkdir -p experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa
nohup python experiments/train.py \
  --seed 1 --B 5 --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
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
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa \
  --save_period 200 --keep_top_k_ckpt 5 \
  > experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/training_log.txt 2>&1 &

````

naa版测试集
````bash
export CUDA_VISIBLE_DEVICES=3
mkdir -p experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_naa
nohup python experiments/train.py \
  --seed 1 --B 5 --lr 5e-4 --use_lr_decay --gradient_accm_step 20 \
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
  --test_data_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_naa.jsonl /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa.jsonl \
  --test_precomputed_embed_keys_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_naa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_naa_qwen-embedding-0.6B_embd_key.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_paths /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/popqa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_naa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_naa_qwen-embedding-0.6B_embd_value.npy /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 200 \
  --total_steps 8000 --N 9999999 \
  --model_save_dir experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_naa \
  --save_period 200 --keep_top_k_ckpt 5 \
  > experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_naa/training_log.txt 2>&1 &

````

#### 时延测试

Profile Online DAG Lantency
  - 创建专门的单样本测试路径(profile_online_latency开启)

总时间：
  - 实体检索时间：
  - DAG时间
  - Prefill
  - Decode

实体检索时间，看vector-rag: 0.0454
````bash
python3 /mnt/n0/PathWeaver/experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/mintqa_test_pruned256.jsonl \
  --queryset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/mintqa_test_pruned64_hop2.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --max-model-len 32768 \
  --index-path /mnt/n0/PathWeaver/experiments/vector_rag_index/mintqa_dev_bge_256
````

DAG时间: 0.042-0.02=0.022
````bash
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa_profile.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4 \
  --profile_online_latency
# avg_per_sample=0.042126s question_encode=0.020508s feature_build=0.016761s model_scoring=0.003895s dag_postprocess=0.000889s export=0.000072s
````

Prefill+Decode时间：0.3905
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset mintqa_pruned64_hop2_dag_naa.jsonl  \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_naa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800 \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_naa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 6800 \
  --t_step 8000 \
  --kb_scale_factor 4 
````

## MoreHopQA (❌️，精度太低)

- Triple Gen & Split
将手工验证和未验证的数据集合并成一个, 验证过的在前面
````bash
cd /mnt/n0/datasets/morehopqa/data
python merge.py
````
生成三元组：
"/mnt/n0/datasets/morehopqa/prepare_kg/total_samples_to_tripled_kg_v1.jsonl"

- Graph Gen
````bash
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --input /mnt/n0/datasets/morehopqa/prepare_kg/total_samples_to_tripled_kg_v1.jsonl \
  --output /mnt/n0/datasets/morehopqa/prepare_kg/total_samples_to_tripled_kg_v1_dag.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_merged_v1_joint.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --batch_size 1024 \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --answer_aware \
  --keep_score \
  --answerable_only \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
````

- 数据集后处理
划分训练集和验证集
````bash
cd /mnt/n0/datasets/morehopqa/prepare_kg
python 1.merge.py
python 1.1.supporting_extract.py samples_tripled_kg_v1_dag.jsonl --output samples_tripled_kg_v1_dag_fixed.jsonl
python 2.split.py

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/morehopqa/prepare_kg/train_v1.jsonl \
  --batch_size 1024
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/morehopqa/prepare_kg/dev_v1.jsonl \
  --batch_size 1024

````


### baseline
````bash
python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/morehopqa/prepare_kg/samples_tripled_kg_v1_dag_fixed.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --oracle-retrieval \
  --index-path ../../experiments/vector_rag_index/morehopqa_bge
# {'rouge1': 0.19091804029304035, 'rouge2': 0.0, 'rougeL': 0.1688238150738151, 'rougeLsum': 0.166213924963925, 'exact_match': 0.09, 'f1_overlap': 0.10786782661782662, 'faithfulness01': 0.22}

python3 ../../experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/morehopqa/prepare_kg/samples_tripled_kg_v1_dag.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 16 \
  --index-path ../../experiments/vector_rag_index/morehopqa_bge

````

### zore-shot
---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.1, 'rouge2': 0.0, 'rougeL': 0.10166666666666666, 'rougeLsum': 0.09999999999999998, 'exact_match': 0.06, 'f1_overlap': 0.065, 'faithfulness01': 0.06}
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/morehopqa/prepare_kg \
  --test_dataset dev_v1.jsonl \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_11000 \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_hop_qwen3_4B_v1.1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_11000_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/morehopqa/prepare_kg/dev_v1_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/morehopqa/prepare_kg/dev_v1_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 11000 \
  --t_step 12000 \
  --kb_scale_factor 4
````

### 小规模继续训练

### 混合训练





# Graph-RAG

## AutoSchemaKG Gen

````bash
source /mnt/n0/PathWeaver/envs/kblam-rag/bin/activate
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
````

转换数据集格式：
````bash
source /mnt/n0/PathWeaver/envs/kblam-rag/bin/activate
cd /mnt/n0/AutoSchemaKG

python atlas_rag/kg_construction/prepare_datasets.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_dataset.json \
  --dataset_type popqa \
  --output /mnt/n0/AutoSchemaKG/example/example_data/popqa_dataset.jsonl

python atlas_rag/kg_construction/prepare_datasets.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_dev.json \
  --dataset_type squad \
  --output /mnt/n0/AutoSchemaKG/example/example_data/squad_dev.jsonl

python /mnt/n0/AutoSchemaKG/atlas_rag/kg_construction/prepare_datasets.py \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64.jsonl \
  --dataset_type mintqa \
  --output /mnt/n0/AutoSchemaKG/example/example_data/mintqa_test_64_triple.jsonl

````

构图：
````bash
export CUDA_VISIBLE_DEVICES=1
nohup bash -c '
set -euo pipefail
python docs/scripts/1.create_kg_2wiki.py \
  --data-directory /mnt/n0/AutoSchemaKG/example/example_data \
  --data-name popqa_dataset.jsonl \
  --output-directory /mnt/n0/AutoSchemaKG/example/generated/popqa_dataset \
  --llm-endpoint http://127.0.0.1:8001/v1 \
  --llm-model qwen_72b \
  --batch-size-triple 256 \
  --max-workers 64

python docs/scripts/1.create_kg_2wiki.py \
  --data-directory /mnt/n0/AutoSchemaKG/example/example_data \
  --data-name squad_dev.jsonl \
  --output-directory /mnt/n0/AutoSchemaKG/example/generated/squad_dev \
  --llm-endpoint http://127.0.0.1:8001/v1 \
  --llm-model qwen_72b \
  --batch-size-triple 256 \
  --max-workers 64
' >> /mnt/n0/AutoSchemaKG/example/example_data/generate_pop_squad.log  2>&1 &


nohup python /mnt/n0/AutoSchemaKG/docs/scripts/1.create_kg_2wiki.py \
  --data-directory /mnt/n0/AutoSchemaKG/example/example_data \
  --data-name mintqa_test_64_triple.jsonl \
  --output-directory /mnt/n0/AutoSchemaKG/example/generated/mintqa_test_64_triple \
  --llm-endpoint http://127.0.0.1:8001/v1 \
  --llm-model qwen_72b \
  --batch-size-triple 256 \
  --max-workers 64 >> /mnt/n0/AutoSchemaKG/example/example_data/generate_mintqa_test_64_triple.log  2>&1 &
````

## Evaluate

启动backend模型服务：
````bash
source /mnt/n0/uv_envs/kblam-rag/bin/activate
export CUDA_VISIBLE_DEVICES=2
nohup python -m vllm.entrypoints.openai.api_server \
  --model /mnt/n0/models/qwen3-4B-Instruct/ \
  --served-model-name qwen3_4B \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 > /mnt/n0/PathWeaver/experiments/qwen3-4B-Instruct.log 2>&1 &
````

- pop
  {'rouge1': 0.4080604893472538, 'rouge2': 0.013523809523809525, 'rougeL': 0.4087815126050418, 'rougeLsum': 0.40904996392496373, 'exact_match': 0.19, 'f1_overlap': 0.40841925558102027, 'faithfulness01': 0.71}
  Average retrieval time: 2.9210 s
  Average generation time: 2.5791 s
  ````bash
  source /mnt/n0/uv_envs/kblam-rag/bin/activate
  export CUDA_VISIBLE_DEVICES=3

  nohup python /mnt/n0/AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
    --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/popqa_queryset.json \
    --encoder-model /mnt/n0/models/bge-en-v1.5/ \
    --kg-path /mnt/n0/AutoSchemaKG/example/generated/popqa_dataset/ \
    --llm-endpoint http://127.0.0.1:8001/v1 \
    --llm-model qwen3_4B \
    --test-samples 100 \
    --topN 16 > /mnt/n0/PathWeaver/experiments/overall_graph_rag_popqa_qwen3_4B_bge.log 2>&1 &
  ````

- squad

  =====Metrics=====
  {'rouge1': 0.7378571238702818, 'rouge2': 0.5179226190476189, 'rougeL': 0.7364400584795322, 'rougeLsum': 0.7396536416799575, 'exact_match': 0.67, 'f1_overlap': 0.7370502079619726, 'faithfulness01': 0.72}
  Average retrieval time: 1.9344 s
  Average generation time: 3.2991 s

  ````bash
  source /mnt/n0/uv_envs/kblam-rag/bin/activate
  export CUDA_VISIBLE_DEVICES=3

nohup python /mnt/n0/AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/squad_dev.json \
  --encoder-model /mnt/n0/models/bge-en-v1.5/ \
  --kg-path /mnt/n0/AutoSchemaKG/example/generated/squad_dev/ \
  --llm-endpoint http://127.0.0.1:8001/v1 \
  --llm-model qwen3_4B \
  --test-samples 100 \
  --seed 2 \
  --topN 16 > /mnt/n0/PathWeaver/experiments/overall_graph_rag_squad_qwen3_4B_bge.log 2>&1 &
  ````

- 2wiki
  {'rouge1': 0.4753593073593073, 'rouge2': 0.3609919908466819, 'rougeL': 0.4738233766233767, 'rougeLsum': 0.47366277056277056, 'exact_match': 0.42, 'f1_overlap': 0.4714424242424242, 'faithfulness01': 0.45}
  Average retrieval time: 2.7416 s
  Average generation time: 3.3045 s
  ````bash
  source /mnt/n0/uv_envs/kblam-rag/bin/activate
  export CUDA_VISIBLE_DEVICES=3

  nohup python  /mnt/n0/AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
    --kg-path /mnt/n0/AutoSchemaKG/example/generated/2wiki_dev/ \
    --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
    --encoder-model /mnt/n0/models/bge-en-v1.5/ \
    --llm-model qwen3_4B \
    --test-samples 100 \
    --llm-endpoint 'http://127.0.0.1:8001/v1' \
    --topN 16 > /mnt/n0/PathWeaver/experiments/overall_graph_rag_2wiki_qwen3_4B_bge.log 2>&1 &
  ````

- hotpot
  Average retrieval time: 6.4916 s
  Average generation time: 3.2367 s
  {'rouge1': 0.5784565580618211, 'rouge2': 0.3277408963585434, 'rougeL': 0.5783909774436089, 'rougeLsum': 0.5776271929824561, 'exact_match': 0.42, 'f1_overlap': 0.5776152882205514, 'faithfulness01': 0.5}

  ````bash
  source /mnt/n0/uv_envs/kblam-rag/bin/activate
  export CUDA_VISIBLE_DEVICES=3

  nohup python  /mnt/n0/AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
    --kg-path /mnt/n0/AutoSchemaKG/example/generated/hotpot_dev/ \
    --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_clean/hotpot_dev_v1.json \
    --encoder-model /mnt/n0/models/bge-en-v1.5/ \
    --llm-model qwen3_4B \
    --test-samples 100 \
    --llm-endpoint 'http://127.0.0.1:8001/v1' \
    --topN 16 > /mnt/n0/PathWeaver/experiments/overall_graph_rag_hotpot_qwen3_4B_bge.log 2>&1 &
  ````

- musique

  Average retrieval time: 3.2884 s
  Average generation time: 3.7312 s
  {'rouge1': 0.26632991220203495, 'rouge2': 0.1814308608058608, 'rougeL': 0.2611060089384897, 'rougeLsum': 0.2662143709852015, 'exact_match': 0.17, 'f1_overlap': 0.2660563996210785, 'faithfulness01': 0.27}

  ````bash
  source /mnt/n0/uv_envs/kblam-rag/bin/activate
  export CUDA_VISIBLE_DEVICES=3

  nohup python  /mnt/n0/AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
    --kg-path /mnt/n0/AutoSchemaKG/example/generated/musique_dev/ \
    --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_clean/musique_dev_answerable.jsonl \
    --encoder-model /mnt/n0/models/bge-en-v1.5/ \
    --llm-model qwen3_4B \
    --test-samples 100 \
    --llm-endpoint 'http://127.0.0.1:8001/v1' \
    --topN 16 > /mnt/n0/PathWeaver/experiments/overall_graph_rag_musique_qwen3_4B_bge.log 2>&1 &
  ````
- mintqa
  Average retrieval time: 1.0019 s
  Average generation time: 2.4292 s
  {'rouge1': 0.46018518518518514, 'rouge2': 0.24, 'rougeL': 0.4606944444444445, 'rougeLsum': 0.4637499999999999, 'exact_match': 0.41, 'f1_overlap': 0.4604629629629629, 'faithfulness01': 0.41}
````bash
source /mnt/n0/uv_envs/kblam-rag/bin/activate
export CUDA_VISIBLE_DEVICES=3

nohup python  /mnt/n0/AutoSchemaKG/docs/scripts/2.kg_benchmark.py \
  --kg-path /mnt/n0/AutoSchemaKG/example/generated/mintqa_test_64_triple/ \
  --dataset-path /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2.jsonl \
  --encoder-model /mnt/n0/models/bge-en-v1.5/ \
  --llm-model qwen3_4B \
  --test-samples 100 \
  --llm-endpoint 'http://127.0.0.1:8001/v1' \
  --topN 16 > /mnt/n0/PathWeaver/experiments/overall_graph_rag_mintqa_qwen3_4B_bge.log 2>&1 &
````


# KBLaM
flat dag：将所有三元组展开，不做额外的检索
````bash
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode baseline \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_2_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_base.jsonl \
  --limit 1

nohup bash -c '
set -euo pipefail

FILES=(
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa.jsonl
)

for file in "${FILES[@]}"; do
  file_base="${file%.jsonl}_base"
  echo "[$(date -Iseconds)] start: $file -> ${file_base}.jsonl"
  python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
    --mode baseline \
    --input "$file" \
    --output "${file_base}.jsonl"
  echo "[$(date -Iseconds)] done:  $file"
done
' > /mnt/n0/PathWeaver/docs/scripts/graph_gen/kblam_base.log 2>&1 &

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode baseline \
  --input /mnt/n0/datasets/multi-hop/MintQA/test_vector_rag_pruned_64_hop_3_tripled.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop3_dag_base.jsonl

````

create embedding：
````bash
nohup bash -c '
set -euo pipefail

FILES=(
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_naa_base.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/hotpot_dev_tripled_v5-qwen3.5-27B_dag_aa_base.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/musique_dev_tripled_v5-qwen3.5-27B_dag_aa_base.jsonl
  /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop2_dag_naa_base.jsonl
)

for file in "${FILES[@]}"; do
  echo "[$(date -Iseconds)] start: $file"
  python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
    --model_name qwen3-embedding-0.6B \
    --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
    --dataset_type dag \
    --dataset_path "$file" \
    --batch_size 1024 \
    --progress
  echo "[$(date -Iseconds)] done:  $file"
done
' > /mnt/n0/PathWeaver/docs/scripts/embedding_v2.log 2>&1 &

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/mintqa_pruned64_hop3_dag_base.jsonl \
  --batch_size 1024 \
  --progress
````

# QwenLong+MultiGPUTraining

accelerate FSDP: 多卡训练，把参数、梯度和优化器状态分片，避免单卡爆显存


````bash
cd /mnt/n0/PathWeaver
PYTHONPATH=/mnt/n0/PathWeaver/src:/mnt/n0/PathWeaver/experiments \
accelerate launch experiments/train_qwenlong_dag.py \
  --dataset_type dag \
  --llm_type qwen_moe \
  --hf_model_spec Tongyi-Zhiwen/QwenLong-L1.5-30B-A3B \
  --train_data_path /path/to/train.jsonl \
  --train_precomputed_embed_keys_path /path/to/key.npy \
  --train_precomputed_embed_values_path /path/to/value.npy \
  --base_embeder_path /path/to/base_embedder \
  --B 1 \
  --gradient_accm_step 8 \
  --total_steps 10 \
  --save_period 10 \
  --model_save_dir /path/to/output


````