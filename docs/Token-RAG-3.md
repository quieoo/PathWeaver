
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
## Hotpot
````bash
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

````

## Musique

````bash
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

````



## AT-From-Base
不在QA数据集上续训练（已经形成了对gold path的依赖），直接从base模型上开始训练，随着训练可以逐渐增加negative path的数量，但是全程使用question和negative path

````bash
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


````
### 提取silver path
silver path是指negative path中指向正确答案的路径
确保在每个阶段中都有silver path
  - extract_silver_path.py, 把数据集中的silver path移动到第一位
  - 在kb_retriever.py中，从头开始读入negative path之后通过random.shuffle(path_triple_base)打乱顺序

提取silver path
````bash
python /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/extract_siver_path.py

# 在脚本里直接改文件名
````

````
Extract silver path from AT2QA_2wiki_test_2hop_compositional_gold.json to ATFB_2wiki_test_2hop_compositional_silver.json
Total sample: 100
Exact match ratio: 0.44
Contain ratio: 0.8

Extract silver path from AT2QA_2wiki_train_2hop_compositional_gold.json to ATFB_2wiki_train_2hop_compositional_silver.json
Total sample: 74988
Exact match ratio: 0.5989091587987412
Contain ratio: 0.8886488504827439
````

重新计算embedding
````bash
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


````

### 训练坍塌问题

````
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
````

目前来看似乎是数据集切换到silver之后出的问题。
  - 数据集变化
  - batch size 5 -> 10

### B=5

````bash
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
````
### keep_top_k_ckpt=10

````bash
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
````

#### top-k ckpt的测试
````bash

# 批量测试

nohup python scripts/batch_inference.py > batch_inference.log 2>&1 &

````
### silver+random; 2,4 path

````bash
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

````

### retrieve; 2-4-8 path

````bash
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
````

### bge-embedding

````bash
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

  
````
NOTE：不确定是过程出的错还是bge-embedidng本身在这类任务上性能更差，导致训练出来的模型精度降低大约10%.

### 复现qwen-embedding
同时做一个小修改:
- 训练过程中的验证集使用和推理时一模一样的配置：2条、无silver
````bash
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

````

- 训练路径：8或16条，1 silver + n random
- 验证路径：和训练一样
````bash

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
````


## 修复训练过程中不稳定的崩溃问题
在训练PathWeaver模型过程中出现精度崩溃的现象，在训练一定步数后loss突然大增，之后的训练虽然可以使得loss继续降低，但是模型的推理精度会受到不可恢复的影响。
这种现象并不总是会发生，但是发现随着训练的批量大小增加发生概率会变大。

### 问题复现
跑两组，不同的B值
````bash
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

````
### 解决1：训练集过滤
filter_valid_sample.py: 将路径为空，或者不包含正确答案的样本过滤掉
````
Original samples: 74988
Valid samples: 66752
Saved valid samples to ATFB_2wiki_train_2hop_compositional_silver_filtered.json
````
重新生成embedding:

````bash
python scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_compositional_silver_filtered.json \
  --batch_size 1024
````

重新训练：
````bash
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

````
效果：似乎稳定了很多，Training loss不再波动


进一步测试：
 - 修改路径选择，训练时候：silver+random -> silver+sequencial, 验证时候：2条逐渐增加到8条 -> 固定2条(固定current_step=0)
 - keep_top_k_ckpt=10

````bash
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

````
### 解决2： 降低batch size
````bash
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

````
结论：减少batch size仍旧会出现崩溃现象



### 解决3：不要分阶段，直接固定路径数量

修改kb_retriever.py中的get_triple_ids_ATFB：
````python
    step_ratio=[1.0]
    stage_num=len(step_ratio)
    negative_path_num=[max_kb_paths]
````

````bash
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
````

结论：这不行，模型完全不收敛



## Hotpot
一些新的挑战：
 - Hotpot上答案可能出现在head、relation位置，而不是一定在tail
 - 答案和上下文可能存在相同意思，但是拼写不同的情况 （数据集过滤时候需要注意）


### 原版AT2QA_v2

````bash
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
````
输出：
````
100条
Average pick time: 0.1253s
Graph recall: 0.4900
Answer recall: 0.4500

1000条
Average pick time: 0.1141s
Graph recall: 0.5470
Answer recall: 0.4930
````
### 原版 + bge-embedding
````bash
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
````
输出：
````
Average pick time: 0.0693s
Graph recall: 0.4900
Answer recall: 0.4200
````
结论：bge更快，但是性能比qwen-embedding-0.6B略微慢点

### 开启verbose debug

````bash
export CUDA_VISIBLE_DEVICES=2
nohup python scripts/AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/hotpot_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_hotpot_paths_train_debug.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 100 \
  --k 16 > at_hotpot.log 2>&1 &
````




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
````python
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
````

## 图谱检索


````bash
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


````
Answer recall: 0.269
增加entity normalization： 0.278

增加字符级的相似度作为得分计算

````bash
nohup python scripts/DAG_KV_Retriever_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/dag_hotpot_dev_tmp.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --limit 1000 \
  --max_hops 3 --beam_width 3 --max_edges 20 --max_nodes 25 --seed_top_m 3 --max_sinks 3 --use_seededge_beam \
  --verbose  >> dag_hotpot.log 2>&1 &

````

直接清理sink
````bash

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
````

回到原版
````bash
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

````


### Related work-1: PropRAG

### Related work-2: HippoRAG
````bash
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
````

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

````bash
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

````
结果：
  - answer recall: 0.24 ~ 0.4


### 现有方法 HippoRAG
在 DAG_KV_HippoRAG.py 中，我们将原始 HippoRAG/HippoRAG2 面向文本证据的检索流程适配到结构化三元组图场景：首先将问题与图中实体节点进行语义匹配与显式提及匹配，选取得分最高的实体作为 query seeds；随后不再采用局部 beam 搜索，而是按照 HippoRAG 的核心思想，在图上执行以这些 seeds 为个性化重启分布的 Personalized PageRank（PPR）传播，从而获得与问题相关的全局实体重要性分数。对于 HippoRAG 版本，PPR 直接在实体图上进行；对于 HippoRAG2 版本，则进一步将每条 KV 边视为一个 memory node，构建“实体—记忆”二部图，在实体节点与 KV 证据节点之间联合传播，使传播结果能够同时刻画实体相关性与证据相关性。最后，根据实体 PPR 分数及 memory node 分数对 KV 边进行排序，并在保留连通性的前提下截取得到规模受限的子图，再通过 DAG 化与 sink 数量约束，使答案更倾向于落在子图末梢。这样，HippoRAG/HippoRAG2 的“query concept linking—graph propagation—evidence readout”框架就被自然迁移到了本文的实体—属性 DAG 检索任务中。

````bash

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

````
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

````bash
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
````
结论：
将 G-Retriever 的 PCST 子图检索思想迁移到 DAG-KV 图后，可以稳定保持较高的 graph recall（约 0.96），说明 PCST 风格的全局子图选择能够有效检索到答案相关区域。然而，answer recall 仅在 0.40–0.47 间小幅波动，表明现有 PCST 检索虽然能够找回答案相关子图，但缺乏将答案节点进一步组织为 DAG 末梢的结构归纳偏置。因此，仅依赖 PCST 风格的相关子图选择不足以满足 DAG-KV 的 answer-terminalization 需求，还需进一步设计面向末梢答案节点的定向与剪枝机制。



### 现有方法-2 UNIQORN：Group Steiner Tree
UNIQORN 是一种面向知识图谱与文本联合问答的统一推理框架，其核心思想是在问题相关证据构建的上下文图（context graph）上，通过问题线索（question cues）生成若干 anchor groups，并利用 Group Steiner Tree (GST) 在图中寻找能够同时连接这些锚点的最小子图，从而得到支持答案推理的证据结构。该机制能够在复杂图结构中自动发现连接多跳证据的关键路径，因此与本文需要从实体–属性图中检索最小推理子图的任务具有天然的契合性。

在本工作中，我们将已抽取的三元组构建为实体–属性图，并将其作为 UNIQORN 的上下文图输入。具体而言，我们首先根据问题文本与节点标签之间的语义相似度生成若干锚点集合，并将其组织为 anchor groups；随后在图上执行 Group Steiner Tree 搜索以获得若干候选证据子图，并根据节点在多棵 GST 中的出现频次进行排序。最终，我们将得到的无向子图定向为一个最小 DAG，使推理路径由问题相关实体逐步收敛至潜在答案节点，从而使答案尽可能位于 DAG 的末梢节点。通过这种方式，我们能够在保持 UNIQORN 推理思想的同时，使其适配本文的 DAG 子图检索任务。

````bash
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

````

answer recall: 0.35~0.4

### 现有方法-3 SubgraphRAG：all-at-once 子图检索

我们选择 SubgraphRAG 作为 DAG-KV 检索任务的重要对比方法，主要因为它与本文问题在检索目标和结构形式上具有较高一致性：本文需要从由实体节点、属性节点及定向 KV 边构成的大图中，直接检索出一个规模受控、结构紧凑且尽可能包含答案证据的子图，而 SubgraphRAG 的核心思想正是对图中候选三元组进行并行打分，并在此基础上一次性选出与问题最相关的子图，避免了传统多跳逐步扩展方法中容易出现的路径误差累积与早期决策偏置问题。因此，本文将 SubgraphRAG 适配到 DAG-KV 场景中：首先将每个三元组对应的两条 KV 边视为有向图中的候选检索单元；然后结合问题与 key、value、relation 语义相似度，以及从问题锚点实体出发的方向性结构特征，对候选边进行统一评分；接着选取得分最高的一组边并进行连通性补全与环路消解，构造满足节点数和边数约束的最小化 DAG 子图；最后以答案是否出现在 DAG 末梢节点对应的 value 中来评估其检索效果。这样的适配既较好保留了 SubgraphRAG“并行子图检索”的方法优势，也使其能够自然服务于本文“答案尽可能落在 DAG 末梢”的任务目标。

````bash
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
````

直接硬推：answer recall: 0.21~0.42

实现2：trainable MLP scorer

trainable MLP scorer：先为每条候选 KV edge 构造 question / src / relation / dst / key / value 的语义表示，再拼上 topic entity indicator 和 DDE 风格方向结构特征，最后用 MLP 学一个 relevance score

先训练，再推理
````bash
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

````
answer recall: 0.61 ~ 0.73

mlp模型的迁移能力，Hotpot上训练，2wiki上推理
````bash
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

````


### 现有方法最优

````bash
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

````
Answer recall: 0.7000
Graph  recall: 0.9600
None-sink recall: 0.8800

#### parameter-free最优
````bash
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
````
Answer recall: 0.4500
Graph  recall: 0.9500
None-sink recall: 0.6900


### Next: Terminal-Aware DAG Pruning

首先确定：答案未成为 sink 的样本里，有多少是因为“答案节点还有出边”导致的。
````
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

````
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



````bash
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
````
结果：
Answer recall: 0.6900
Graph  recall: 0.9600
None-sink recall: 0.8600

纯后处理、又不依赖 answer 的 stop-expansion 偏置，已经开始伤到 coverage 了

### 增强hard negatives

把“答案节点的出边”强行加入负样本池，让模型学会“到答案后不要继续扩展”

重新collect_training_examples，再训练。
````bash
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
````

训练参数 hard_neg_ratio:
 - 0.5: Answer recall: 0.6900 Graph  recall: 0.9600 None-sink recall: 0.8100
 - 0.25: Answer recall: 0.7400 Graph  recall: 0.9600 None-sink recall: 0.8800

扫参数：
````bash
cd scripts/graph_gen
nohup ./run_sweep_subgrapn_train_infer_v3.sh > sweep_train_infer_v3.log  2>&1 &
cd ..
````




### 增加end_node model

把 edge scorer 改成 node-ending aware 的两阶段打分。

````bash
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

````
Answer recall: 0.7800
Graph  recall: 0.9600
None-sink recall: 0.8600

### 共享编码器-联合训练


````bash
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

````

Answer recall: 0.8200
Graph  recall: 0.9600
None-sink recall: 0.8800

#### 微调参数



尝试增加 Node 任务权重
--joint_lambda 0.5    # 或 0.6
--patience 15         # 给更多收敛时间

````bash
python scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.py \
  --mode train \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5_joint.pt \
  --st_model /home/sdu/zhu/models/bge-en-v1.5/ \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 6 \
  --dde_hops 3 \
  --epochs 1000 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v4.pkl \
  --hidden_dim 768 \
  --patience 15 \
  --joint_training \
  --joint_lambda 0.5 \
  --end_alpha 0.50 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3


````