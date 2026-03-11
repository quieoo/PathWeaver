# 开启DS服务
````bash

# 启动镜像
./start_container.sh 0465ed35a85c triple-0
# 或者如果已经创建过容器，直接启动
docker start triple-0

# 进入容器
docker exec -it triple-0 bash

# 测试NPU是否可用
python3 -c "import torch; x=torch.zeros((1024,1024), device='npu'); print('ok', x.shape)"

#运行DeepSeek模型服务
# 关闭MTP
cd /home/zhu/triple_gen/omniinfer/tools/scripts/
nohup ./run_single.sh > ds_serve.log 2>&1 &

# 开启prefix_caching, 调小：num_seqs, block_size, gpu_utilization, max_num_batched_tokens
# nohup ./run_single_v2.sh > ds_serve.log 2>&1 &


# 测试
curl http://127.0.0.1:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "temperature": 0.7
  }'

````

# AutoSchemaKG
````bash
conda activate autoschemakg


````


# 三元组生成

1. 下载数据集
2. 预处理
````bash
conda activate triple_gen
python squad_prepare.py

# 修改脚本中的输入输出路径
# 将相同上下文的问答对合并到一起

````

3. 生成三元组
````bash
docker start triple-0

#运行DeepSeek模型服务
docker exec -it triple-0 bash
cd /home/zhu/triple_gen/omniinfer/tools/scripts/
nohup ./run_single.sh > ds_serve.log 2>&1 &
# 83133


# 假设模型服务已经在"http://localhost:7000"运行
nohup python squad_gen_kv.py --input /docker/datasets/squad/plain_text/train_merged.json --output /docker/datasets/squad/plain_text/train_merged_kv_v3.json --batch-size 64 >> gen.log 2>&1 &

nohup python squad_gen_triples.py --input /docker/datasets/squad/plain_text/train_merged.json --output /docker/datasets/squad/plain_text/train_merged_kv_v4.json --batch-size 64 >> gen_v2.log 2>&1 &

nohup python squad_gen_kv.py --input /docker/datasets/squad/plain_text/validation_merged.json --output /docker/datasets/squad/plain_text/validation_merged_kv_v3.json --batch-size 64 >> gen.log 2>&1 &

# 生成2wiki
nohup python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/merge_data/merged_train.json --output /docker/datasets/2wiki_hotpot_musique/train_datasets.json --type 2wiki_hotpot --specific_type 2wiki --batch-size 64 --template_version 4 >> gen_2wiki_v3.log 2>&1 &

python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/merge_data/merged_train.json --output /docker/datasets/2wiki_hotpot_musique/hotpot_train_datasets.json --type 2wiki_hotpot --specific_type hotpot --batch-size 64 --template_version 3

python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/merge_data/merged_dev.json --output /docker/datasets/2wiki_hotpot_musique/hotpot_test_datasets.json --type 2wiki_hotpot --specific_type hotpot --batch-size 64 --template_version 3


# hotpot-2hop
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/hotpot_2hop_train.json --output ./tmp_hotpot.json --type 2wiki_hotpot --specific_type hotpot --batch-size 64 --template_version 3 --limit 3

python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/hotpot_2hop_train.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/hotpot_2hop/train_datasets.json --type 2wiki_hotpot --specific_type hotpot --batch-size 32 --template_version 3

python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/hotpot_2hop_test.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/hotpot_2hop/test_datasets.json --type 2wiki_hotpot --specific_type hotpot --batch-size 32 --template_version 3


# musique_2hop

## 预处理
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets_prepare.json --output ./tmp_hotpot.json --type musique_preprocess --specific_type musique --batch-size 64 --template_version 5 --limit 3

python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets_prepare.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets.json --type musique_preprocess --specific_type musique --batch-size 64 --template_version 5 
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/test_datasets_prepare.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/test_datasets.json --type musique_preprocess --specific_type musique --batch-size 64 --template_version 5 

## 生成三元组
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets.json --output ./tmp_musique.json --type musique --specific_type musique --batch-size 64 --template_version 3 --limit 64
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets_triples.json --type musique --specific_type musique --batch-size 64 --template_version 3
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/test_datasets.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/test_datasets_triples.json --type musique --specific_type musique --batch-size 64 --template_version 3

## batch_size 64似乎性能会崩，准确来说达到54就会崩
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/train_datasets_triples.json --type musique --specific_type musique --batch-size 48 --template_version 3
python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/test_datasets.json --output /docker/datasets/2wiki_hotpot_musique/filtered_data/musique_2hop/test_datasets_triples.json --type musique --specific_type musique --batch-size 48 --template_version 3

````


# 三元组生成(v2)

````bash
python gen_triples_v2.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json --output ./tmp_hotpot.json  --batch-size 64 --limit 3  --template-version fidelity

# 必须要足够长的上下文，否则会截断三元组
python gen_triples_v2.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json --output ./tmp_hotpot.json  --batch-size 64 --limit 1  --template-version efficiency --max-tokens 4096


python gen_triples_2wiki_v2.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json --output ./tmp_hotpot.json  --batch-size 64 --limit 1

# 打太满精度出问题
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json --output ./tmp_hotpot.json  --batch-size 64 --max-tokens 512 --limit 64 --inflight-batches 2
# 删除流水线方法
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json --output ./tmp_hotpot.json  --batch-size 64 --max-tokens 512 --limit 128


````
## 2wiki
````bash
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/2wiki_train_2hop.jsonl  --batch-size 64 --max-tokens 512
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_dev_2hop.json --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/2wiki_dev_2hop.jsonl  --batch-size 64 --max-tokens 512
````
## hotpot

````bash

#测试
python gen_triples_2wiki_v2.1.py --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/hotpot_dev.json --output ./tmp_hotpot.jsonl  --batch-size 64 --max-tokens 512 --dataset hotpot --supporting-only --limit 3

python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/hotpot_train.json --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/hotpot_train.jsonl  --batch-size 64 --max-tokens 512 --dataset hotpot --supporting-only
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/hotpot_dev_2hop.json --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/hotpot_dev.jsonl  --batch-size 64 --max-tokens 512 --dataset hotpot --supporting-only
````

## musique

````bash
cd /home/zhu/triple_gen/scripts/
#测试
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/musique_train.jsonl --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/musique_train.jsonl  --batch-size 64 --max-tokens 512 --dataset musique --supporting-only --limit 3

python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/musique_train.jsonl --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/musique_train.jsonl  --batch-size 64 --max-tokens 512 --dataset musique --supporting-only
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/musique_dev.json --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/musique_dev.jsonl  --batch-size 64 --max-tokens 512 --dataset musique --supporting-only
````


# V3 (build_graph)

````bash
cd /home/zhu/triple_gen/scripts/

python build_knowledge_graph.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/hotpot_dev.json \
  --output ./tmp_hotpot.jsonl \
  --dataset hotpot \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --batch-size 64 \
  --max-tokens 512 \
  --retries 1 \
  --limit 3 \
  --supporting-only --hotpot-bridge-only

python build_knowledge_graph.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/hotpot_train.json \
  --output ./tmp_hotpot.jsonl \
  --dataset hotpot \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --batch-size 64 \
  --max-tokens 512 \
  --retries 1 \
  --limit 3 \
  --random-seed 42 \
  --supporting-only --hotpot-bridge-only

python build_knowledge_graph.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json \
  --output ./tmp_hotpot.jsonl \
  --dataset 2wiki \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --batch-size 64 \
  --max-tokens 512 \
  --retries 1 \
  --limit 3 \
  --random-seed 42 \
  --supporting-only --compositional-only

python build_knowledge_graph.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --output ./tmp_hotpot.jsonl \
  --dataset 2wiki \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --batch-size 64 \
  --max-tokens 512 \
  --retries 1 \
  --limit 3 \
  --random-seed 42 \
  --supporting-only --compositional-only


python build_knowledge_graph.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/musique_train.jsonl \
  --output ./tmp_hotpot.jsonl \
  --dataset musique \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --batch-size 64 \
  --max-tokens 512 \
  --retries 1 \
  --limit 3 \
  --random-seed 42 \
  --supporting-only

python build_knowledge_graph.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/musique_dev.jsonl \
  --output ./tmp_hotpot.jsonl \
  --dataset musique \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --batch-size 64 \
  --max-tokens 512 \
  --retries 1 \
  --limit 3 \
  --random-seed 42 \
  --supporting-only
````


# v3.1 concurrent requests

````bash
python build_knowledge_graph_v2.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/hotpot_train.json \
  --output /home/zhu/datasets/2wiki_hotpot_musique/merged_data/dag_kv/hotpot_train_tmp.json \
  --dataset hotpot \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --concurrent-requests 4 \
  --num-mini-batch 4 \
  --batch-size 64 \
  --max-tokens 1024 \
  --retries 3 \
  --supporting-only --hotpot-bridge-only

# 测试 batch_size可以开到128， concurrent_requests同样开到128. 打满vllm，KV Cache占用大概50%
nohup python build_knowledge_graph_v2.py \
  --input /home/zhu/datasets/2wiki_hotpot_musique/merged_data/source_data/hotpot_train.json \
  --output /home/zhu/datasets/2wiki_hotpot_musique/merged_data/dag_kv/hotpot_train.json \
  --dataset hotpot \
  --endpoint http://127.0.0.1:7000/v1/completions \
  --model deepseek \
  --concurrent-requests 128 \
  --batch-size 128 \
  --max-tokens 1024 \
  --retries 3 \
  --supporting-only --hotpot-bridge-only >> build_knowledge_graph_hotpot_train_2.log 2>&1 &
````


# build_knowledge_graph with qwen-72B

````bash
conda activate vllm-13
export CUDA_VISIBLE_DEVICES=0,1
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/sdu/zhu/models/qwen2.5-72B-4bit \
  --served-model-name qwen_72b \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 > qwen_72b.log 2>&1 &

python build_knowledge_graph_v3.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev.json \
  --output ./tmp/2wiki_dev.jsonl \
  --dataset 2wiki \
  --endpoint http://127.0.0.1:8001/v1/completions \
  --model qwen_72b \
  --concurrent-requests 128 \
  --batch-size 128 \
  --max-tokens 1024 \
  --limit 64 \
  --retries 3 \
  --supporting-only --compositional-only --answer-aware

nohup python build_knowledge_graph_v3.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_train_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/2wiki_train_2hop.jsonl \
  --dataset 2wiki \
  --endpoint http://127.0.0.1:8001/v1/completions \
  --model qwen_72b \
  --concurrent-requests 128 \
  --batch-size 128 \
  --max-tokens 1024 \
  --retries 3 \
  --supporting-only --compositional-only --answer-aware >> build_knowledge_graph_v3.log 2>&1 &

````