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

# v4

````bash
export OPENAI_BASE_URL=https://api.ofox.ai/v1
export OPENAI_API_KEY=sk-of-vKwFSPMrqYFFOrsrYNjvOgdqTgTidxmRocLlpBQurvCDoqIqaLJMtzxPcVxfUyAF

# OfoxAI API endpoint. Get your API Key at https://app.ofox.ai
curl https://api.ofox.ai/v1/chat/completions \
  -H "Authorization: Bearer sk-of-vKwFSPMrqYFFOrsrYNjvOgdqTgTidxmRocLlpBQurvCDoqIqaLJMtzxPcVxfUyAF" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bailian/qwen3-235b-a22b:free",
    "messages": [
      {"role": "user", "content": "What is the meaning of life?"}
    ]
  }'


mkdir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_v4_cached
python build_knowledge_graph_v4.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_tripled_v4.3.jsonl \
  --model gpt-5.4 \
  --concurrency 64 \
  --skip-comparison \
  --resume --limit 200

python build_knowledge_graph_v4.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_tripled_v4.3.jsonl \
  --model openai/gpt-5.4-mini \
  --concurrency 64 \
  --skip-comparison \
  --resume --limit 200


# 本地模型
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

python build_knowledge_graph_v4.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_tripled_v4.3.jsonl \
  --api-base http://127.0.0.1:8001/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model qwen_72b \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --limit 4


````

# deepseek-v3的替代

- Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen/Qwen3.5-27B

使用8u4090
修改v4.1： 
- 加入answer-aware参数，有一个专门的Prompt要求根据答案和问题生成具有完整证据链的三元组
- 支持Musique数据集（没有支撑段落的样本直接跳过）
- 统一输出格式：
  ````
  "context": [
    {
      "title": "...",
      "sentences": ["...", "..."],
      "triple_list": [...]
    }
  ]
  ````


结论：
- A3B的输出质量太差
- 3.5-27B质量挺好但是输出速度太慢，关闭Thinking之后好一点（2分钟）

````bash

conda activate triple
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 65536  \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}'  \
  --trust-remote-code > Qwen3.5-27B.log 2>&1 &

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen2.5-72B \
  --served-model-name Qwen2.5-72B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code > Qwen2.5-72B.log 2>&1 &

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 vllm serve models/qwen2.5-72B \
  --served-model-name Qwen2.5-72B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code

curl http://10.102.34.67:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ],
    "temperature": 0.7
  }'


python build_knowledge_graph_v4.1.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev_tripled_build_graph_v4.1_qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --overwrite \
  --limit 4


python build_knowledge_graph_v4.1.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --answer-aware \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.1-qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --limit 2

python build_knowledge_graph_v4.2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --answer-aware \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.2-qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --limit 1 --verbose --seed 42


python build_knowledge_graph_v4.3.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.3-qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --answer-aware \
  --limit 1 --verbose --seed 42

````

## 多阶段生成（v4.4 ⭐）
更弱的模型，更复杂的数据集
4阶段：
- 1. 尽可能全的抽取实体和三元组
- 2. 根据问题和答案修正（answer-aware）
- 3. 抽取正向KV
- 4. 抽取反向KV

````bash
python build_knowledge_graph_v4.4.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.4-qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --answer-aware \
  --limit 1 --verbose --seed 42

python build_knowledge_graph_v4.4.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.4-qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --answer-aware \
  --limit 1 --verbose --seed 42


python build_knowledge_graph_v4.4.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_dev.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.4-qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 8 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --answer-aware \
  --limit 1 --verbose --seed 42

# production

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 16384  \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 2048 \
  --disable-log-stats \
  --kv-cache-dtype auto \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}'  \
  --trust-remote-code > Qwen3.5-27B.log 2>&1 &

nohup python build_knowledge_graph_v4.4.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.4-qwen3.5_27B.jsonl \
  --api-base http://10.102.34.67:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen3.5-27B \
  --concurrency 64 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 2 \
  --answer-aware > build_knowledge_graph_v4.4.log 2>&1 &

````
- 在qwen3.5-27B上运行，生成质量挺好

### qwen2.5-72B(/4bit)

model serving:
````bash
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 vllm serve models/qwen2.5-72B \
  --served-model-name Qwen2.5-72B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code


VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 vllm serve models/qwen2.5-72B-4bit/ \
  --served-model-name Qwen2.5-72B-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code
````

request:
````bash
python build_knowledge_graph_v4.4.py   --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl   --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.4-qwen2.5_72B.jsonl   --api-base http://localhost:8000/v1   --api-mode chat   --allow-empty-api-key   --model Qwen2.5-72B   --concurrency 8   --skip-comparison   --resume   --overwrite   --sample-retries 0   --answer-aware   --limit 1 --verbose --seed 42

python build_knowledge_graph_v4.4.py   --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl   --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v4.4-qwen2.5_72B_4bit.jsonl   --api-base http://localhost:8000/v1   --api-mode chat   --allow-empty-api-key   --model Qwen2.5-72B-4bit   --concurrency 8   --skip-comparison   --resume   --overwrite   --sample-retries 0   --answer-aware   --limit 1 --verbose --seed 42

````

结论：
 - 4bit版本支持的batch size和单 batch下TPOT(18token/s -> 48token/s)都有明显提升
 - 生成质量有些微偏移
 - 非量化版本倾向于生成“_”连接的实体和关系，4bit版本没有这个问题


 # Triple-Only (v5)

仅生成三元组，不再要求抽取KV对。
属性三元组：
- 正向：the <rel> of <head> is <tail>
- 反向：<tail> is the <rel> of <head>
关系三元组：
- 正向：<head> <rel> <tail>
- 反向：the entity that is related to <tail> by "<rel>" is <head>

````bash

# 启动模型服务
conda activate vllm-graph
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen2.5-72B-4bit/ \
  --served-model-name Qwen2.5-72B-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code > Qwen2.5-72B-4bit.log 2>&1 &

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384  \
  --kv-cache-dtype auto \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}'  \
  --trust-remote-code > Qwen3.5-27B.log 2>&1 &


#8卡服务器
conda activate triple
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen2.5-72B-4bit/ \
  --served-model-name Qwen2.5-72B-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 16384  \
  --enable-prefix-caching \
  --trust-remote-code > Qwen2.5-72B-4bit.log 2>&1 &

# 单样本测试
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 64 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --answer-aware \
  --limit 1 --verbose --seed 48


# 抽取示例样本 -> 抽20个样本，模型自己的 answer_sufficient 是 18/20 = True，但如果严格按“只看 triple_list，不能借常识补桥”来评，我觉得是高估了。更贴切的结论是：严格可答率：13/20， 宽松可答率：16/20； 总用时2分40，rate=0.12/s。
python build_knowledge_graph_v5.py   --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl   --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl   --api-base http://localhost:8000/v1   --api-mode chat   --allow-empty-api-key   --model Qwen2.5-72B-4bit   --concurrency 64   --skip-comparison   --resume   --overwrite   --sample-retries 0   --answer-aware   --limit 20 --seed 42

# 对比同样的样本，用qwen3.5-27B抽的效果 -> 严格可答：14/20， 宽松可答：15/20。整体差不多。用时3分10秒，rate=0.10/s。最终结论，还是用2.5的抽
python build_knowledge_graph_v5.py   --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl   --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/musique_train_tripled_v5-qwen3.5-27B.jsonl   --api-base http://localhost:8000/v1   --api-mode chat   --allow-empty-api-key   --model Qwen3.5-27B   --concurrency 64   --skip-comparison   --resume   --overwrite   --sample-retries 0   --answer-aware   --limit 20 --seed 42


# 测试Hotpot数据集 -> 感觉还行
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_train.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 64 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --answer-aware \
  --limit 1 --verbose --seed 48

# 测试2wiki数据集 -> 
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_train_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 64 \
  --skip-comparison \
  --resume \
  --overwrite \
  --sample-retries 0 \
  --answer-aware \
  --limit 1 --verbose --seed 48
````

批次大小测试:
  结论：64够了，再大性能不会再提
````bash
#=========4ul40===========

# 64批次 -> 最高Generation Throughput 947.1 tokens/s，GPU KV Cache= 27%, rate=0.08, 0.11, 0.15, 0.17
python build_knowledge_graph_v5.py   --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl   --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl   --api-base http://localhost:8000/v1   --api-mode chat   --allow-empty-api-key   --model Qwen2.5-72B-4bit   --concurrency 64   --skip-comparison   --resume   --overwrite   --sample-retries 0   --answer-aware

# 128批次 -> 最高Generation Throughput 1049.6 tokens/s，GPU KV Cache= 42.5%, rate=0.05, 0.09, 0.12, 0.15
python build_knowledge_graph_v5.py   --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl   --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl   --api-base http://localhost:8000/v1   --api-mode chat   --allow-empty-api-key   --model Qwen2.5-72B-4bit   --concurrency 128   --skip-comparison   --resume   --overwrite   --sample-retries 0   --answer-aware

````

## 实操
````bash
# 4ul40
nohup python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/musique_train.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/musique_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 64 \
  --skip-comparison \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/musique \
  --sample-retries 2 \
  --answer-aware >> musique_train_tripled_v5-qwen2.5-72B_4bit.log 2>&1 &
nohup python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_train_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 64 \
  --skip-comparison \
  --resume \
  --stage-cache-dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/v5_cache/2wiki \
  --sample-retries 2 \
  --answer-aware >> 2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.log 2>&1 &

# 8u4090
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
  --input /home/zhchen/zwb/datasets/hotpot_train.json \
  --output /home/zhchen/zwb/datasets/hotpot_train_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 128 \
  --skip-comparison \
  --resume \
  --stage-cache-dir /home/zhchen/zwb/datasets/v5_cache/hotpot \
  --sample-retries 2 \
  --answer-aware > hotpot_train_tripled_v5-qwen2.5-72B_4bit.log 2>&1 &

nohup python build_knowledge_graph_v5.py \
  --input /home/zhchen/zwb/datasets/2wiki_train_2hop.json \
  --output /home/zhchen/zwb/datasets/2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --concurrency 64 \
  --skip-comparison \
  --resume \
  --stage-cache-dir /home/zhchen/zwb/datasets/v5_cache/2wiki \
  --sample-retries 2 \
  --answer-aware > 2wiki_train_2hop_tripled_v5-qwen2.5-72B_4bit.log 2>&1 &

````

## triples/s

- 4ul40, qwen2.5-72B-4bit, TP4
  - Musique， batch size 64, SO(supporting only) -> 11.28  triples/s
  - Hotpot, batch size 64, SO -> 13.06 triples/s
  - Hotpot, batch size 128, SO -> 7.33 triples/s
- 4ul40, qwen2.5-72B-4bit, TP1, PP4 (❌️)
- 4ul40, qwen2.5-72B-4bit, 8192
  - Hotpot, batch size 64, SO -> 13.5 triples/s

````bash
conda activate vllm-graph
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
nohup vllm serve models/qwen2.5-72B-4bit/ \
  --served-model-name Qwen2.5-72B-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192  \
  --enable-prefix-caching \
  --trust-remote-code > Qwen2.5-72B-4bit.log 2>&1 &

conda activate kblam
python build_knowledge_graph_v5.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/hotpot_train.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/tmp.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-mode chat \
  --allow-empty-api-key \
  --model Qwen2.5-72B-4bit \
  --overwrite \
  --concurrency 64


````

