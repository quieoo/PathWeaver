# Triples Gen
调用LLM提取三元组。
注：如果直接在句子embedding上训练这步可以跳过。
注：直接在句子上做可能语义压缩太大，可以考虑3/4元组手动切割。


1. 运行LLM，这里是用昇腾卡跑的VLLM-DeeepSeekV3。其它模型也行，主要关注端口号和模型名称
````bash
export MODEL_EXTRA_CFG_PATH=../../tests/test_config/test_config_single.json
MODEL_PATH=/models/deepseekv3-w4a8
MODEL_LEN=73800
TOTAL_LEN=73800
GPU_UTIL=0.975
MAX_NUM_SEQS=128
export GLOO_SOCKET_IFNAME=enp67s0f5
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
# export VLLM_LOGGING_LEVEL=INFO
export VLLM_LOGGING_LEVEL=ERROR
export HCCL_OP_EXPANSION_MODE="AIV"
export TNG_HOST_COPY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=20
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export ASCEND_GLOBAL_LOG_LEVEL=3
export NUM_SPECULATIVE_TOKENS=1
python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 7000 \
  --model $MODEL_PATH \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --max-model-len $MODEL_LEN \
  --max-num-batched-tokens $TOTAL_LEN \
  --trust_remote_code \
  --gpu_memory_utilization $GPU_UTIL \
  --block_size 128 \
  --served-model-name deepseek \
  --distributed-executor-backend mp \
  --max-num-seqs $MAX_NUM_SEQS \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --preemption-mode swap \
  --additional-config '{"graph_model_compile_config": {"level": 1}, "enable_hybrid_graph_mode": true}'
````

2. 调用scripts/gen_triples_2wiki_v2.1.py脚本，提取三元组
````bash
python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_train_2hop.json --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/2wiki_train_2hop.jsonl  --batch-size 64 --max-tokens 512 --endpoint http://localhost:7000/v1/completions --model deepseek --supporting-only

python gen_triples_2wiki_v2.1.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/source_data/2wiki_dev_2hop.json --output /docker/datasets/2wiki_hotpot_musique/merged_data/all_triples/2wiki_dev_2hop.jsonl  --batch-size 64 --max-tokens 512 --endpoint http://localhost:7000/v1/completions --model deepseek --supporting-only
````
注：添加--supporting-only，比较样本中的supporting_facts，跳过样本中和问答不相关的段落，可以提高提取效率。

输入数据集格式：
````
{
  "_id": "字符串，唯一ID",
  "type": "字符串，comparison 或者 bridge",
  "question": "字符串，对比类问题",
  "context": "[[主题, [文本片段,...]],...]",
  "supporting_facts": "[[主题, 索引],...]",
  "evidences": "[[实体, 属性, 值],...]",
  "answer": "yes / no"
}
````
输出：
````
{
  "_id": "字符串，唯一ID",
  "type": "字符串，comparison 或者 bridge",
  "question": "字符串，对比类问题",
  "context": [
    {
        'title': '主题',
        'context': '文本片段',
        'triple_list': [
            {
                'type': '三元组类型',
                'name': '实体',
                'descprition_type': '属性',
                'description': '值',
                'key_string': 'the <descprition_type> of <name>'
            },...
        ]
    },...
  ],
  "supporting_facts": "[[主题, 索引],...]",
  "evidences": "[[实体, 属性, 值],...]",
  "answer": "yes / no"
}
````

# Prepare Training Set
做一次离线的建图+检索，为每个训练样本生成top-k negative path以训练。

调用scripts/AT2QA_v2.py：
````bash
python AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_train_2hop.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --k 16 

python AT2QA_v2.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop.json \
  --st_model /home/sdu/zhu/models/qwen-embedding-0.6B \
  --batch_size 256 \
  --keep_score \
  --k 16
````
注：现在是向量检索+路径排序。如果是句子embedding的话这里的检索逻辑应该做一定的简化。直接向量检索（question vs key_string）应该就够了。


## Pre-Train Enhancement
silver path：指向正确答案的路径
从top-k negative path中找到silver path，移动到最前面 (即使经过前一步排序，top-1正确答案召回率只有50%，训练稳定性会低)。

````bash
python extract_silver_path.py
````


# Embeddings Gen
对每个三元组计算key/value embedding, 并存储到文件中。
统计生成每个样本的起始embedding索引和数量（因为有些样本可能找不到negative path，所以长度不固定）。

scripts/embedding_v2.py
````bash
python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_silver.json \
  --batch_size 1024

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type at2qa_2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_train_2hop_silver.json \
  --batch_size 1024
````

最后生成三个文件：
- 数据集文件：
````bash
  {
    "id": "",
    "Q": "",
    "A": "",
    "triple_lists": [
        [
            {
                'key_string': '',
                'description': ''
            },
            ,...
        ],
        [其它路径],...
    ],
    "start_id": 0,
    "num_triples": 32
  },
````
- precomputed_embed_keys
- precomputed_embed_values

注：如果是句子embedding，可以不区分key和value，这里应该生成一份embedding就行。

# Train

假设训练/验证的数据集文件、embedding文件都有，以下是一个参数配置示例：
````bash
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
````

关键参数：
- B：一个批次的请求共享一份KB，一个请求有16条negative path，所以B大了之后无关KB增多容易导致精度崩溃（多跳时候测试过B=5）
- lr: 学习率，目前测试5e-4比较稳
- gradient_accm_step：梯度累计步数，这里设置为20，相当于每个批次训练20步再做一次参数更新
- dataset_type：数据集类型，决定训练脚本里面样本处理、kb拼接相关的逻辑
- path_attn: 决定在注意力计算时是否做路径分数传播
- total_steps：训练步数
- N：训练样本数，和实际样本总大小取min
- eval_step：每隔多少步跑一次验证
- save_period：每隔多少步保存一份checkpoint
- keep_top_k_ckpt：保留验证精度最高的前k份checkpoint，不设置的话只保留最新的
- encoder_spec：使用什么embedding模型作为Adapter的基础模型
- kb_token_layer_frequency：注入KB的频率，默认是3，每隔3层注入一次KB
- use_cached_embd：是否使用提前计算好的embedding，需要设置precomputed_*参数
- hf_model_spec：基础模型


关键代码(train.py):
- get_batch/get_batch_at2qa: 决定选择哪些样本，生成问题和答案采样用于训练
- kbretriever.get_embeddings: 获得当前训练批次的KB（通过训练的Adapter转换成KV Cache格式），如果是多条数据集还会生成标识路径的邻接矩阵
- self.model: 调用src/models/llama3_model.py(模型文件), src/kblam_attention/kblam_injector.py(KB注入)，src/kblam_attention/kblam_path.py(路径注意力传播)
- safe_evaluate_wrapper: 组装验证集参数，调用eval_generation.py（推理代码）

# Sentence-Level Knowledge Injection

建议：
1. 句子可以先尝试按照n元组切（n=3/4）
2. 构建训练集：将和问题最相关的n元组提取出来，结合答案，构建每个样本的候选n元组（一条最正确+n条相关）。答案可以先短一点，比如只需要一个n元组就能回答。
3. 使用同样的方法构建验证集
4. 离线为每个n元组都计算好候选n元组的embedding，在线时直接按照索引访问
5. 修改src/KBRetriever.py，支持同一份embedding，分别调用encode_key和encode_value，生成KB；组装KB，前期一条最相关n元组+一条无关n元组，后期慢慢加入相关n元组；可以参考get_key_embeddings(),这是原始的单跳组装方法
6. 修改train.py和eval_generation.py的相关代码，调用新的get_embeddings()方法
7. 去掉“--path_attn”, 不需要开启路径注意力

我在训练的时候遇到过一个坑，就是将embedding改成在线计算之后训练精度波动会非常大，即使使用和离线计算embedding一模一样的代码，从头基于在线计算重新训练也不行，结果就是发现离线和在线计算出来的embedding是不一样的。最后只能改成离线批量为所有三元组提前计算好embedding。推测可能和训练过程中本身是在微调Adapter有关，可能训练过程中的参数影响了这个计算。