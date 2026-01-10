## Synthetic

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
  --eval_step 100 \
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
    --kb_size=10 \
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

### 梯度破坏验证

#### 模型状态没有正确恢复到 train() 模式
  模型本来就应该都是eval状态，去除调整model.train()和gradient checkpointing的操作

#### 评估阶段改变了随机数状态

#### 优化器或 Scheduler 状态未同步恢复

#### 评估过程使用了 KB embedding 生成函数（带梯度依赖）


## squad 数据集
准备好squad数据集，转换为synthetic格式

创建embedding，为了控制变量，先使用旧版的KBLaM创建方法和all-MiniLM-L6-v2模型
````bash
python generate_kb_embeddings.py \
    --model_name all-MiniLM-L6-v2 \
    --dataset_name squad_train --dataset_path /mnt/n0/datasets/squad/v4/train_datasets.json \
    --output_path /mnt/n0/datasets/squad/v4/

python generate_kb_embeddings.py \
    --model_name all-MiniLM-L6-v2 \
    --dataset_name squad_test --dataset_path /mnt/n0/datasets/squad/v4/test_datasets.json \
    --output_path /mnt/n0/datasets/squad/v4

# 尝试新的qwen3-embedding-0.6B模型
conda activate qwen-embedding
python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type synthetic \
  --dataset_path /mnt/n0/datasets/squad/v4/train_datasets.json \
  --batch_size 1024

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type synthetic \
  --dataset_path /mnt/n0/datasets/squad/v4/test_datasets.json \
  --batch_size 1024

````

### squad训练性能优化-1

#### 训练时随机化正确键位置
在训练时，为了增加模型的鲁棒性，我们可以在每个批次中随机化正确键的位置。这样可以防止模型依赖于键的固定位置，提高其泛化能力。



#### 评估与解码对齐“短答案”

在提示里固定一句：“Answer with the shortest span from the context; do not add extra words.” 或者解码参数：temperature=0, top_p=1, no_repeat_ngram_size=3, max_new_tokens=20–30。

````python
# 修改eval_util.py, 添加如下方法，并实现当dataset_type为squad时，使用短答案格式
def format_Q_llama_short(Q: str):
    # short answer 
    Q = f"{Q} Answer with the shortest span from the context; do not add extra words."
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )

def format_QA_llama_short(Q: str, A: str):
    
    # short answer 
    Q = f"{Q} Answer with the shortest span from the context; do not add extra words."

    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )
````

解码参数我现在已经在answer_questions_deterministic中添加了,暂时不需要继续修改

重新训练：

````bash
nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 5000 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/train_datasets.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/squad-1 \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --dataset_type squad \
  >> train_squad_1.log 2>&1 &

````

#### 放大短答案的监督强度
把 effective batch size 提大（真批或梯度累积 ×2）

````bash
nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 5000 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/train_datasets.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/squad-2 \
   --save_period 100 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --dataset_type squad \
  --gradient_accm_step 20 --total_steps 5000 \
  >> train_squad_2.log 2>&1 &

# 继续训练
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/train_datasets.json \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/squad-2.1 \
   --save_period 100 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --dataset_type squad \
  --gradient_accm_step 20 --total_steps 10000 \
  --N 9999999 \
  >> train_squad_2.1.log 2>&1 &

````


#### 把答案定位的掩码做稳
用整段 token匹配 <assistant header>，偏移 = 头部长度；
labels[attention_mask==0] = -100；
找不到头部就跳过样本（不要让 argmax 在全 0 上返回 0）

使用create_label_for_llama_enchance取代原来的label方法

````bash

nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 5000 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/debug_squad_0 \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  >> train_squad_0.log 2>&1 &

````


### Squad数据集上的语义对齐

````bash

python ../tools/semantic_alignment_eval.py --dataset_path ../datasets/synthetic_embd/train_datasets.json --base_encoder all-MiniLM-L6-v2 --trained_encoder_path ./train/debug_train_evaluate/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4799_encoder/encoder.pt --out_dim 135168 --post_sample 10000
# Samples = 10000 | Dim = 135168
# Pre:  MRR=0.9840, Top1=0.9691, Top5=0.9999
# Post: MRR=0.9773, Top1=0.9577, Top5=0.9995
# DiagCos: Pre=0.9560, Post=0.9780
# RankCorr: Spearman=0.8476, Kendall=0.6957


python ../tools/semantic_alignment_eval.py --dataset_path /mnt/n0/datasets/squad/train_datasets.json --base_encoder all-MiniLM-L6-v2 --trained_encoder_path ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_4999_encoder/encoder.pt --out_dim 135168 --post_sample 10000

# Samples = 10000 | Dim = 135168
# Pre:  MRR=0.9698, Top1=0.9570, Top5=0.9859
# Post: MRR=0.9543, Top1=0.9359, Top5=0.9763
# DiagCos: Pre=0.8579, Post=0.9455
# RankCorr: Spearman=0.9117, Kendall=0.7595
````

### squad的刻度问题

不同的kb_scale_factor对结果的影响
````bash
nohup python eval.py generation \
    --kb_size=10 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_4999_encoder/encoder.pt \
    --model_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_4999 \
    --kb_layer_frequency 1 --kb_scale_factor_range 0.25 8 \
    --dataset_dir /mnt/n0/datasets/squad/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed 1  >> eval_squad.log 2>&1 &

````
测试结果：
- true KB Token 召回率没有明显的提升
- 整体精度先上升再下降，在kb_scale_factor=1时达到最高


kb_size的影响
````bash
nohup python eval.py generation \
    --kb_size=2 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_4999_encoder/encoder.pt \
    --model_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_4999 \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed 1 --save_dir ./gen_tmp >> eval_squad.log 2>&1 &
````

NO-KB



### squad数据集重构

Squad数据集的一个样本中包含一个qas列表，对于其中的每一个qa对构建一条类似Synthetic数据集的样本用来训练KBLaM，具体要求如下：
1. 新的样本中必须包含字段：name, description_type, description, Q, A, key_string
2. 首先从qa对中抽取三元组, (name, description_type, description)
3. Q可以表示为：What is the <description_type> of <name>?
4. A可以表示为：The <description_type> of <name> is <description>.
5. key_string可以表示为：the <description_type> of <name>
以下是Squad数据集中的一个qa对示例，如果是你的话你会怎么抽取样本：
"question": "What is in front of the Notre Dame Main Building?", 
"answer": "a copper statue of Christ"

{
  "name": "Notre Dame Main Building",
  "description_type": "object in front of",
  "description": "a copper statue of Christ",
  "Q": "What is the object in front of Notre Dame Main Building?",
  "A": "The object in front of Notre Dame Main Building is a copper statue of Christ.",
  "key_string": "the object in front of Notre Dame Main Building"
}


提示词模板：

````python
PROMPT_TEMPLATE = Template("""You are an expert knowledge extraction system designed to convert natural language QA pairs into structured knowledge samples for a knowledge-grounded language model called KBLaM.

You are given a question and its corresponding answer from a QA dataset (e.g., SQuAD).  
Your task is to extract a structured sample in JSON format compatible with the KBLaM training format.

Follow these rules carefully:
1. You must output a single JSON object with the following fields:
   - "name": the main entity mentioned in the question.
   - "description_type": the relation or attribute being asked about (e.g., "location of", "object in front of", "person who discovered", etc.).
   - "description": the factual answer to the question.
   - "Q": reformulate the question as "What is the <description_type> of <name>?" or an equivalent clear form.
   - "A": reformulate the answer as "The <description_type> of <name> is <description>."
   - "key_string": form it as "the <description_type> of <name>".

2. Keep all values in natural English and ensure factual consistency with the question and answer.

3. Do not add extra explanations or text — output only the JSON object.

Example:
Input:
Question: "What is in front of the Notre Dame Main Building?"
Answer: "a copper statue of Christ"

Output:
{
  "name": "Notre Dame Main Building",
  "description_type": "object in front of",
  "description": "a copper statue of Christ",
  "Q": "What is the object in front of Notre Dame Main Building?",
  "A": "The object in front of Notre Dame Main Building is a copper statue of Christ.",
  "key_string": "the object in front of Notre Dame Main Building"
}

Now, process the following QA pair and output the structured JSON:
Question: $question
Answer: $answer
""")
````





### squad-fromated-Synthetic参数
````bash
nohup python train.py \
  --seed 1 --N 99999999 --B 10  --lr 5e-4 --total_steps 5000 \
  --sep_query_head --use_cached_embd --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/v4/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/squad_2.4 \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/v4/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/squad_test_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/squad_test_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  >> train_squad_2.4.log 2>&1 &

````

训练
````bash
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
    --kb_size=10 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_9700_encoder/encoder.pt  \
    --model_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_9700 \
    --kb_layer_frequency 1 --kb_scale_factor_range 0.25 4 \
    --dataset_dir /mnt/n0/datasets/squad/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed 1 >> eval_squad.log 2>&1 &

# rag
 python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/squad/plain_text/validation_merged.json     --dataset-type squad     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10

````


### squad在formatted基础上调参


````bash

# layer_frequency 3
nohup python train.py \
  --seed 1 --N 99999999 --B 10  --lr 5e-4 --total_steps 5000 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/v4/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/squad_train_all-MiniLM-L6-v2_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/squad_train_all-MiniLM-L6-v2_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/squad_2.5 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/v4/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/squad_test_all-MiniLM-L6-v2_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/squad_test_all-MiniLM-L6-v2_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  >> train_squad_2.5.log 2>&1 &


# embedding
nohup python train.py \
  --seed 1 --N 99999999 --B 10  --lr 5e-4 --total_steps 5000 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/v4/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/v4/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --model_save_dir ./train/squad_2.6 \
  >> train_squad_2.6.log 2>&1 &


## 扩展式训练，先4000步 120K数据集，再8000步，780K数据集
nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 4000 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/v4/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/v4/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --model_save_dir ./train/squad_2.7 \
  >> train_squad_2.7.log 2>&1 &


  nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/v4/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/v4/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000   --N 9999999   --model_dir_to_resume ./train/squad_2.7/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_3999 \
  --model_save_dir ./train/squad_2.7_stage_2 \
  >> train_squad_2.7.log 2>&1 &


## 最后的评估
nohup python eval.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/squad_2.7_stage_2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/squad_2.7_stage_2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/v4/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 100 --seed 1 --save_dir ./gen_tmp >> eval_squad2.7.log 2>&1 &

````






## Musique数据集
### 抽取三元组
Musique数据集的一个样本包含若干段落，从每个段落中抽取三元组，要求如下：
1. 从每个段落中抽取所有的实体(Entity)，实体可以是特指的人物、组织、地点等，但是不能是通用的名词或动词。
2. 对于每个实体，从段落中地抽取其所有的属性(Attribute)，属性可以是实体的一些特征或行为。
3. 对于每个实体及其属性可以构成一条属性三元组：(Entity, Attribute, Value)，并且可以表达为""The <Attribute> of <Entity> is <Value>.""
以下是Musique数据集中一个样本中的两个段落：
"title": "Learjet 60",
"paragraph_text": "The Learjet 60 is a mid-size cabin, medium-range business jet aircraft manufactured by Bombardier Aerospace in Wichita, Kansas. Powered by two Pratt & Whitney Canada PW305A engines, it has a range (with 4 passengers and 2 crew) of with NBAA reserves, ISA. In July 2012 Bombardier Aerospace announced a temporary \"production pause\" of the latest variant Learjet 60XR to begin in the fourth quarter of 2012.",

"title": "List of Bombardier CRJ operators",
"paragraph_text": "Produced by Bombardier Aerospace, aerospace division of the Canadian aerospace and defence company Bombardier Inc. the former CRJ100 and CRJ200 series are no longer in production but still in active airline service, while the more recent CRJ700, CRJ900 and CRJ1000 series are in production and in service.",


### 多跳问答

#### 2*属性
转换样本的格式，从其中抽取三元组。

输入样本包含“question”和“answer”两个字段，以及若干个段落。
对于一个两跳的问答对样本，输出应包含以下字段：
triple_lists：一个列表，包含两个三元组，每个三元组包含字段name, description_type, description, key_string。每个三元组可以组成知识“the <description_type> of <name> is <description>.”。要求抽取的两个三元组中第二个三元组的<name>是第一个三元组的<description>值，第二个三元组的<description>值为answer。两个三元组合起来一定能够正确回答question。key_string为“the <description_type> of <name>”。
Q: 语义和question等价，但是格式可以表达为"What is the <description_type_2> of <description_type_1> of <name>?"
A: answer

返回如下所示：
{
  "Q": "What is the date of birth of the father of Mina Gerhardsen?",
  "A": "13 June 1946",
  "triple_lists": [
    {
      "name": "Mina Gerhardsen",
      "description_type": "father",
      "description": "Rune Gerhardsen",
      "key_string": "the father of Mina Gerhardsen"
    },
    {
      "name": "Rune Gerhardsen",
      "description_type": "date of birth",
      "description": "13 June 1946",
      "key_string": "the date of birth of Rune Gerhardsen"
    }
  ],
}
以下是一个输入样本：
  {
    "id": "08a02c840bdb11eba7f7acde48001122",
    "question": "When did Fatima Bint Mubarak Al Ketbi's husband die?",
    "answer": "2 November 2004",
    "paragraphs": [
      {
        "title": "Fatima bint Mubarak Al Ketbi",
        "paragraph_text": "Mubarak Al Ketbi  is the third wife of Sheikh Zayed bin Sultan Al Nahyan, the founder and inaugural president of United Arab Emirates, and late emir (ruler) of Abu Dhabi."
      },
      {
        "title": "Zayed bin Sultan Al Nahyan",
        "paragraph_text": "Sheikh Zayed bin Sultan Al Nahyan ; 6 May 1918 – 2 November 2004) was the ruler of Abu Dhabi for more than 30 years (6 August 1966 – 2 November 2004)."
      }
    ],
    "source": "2wiki"
  },

#### 2*属性，属性+关系，2*关系

你需要从输入样本中抽取两跳推理链，并将其转换为结构化三元组。

输入样本包含“question”和“answer”两个字段，以及若干个段落。
对于一个两跳的问答对样本，输出应包含以下字段：
triple_lists：
    一个列表，包含两个三元组，可以保证回答question中的问题。
    每个三元组包含字段name, description_type, description, key_string。
    根据语义判断使用属性三元组还是关系三元组。
    被抽取的两个三元组中第二个三元组的<name>与第一个三元组的<description>值相同，第二个三元组的<description>与answer相同。
    属性三元组可以形成一条属性知识“the <description_type> of <name> is <description>”，关系三元组可以形成一条关系知识"<name> <description_type> <description>"。
    对于属性三元组key_string为“the <description_type> of <name>”，对于关系三元组key_string为“<name> <description_type>”。
Q: 
    语义和question等价。
    如果两个三元组都是属性三元组，那么Q可以表达为"What is the <description_type_2> of <description_type_1> of <name>?"。
    如果第一个三元组是关系三元组，第二个三元组是属性三元组，那么Q可以表达为"What is the <description_type_2> of which <name> <description_type_1>?"
    如果两个三元组都是关系三元组，那么Q可以表达为"What the one that <name1> <description_type_1> <description_type_2>?"
A: answer

从[2*属性三元组，关系三元组+属性三元组，2*关系三元组]中选择一种方案，使得Q更好。


以下是生成2*属性三元组的一个例子：
{
  "Q": "What is the date of birth of the father of Mina Gerhardsen?",
  "A": "13 June 1946",
  "triple_lists": [
    {
      "name": "Mina Gerhardsen",
      "description_type": "father",
      "description": "Rune Gerhardsen",
      "key_string": "the father of Mina Gerhardsen"
    },
    {
      "name": "Rune Gerhardsen",
      "description_type": "date of birth",
      "description": "13 June 1946",
      "key_string": "the date of birth of Rune Gerhardsen"
    }
  ],
}

以下是生成关系三元组+属性三元组的一个例子：
{
  "Q": "What is the other name of which Cadmium chloride slightly soluble in?",
  "A": "alcohol",
  "triple_lists": [
    {
      "name": "Cadmium chloride",
      "description_type": "is slightly soluble in",
      "description": "alcohol",
      "key_string": "Cadmium chloride is slightly soluble in"
    },
    {
      "name": "alcohol",
      "description_type": "other name",
      "description": "alcohol",
      "key_string": "the other name of alcohol"
    }
  ]
}

以下是生成2*关系三元组的一个例子：
{
  "Q": "What the one that Allie Goertz wrote a song about is named after?",
  "A": "President Richard Nixon",
  "triple_lists": [
    {
      "name": "Allie Goertz",
      "description_type": "wrote a song about",
      "description": "Milhouse",
      "key_string": "Allie Goertz wrote a song about"
    },
    {
      "name": "Milhouse",
      "description_type": "is named after",
      "description": "President Richard Nixon",
      "key_string": "Milhouse is named after"
    }
  ]
}

转换以下例子：
  {
    "id": "5aba66c855429939ce03dcdb",
    "question": "Gunmen from Laredo starred which narrator of \"Frontier\"?",
    "answer": "Walter Darwin Coy",
    "paragraphs": [
      {
        "title": "Gunmen from Laredo",
        "paragraph_text": "Gunmen from Laredo is a 1959 American western film produced and directed by Wallace MacDonald, which stars Robert Knapp, Maureen Hingert, and Walter Coy."
      },
      {
        "title": "Walter Coy",
        "paragraph_text": "Walter Darwin Coy (January 31, 1909 – December 11, 1974) was an American stage, radio, film, and, principally, television actor, originally from Great Falls, Montana. He was best known for narrating the NBC western anthology series, \"Frontier\", which aired early Sunday evenings in the 1955–1956 season."
      }
    ],
    "source": "hotpot"
  }

#### v2

以下是从输入样本中抽取两跳推理三元组并生成问答格式的规范。请严格遵守全部规则。任何输出都必须满足两跳链路约束。

所有样例必须形成严格的两跳推理链路：第二个三元组的 name 必须与第一个三元组的 description 完全一致；第二个三元组的 description 必须与最终 answer 完全一致。这是必须满足的强约束，任何情况下不得违反、跳步、绕过或跨实体替换。

输入样本包含字段 “question”、“answer” 以及若干段落。输出必须包含以下字段：

triple_lists：一个包含两个三元组的列表。每个三元组包含字段 name, description_type, description, key_string。

所有三元组必须根据语义选择属性三元组或关系三元组：

属性三元组表示为 “the <description_type> of <name> is <description>”

关系三元组表示为 “<name> <description_type> <description>”

属性三元组的 key_string 必须为 “the <description_type> of <name>”

关系三元组的 key_string 必须为 “<name> <description_type>”

两跳链路必须满足：
(1) 第一个三元组的 description 代表中间实体 E。
(2) 第二个三元组的 name 必须严格等于 E。
(3) 第二个三元组的 description 必须严格等于 answer。
这三个条件必须同时满足。

你必须在以下三种三元组组合中选择语义最自然的一种，并严格使用对应问答对模板：
(1) 若两个三元组都是属性三元组，Q和A 必须表达为：
“What is the <description_type_2> of the <description_type_1> of <name>?”
“The <description_type_2> of the <description_type_1> of <name> is <description_2>.”
(2) 若第一个三元组是关系三元组，第二个是属性三元组，Q和A 必须表达为：
“What is the <description_type_2> of which <name> <description_type_1>?”
“The <description_type_2> of which <name> <description_type_1> is <description_2>.”
(3) 若两个三元组都是关系三元组，Q和A必须表达为：
“What the one that <name1> <description_type_1> <description_type_2>?”
“The one that <name1> <description_type_1> <description_type_2> is <description_2>.”


生成的 Q 必须语义等价于原始 question，不得添加无关信息，不得改变推理链条。

输出格式必须为：
{
"Q": "...",
"A": "...",
"triple_lists": [
{...},
{...}
]
}

以上全部规则必须严格遵守，并确保抽取的两跳链路可以完整、正确地回答原始 question。
以下是生成2*属性三元组的一个例子：
{
  "Q": "What is the date of birth of the father of Mina Gerhardsen?",
  "A": "The date of birth of the father of Mina Gerhardsen is 13 June 1946.",
  "triple_lists": [
    {
      "name": "Mina Gerhardsen",
      "description_type": "father",
      "description": "Rune Gerhardsen",
      "key_string": "the father of Mina Gerhardsen"
    },
    {
      "name": "Rune Gerhardsen",
      "description_type": "date of birth",
      "description": "13 June 1946",
      "key_string": "the date of birth of Rune Gerhardsen"
    }
  ],
}

以下是生成关系三元组+属性三元组的一个例子：
{
  "Q": "What is the other name of which Cadmium chloride slightly soluble in?",
  "A": "The other name of which Cadmium chloride slightly soluble in is alcohol.",
  "triple_lists": [
    {
      "name": "Cadmium chloride",
      "description_type": "is slightly soluble in",
      "description": "alcohol",
      "key_string": "Cadmium chloride is slightly soluble in"
    },
    {
      "name": "alcohol",
      "description_type": "other name",
      "description": "alcohol",
      "key_string": "the other name of alcohol"
    }
  ]
}

以下是生成2*关系三元组的一个例子：
{
  "Q": "What the one that Allie Goertz wrote a song about is named after?",
  "A": "The one that Allie Goertz wrote a song about is named after President Richard Nixon.",
  "triple_lists": [
    {
      "name": "Allie Goertz",
      "description_type": "wrote a song about",
      "description": "Milhouse",
      "key_string": "Allie Goertz wrote a song about"
    },
    {
      "name": "Milhouse",
      "description_type": "is named after",
      "description": "President Richard Nixon",
      "key_string": "Milhouse is named after"
    }
  ]
}

转换以下样本：
  {
    "id": "5aba66c855429939ce03dcdb",
    "question": "Gunmen from Laredo starred which narrator of \"Frontier\"?",
    "answer": "Walter Darwin Coy",
    "paragraphs": [
      {
        "title": "Gunmen from Laredo",
        "paragraph_text": "Gunmen from Laredo is a 1959 American western film produced and directed by Wallace MacDonald, which stars Robert Knapp, Maureen Hingert, and Walter Coy."
      },
      {
        "title": "Walter Coy",
        "paragraph_text": "Walter Darwin Coy (January 31, 1909 – December 11, 1974) was an American stage, radio, film, and, principally, television actor, originally from Great Falls, Montana. He was best known for narrating the NBC western anthology series, \"Frontier\", which aired early Sunday evenings in the 1955–1956 season."
      }
    ],
    "source": "hotpot"
  }
  
  #### 2*关系

  
转换样本的格式，从其中抽取三元组。

输入样本包含“question”和“answer”两个字段，以及若干个段落。
对于一个两跳的问答对样本，输出应包含以下字段：
triple_lists：
    一个列表，包含两个三元组，可以保证回答question中的问题。
    每个三元组包含字段name, description_type, description, key_string。
    根据语义判断使用属性三元组还是关系三元组。
    被抽取的两个三元组中第二个三元组的<name>与第一个三元组的<description>值相同，第二个三元组的<description>与answer相同。
    每条三元组可以形成一条关系知识"<name> <description_type> <description>"。
    key_string为“<name> <description_type>”。
Q: 
    语义和question等价。
    表达为"What the one that <name1> <description_type_1> <description_type_2>?"
"
A: answer

以下是一个例子：
输入：
  {
    "id": "5a8d7341554299441c6b9fe5",
    "question": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
    "answer": "President Richard Nixon",
    "paragraphs": [
      {
        "title": "Allie Goertz",
        "paragraph_text": "Allison Beth \"Allie\" Goertz (born March 2, 1991) is an American musician. Goertz is known for her satirical songs based on various pop culture topics. Her videos are posted on YouTube under the name of Cossbysweater."
      },
      {
        "title": "Milhouse Van Houten",
        "paragraph_text": "Milhouse Mussolini van Houten is a fictional character featured in the animated television series \"The Simpsons\", voiced by Pamela Hayden, and created by Matt Groening who named the character after President Richard Nixon's middle name."
      }
    ],
    "source": "hotpot"
  },
输出：
{
  "Q": "What the one that Allie Goertz wrote a song about is named after?",
  "A": "President Richard Nixon",
  "triple_lists": [
    {
      "name": "Allie Goertz",
      "description_type": "wrote a song about",
      "description": "Milhouse",
      "key_string": "Allie Goertz wrote a song about"
    },
    {
      "name": "Milhouse",
      "description_type": "is named after",
      "description": "President Richard Nixon",
      "key_string": "Milhouse is named after"
    }
  ]
}

处理以下样本：
  {
    "id": "5adf44985542993a75d2646d",
    "question": "Which genus of moth in the world's seventh-largest country contains only one species?",
    "answer": "Crambidae",
    "paragraphs": [
      {
        "title": "Indogrammodes",
        "paragraph_text": "Indogrammodes is a genus of moths of the Crambidae family. It contains only one species, Indogrammodes pectinicornalis, which is found in India."
      },
      {
        "title": "India",
        "paragraph_text": "India, officially the Republic of India (\"Bhārat Gaṇarājya\"), is a country in South Asia. It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world."
      }
    ],
    "source": "hotpot"
  },


#### 先三元组再问题

从样本中抽取三元组。
输入样本包含一个问答对以及若干个段落。
首先从段落中提取关键信息，保证能够回答问题给出正确的答案。
之后将关键信息组成三元组。
以下是样本内容：
  {
    "id": "2hop__66890_93263",
    "question": "The athlete that became the highest-paid went to manchester United when?",
    "answer": "2003",
    "paragraphs": [
      {
        "title": "Forbes' list of the world's highest-paid athletes",
        "paragraph_text": "Rank Name Sport Nation Total Salary / Winnings Endorsements Cristiano Ronaldo Association football Portugal $93 million $58 million $35 million LeBron James Basketball United States $86.2 million $31.2 million $55 million Lionel Messi Association football Argentina $80 million $53 million $27 million Roger Federer Tennis Switzerland $64 million $6 million $58 million 5 Kevin Durant Basketball United States $60.6 million $26.6 million $34 million 6 Andrew Luck American football United States $50 million $47 million $3 million 6 Rory McIlroy Golf Northern Ireland $50 million $16 million $34 million 8 Stephen Curry Basketball United States $47.3 million $12.3 million $35 million 9 James Harden Basketball United States $46.6 million $26.6 million $20 million 10 Lewis Hamilton Auto racing England $46 million $38 million $8 million"
      },
      {
        "title": "Cristiano Ronaldo",
        "paragraph_text": "Cristiano Ronaldo GOIH, ComM Ronaldo at the 2017 FIFA Confederations Cup Full name Cristiano Ronaldo dos Santos Aveiro Date of birth (1985 - 02 - 05) 5 February 1985 (age 32) Place of birth Funchal, Madeira, Portugal Height 1.85 m (6 ft 1 in) Playing position Forward Club information Current team Real Madrid Number 7 Youth career 1992 -- 1995 Andorinha 1995 -- 1997 Nacional 1997 -- 2002 Sporting CP Senior career * Years Team Apps (Gls) 2002 -- 2003 Sporting CP B (0) 2002 -- 2003 Sporting CP 25 (3) 2003 -- 2009 Manchester United 196 (84) 2009 -- Real Madrid 270 (286) National team 2001 Portugal U15 9 (7) 2001 -- 2002 Portugal U17 7 (5) 2003 Portugal U20 5 (1) 2002 -- 2003 Portugal U21 10 (3) Portugal U23 (2) 2003 -- Portugal 147 (79) Honours (show) Representing Portugal UEFA European Championship Winner 2016 France Runner - up 2004 Portugal 2012 Poland & Ukraine FIFA Confederations Cup 2017 Russia * Senior club appearances and goals counted for the domestic league only and correct as of 23: 00, 22 October 2017 (UTC). ‡ National team caps and goals correct as of 22: 40, 10 October 2017 (UTC)"
      }
    ],
    "source": "musique"
  },


#### 2*属性，强制what

Below are the specifications for extracting two-hop reasoning triples from the input sample and generating a QA-format output. You must follow ALL rules strictly. Every output MUST satisfy the two-hop reasoning chain constraints.

All samples MUST form a strict two-hop reasoning chain:
- The “name” field of the second triple MUST match exactly the “description” field of the first triple.
- The “description” of the second triple MUST match exactly the final answer.
These are hard constraints. They must NEVER be violated, skipped, bypassed, or replaced across entities.

The input sample contains the fields “question”, “answer”, and several paragraphs. The output MUST contain:

triple_lists: a list of exactly two triples.
Each triple contains fields: name, description_type, description, key_string.

You MUST treat BOTH triples as ATTRIBUTE triples. Relation triples are NOT allowed in any case.

- Attribute triple format: “the <description_type> of <name> is <description>”
- Attribute triple key_string: “the <description_type> of <name>”

The two-hop chain MUST satisfy:
(1) The description of the first triple is the intermediate entity E.
(2) The name of the second triple MUST equal E exactly.
(3) The description of the second triple MUST equal the final answer exactly.
All three conditions MUST be satisfied.

Since BOTH triples are attribute triples, you MUST ALWAYS use the following QA template STRICTLY:

Q: “What is the <description_type_2> of the <description_type_1> of <name>?”
A: “The <description_type_2> of the <description_type_1> of <name> is <description_2>.”

The generated Q MUST:
- Be an English interrogative sentence.
- START with the word “What”.
- Be semantically equivalent to the original question, preserving the same reasoning chain and answer.
No irrelevant information may be added, and the reasoning chain must never be altered.

OUTPUT FORMAT (MUST follow exactly):
{
  "Q": "...",
  "A": "...",
  "triple_lists": [
    {...},
    {...}
  ]
}

All rules above MUST be enforced strictly, ensuring the extracted two-hop chain fully and correctly answers the original question.

Below is an example demonstrating correct outputs for two attribute triples:

=== Example: Two Attribute Triples ===
{
  "Q": "What is the date of birth of the father of Mina Gerhardsen?",
  "A": "The date of birth of the father of Mina Gerhardsen is 13 June 1946.",
  "triple_lists": [
    {
      "name": "Mina Gerhardsen",
      "description_type": "father",
      "description": "Rune Gerhardsen",
      "key_string": "the father of Mina Gerhardsen"
    },
    {
      "name": "Rune Gerhardsen",
      "description_type": "date of birth",
      "description": "13 June 1946",
      "key_string": "the date of birth of Rune Gerhardsen"
    }
  ]
}

Convert the following sample:


## 2wiki
训练集规模：76481

````bash
# 生成三元组
nohup python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/merge_data/merged_train.json --output /docker/datasets/2wiki_hotpot_musique/train_datasets.json --type 2wiki_hotpot --specific_type 2wiki --batch-size 64 >> gen_2wiki_v3.log 2>&1 &

nohup python squad_gen_triples.py --input /docker/datasets/2wiki_hotpot_musique/merged_data/merge_data/merged_dev.json --output /docker/datasets/2wiki_hotpot_musique/test_datasets.json --type 2wiki_hotpot --specific_type 2wiki --batch-size 64 >> gen_2wiki_v3.log 2>&1 &

# 拷贝三元组文件到本地目录
/mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json
/mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets.json



# 创建embedding
python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json \
  --batch_size 1024

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets.json \
  --batch_size 1024

# 生成的embedding文件: 
/mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy
/mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy
/mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy
/mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy



# 支持新的数据集训练
## 新增并修改Trainer初始化中的get_batch函数
## 新增并修改KBRetriever中的get_key_embeddings函数

# 两阶段训练
nohup python train.py \
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
  --model_save_dir ./train/2wiki1_1.0 \
  >> train_2wiki1_1.0.log 2>&1 &


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
  --total_steps 8000   --N 9999999   --model_dir_to_resume ./train/2wiki_1.0/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_3999 \
  --model_save_dir ./train/2wiki_1.0 \
  >> train_2wiki1_1.0.log 2>&1 &

````

### 两跳合并一跳

````bash
python merge_hop.py

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type synthetic \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_train_datasets.json \
  --batch_size 1024

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B \
  --dataset_type synthetic \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_test_datasets.json \
  --batch_size 1024


nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 4000 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --model_save_dir ./train/2wiki_2.0 \
  >> train_2wiki_2.0.log 2>&1 &


  nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/single_hop/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000   --N 9999999   --model_dir_to_resume ./train/2wiki_2.0/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_3999 \
  --model_save_dir ./train/2wiki_2.0 \
  >> train_2wiki_2.0.log 2>&1 &

nohup ./train_2wiki_1.0.sh   >> train_2wiki_1.0.log 2>&1 &

nohup ./train_2wiki_2.0.sh   >> train_2wiki_2.0.log 2>&1 &

````

### 两跳不行的原因

我使用2wiki数据集(一种两跳数据集，格式固定，简单)训练KBLaM。
训练的配置完全一样的情况下我使用了两种三元组构建方法。
A：针对每条知识构建一个三元组，对于两跳数据集一个问题需要两个三元组才可以回答，并且第二个三元组的name就是第一个三元组的description值。
B：将针对同一个问题的两个三元组合并成一个三元组，合并后的三元组的键为两个三元组的键的拼接，值为正确答案。
以下是两种构建方法下的样本：
A：
{
    "Q": "What is the date of birth of the father of Mina Gerhardsen?",
    "A": "The date of birth of the father of Mina Gerhardsen is 13 June 1946.",
    "triple_lists": [
      {
        "name": "Mina Gerhardsen",
        "description_type": "father",
        "description": "Rune Gerhardsen",
        "key_string": "the father of Mina Gerhardsen"
      },
      {
        "name": "Rune Gerhardsen",
        "description_type": "date of birth",
        "description": "13 June 1946",
        "key_string": "the date of birth of Rune Gerhardsen"
      }
    ]
  },
B：
  {
    "name": "Mina Gerhardsen",
    "description_type": "date of birth of the father",
    "description": "13 June 1946",
    "Q": "What is the date of birth of the father of Mina Gerhardsen?",
    "A": "The date of birth of the father of Mina Gerhardsen is 13 June 1946.",
    "key_string": "the date of birth of the father of Mina Gerhardsen"
  },
结果表明在A组中模型推理精度很差，在训练4000步的情况下推理精度仅为0.14并且没有明显提高。而B组在相同的训练步数下推理精度快速提高到0.8，请分析原因

解决方法(ChatGPT)
  方法 1：显式链路建模

  例如让 triple1 的 description 向量 == triple2 的 name 向量
  并强制 encoder 学习对齐

  方法 2：引入 multi-hop retrieval 模块

  比如：

  recurrent retrieval (Retriever → LLM → requery → retriever → answer)

  graph-based retrieval

  FiD (Fusion-in-Decoder)

  TPR-like relational binding

  方法 3：让 LLM 先生成中间变量 B，再检索第二跳
  方法 4：训练一个集成 key-value composition 的特殊 attention


### 路径 attention

 - 修改kb_retriever.py，增加get_embeddings_with_adj_2wiki方法，用于获取2wiki数据集的三元组embedding和邻接矩阵
 - 修改llama3_model.py，增加路径attention的实现（基于给定的kb_adj）
 - 修改train.py和eval.py，调用get_embeddings_with_adj_2wiki方法，传入kb_adj


问题：
 - kb_adj的计算需要展开kb_len^2,导致显存爆炸，必须限制kb_size

````bash

# 开启路径attention
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
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
  --total_steps 8000   --N 9999999 \
  --model_save_dir ./train/2wiki_1.1 \
  >> train_2wiki_1.1.log 2>&1 &

# 关闭路径attention
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
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
  --total_steps 8000   --N 9999999 \
  --model_save_dir ./train/2wiki_1.2 \
  >> train_2wiki_1.2.log 2>&1 &

````


### 推理性能

````bash

# RAG

python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/merge_data/merged_dev.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10
 

# 2wiki_1.1
nohup python eval.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --save_dir ./gen_tmp >> eval_2wiki_1.1.log 2>&1 &

# 路径注意力版本
nohup python eval.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_2wiki_1.1.log 2>&1 &


````


## hotpot_2hop

原来的Prompt_V3处理hotpot时总是容易出现错误
- 修改：重新制作hotpot_2hop数据集，只挑出两跳样本，并且只保留上下文中的关键句，减少模型解析压力


````bash

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl \
  --batch_size 1024

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --batch_size 1024

# 在2wiki_1.1模型上增量训练，没开path_attn
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 13000   --N 9999999   --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --model_save_dir ./train/hotpot_2hop_v1 >> train_hotpot_2hop_v1.log 2>&1 &

# 对照组：在base模型上增量训练,没开path_attn
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 5000   --N 9999999 \
  --model_save_dir ./train/hotpot_2hop_v2 >> train_hotpot_2hop_v2.log 2>&1 &


# 打开PathAttn，注意kb_size不能过大
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 13000 --path_attn  --N 9999999   --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --model_save_dir ./train/hotpot_2hop_v1.1 >> train_hotpot_2hop_v1.1.log 2>&1 &

nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 5000 --path_attn  --N 9999999 \
  --model_save_dir ./train/hotpot_2hop_v2.1 >> train_hotpot_2hop_v2.1.log 2>&1 &


# 重跑2wiki1.1,验证内存泄露问题
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --path_attn  --N 9999999 \
  --model_save_dir ./train/2wiki_1.1.0 \
  >> train_2wiki_1.1.0.log 2>&1 &

# 暂时将batch缩小一半
nohup python train.py \
  --seed 1 --B 5  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 18000 --path_attn  --N 9999999   --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --model_save_dir ./train/hotpot_2hop_v1.1 >> train_hotpot_2hop_v1.1.log 2>&1 &


# 临时跑：把kb size缩小
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 20 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 13000   --N 9999999   --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --model_save_dir ./train/hotpot_2hop_v1.1 >> train_hotpot_2hop_v1.1.log 2>&1 &

````

### 内存泄露debug

#### RAG

````bash
nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/hotpot_2hop_test.json     --dataset-type hotpot     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_hotpot_2hop_1.1.log 2>&1 &

# key不重合数据集
nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_bridge.json     --dataset-type hotpot     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_hotpot_2hop_1.1.log 2>&1 &

````

````bash

# 训练hotpot

# 修改梯度检查点
# 禁用torch.compile
export TORCHDYNAMO_DISABLE=1


nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 5000 --path_attn  --N 9999999 \
  --model_save_dir ./train/hotpot_2hop_v2.1 >> train_hotpot_2hop_v2.1.log 2>&1 &


# 训练2wiki
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 --path_attn  --N 9999999 \
  --model_save_dir ./train/2wiki_1.1.0 \
  >> train_2wiki_1.1.0.log 2>&1 &
````

#### 推理

````bash

nohup python eval.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/hotpot_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_12999_encoder/encoder.pt  \
    --model_dir ./train/hotpot_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_12999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop \
    --test_dataset test_datasets.jsonl \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_hotpot_2hop_1.1.log 2>&1 &

````
#### 验证调整过后的路径注意力的正确性

````bash
## XXXXXX
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 13000 --path_attn  --N 9999999   --model_dir_to_resume ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
  --model_save_dir ./train/hotpot_2hop_v1.1.0 >> train_hotpot_2hop_v1.1.0.log 2>&1 &

````

## Musique


预处理
````
You are an expert at analyzing multi-hop question answering datasets. Your task is to extract the minimal set of key sentences from the given context paragraphs that directly support answering both the decomposed sub-questions and the main question.

**Input Structure:**
- `main_question`: The primary question to be answered
- `main_answer`: The final answer to the main question
- `paragraphs`: A list of article paragraphs, each containing:
  - `title`: The paragraph title
  - `paragraph_text`: The full text content
- `question_decomposition`: A list of reasoning steps, each containing:
  - `id`: Step identifier
  - `question`: Sub-question for this reasoning step
  - `answer`: Answer to the sub-question
  - `paragraph_support_idx`: Index of the supporting paragraph

**Your Task:**
For each item in `question_decomposition`, extract **exactly one** sentence from the specified `paragraph_support_idx` that provides the essential factual basis for that step's answer.

**CRITICAL CONSTRAINTS:**
1. Output **only** the extracted sentences—NO explanations, NO reasoning, NO summaries
2. Extract verbatim text—do not paraphrase or alter
3. Do not include any analysis, comments, or markdown formatting
4. If a sentence is long, extract the minimal sufficient fragment

Now process the following sample:

{'id': '2hop__269805_135710', 'question': 'What is the country where Nissedal is located named after?', 'answer': 'north', 'paragraphs': [{'title': 'Norway', 'paragraph_text': "Norway has a total area of and a population of 5,312,300 (as of August 2018). The country shares a long eastern border with Sweden (1,619 km or 1,006\xa0mi long). Norway is bordered by Finland and Russia to the north-east, and the Skagerrak strait to the south, with Denmark on the other side. Norway has an extensive coastline, facing the North Atlantic Ocean and the Barents Sea. The maritime influence also dominates Norway's climate with mild lowland temperatures on the sea coasts, whereas the interior, while colder, also is a lot milder than areas elsewhere in the world on such northerly latitudes. Even during polar night in the north, temperatures above freezing are commonplace on the coastline. The maritime influence brings high rainfall and snowfall to some areas of the country."}, {'title': 'Tveitsund', 'paragraph_text': 'Tveitsund is a village in Nissedal municipality, Norway. The urban area Tveitsund, which consists of Tveitsund and Treungen, has a population of 361.'}], 'question_decomposition': [{'id': 269805, 'question': 'Nissedal >> country', 'answer': 'Norway', 'paragraph_support_idx': 10}, {'id': 135710, 'question': 'The #1 was named for whom?', 'answer': 'north', 'paragraph_support_idx': 6}], 'source': 'musique'}
````

### 续训练

````bash
python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples.jsonl \
  --batch_size 1024

python embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /home/sdu/zhu/models/qwen-embedding-0.6B/ \
  --dataset_type 2wiki \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples.jsonl \
  --batch_size 1024

nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 18000 --path_attn  --N 9999999   --model_dir_to_resume ./train/hotpot_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_12999 \
  --model_save_dir ./train/musique_2hop_v1 >> train_musique_2hop_v1.log 2>&1 &


# 发现随着训练似乎是过拟合，精度反而下降

nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 13300 --path_attn  --N 9999999   --model_dir_to_resume ./train/hotpot_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_12999 \
  --model_save_dir ./train/musique_2hop_v1.1 >> train_musique_2hop_v1.1.log 2>&1 &

````

### 推理

````bash

nohup python eval.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299_encoder/encoder.pt  \
    --model_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop \
    --test_dataset test_datasets_triples.jsonl \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_musique_2hop_1.1.log 2>&1 &

nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299_encoder/encoder.pt  \
    --model_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop \
    --test_dataset test_datasets_triples.jsonl \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_musique_2hop_1.1.log 2>&1 &

# 关闭path_attn
nohup python eval.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299_encoder/encoder.pt  \
    --model_dir ./train/musique_2hop_v1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_13299 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop \
    --test_dataset test_datasets_triples.jsonl \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --save_dir ./gen_tmp >> eval_musique_2hop_1.1.log 2>&1 &

nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_prepare.json     --dataset-type musique     --model-path /home/sdu/zhu/models/llama3_8B_instruct     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_musique_2hop_1.1.log 2>&1 &
````


# Olmo3模型
## RAG性能测试

````bash
# 替换kblam成为默认执行环境
conda activate kblam-rag

nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_prepare.json     --dataset-type musique     --model-path /home/sdu/zhu/models/olmo3-7b/     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_olmo3-7b_musique_2hop.log 2>&1 &

nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_prepare.json     --dataset-type musique     --model-path /home/sdu/zhu/models/olmo3-32B-awq-8b/     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_olmo3-32b_musique_2hop.log 2>&1 &

# -------------- 使用官方推荐的8bit bytealone 量化方法，精度更好 

nohup python llama_rag_v3.py     --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_prepare.json     --dataset-type musique     --model-path /home/sdu/zhu/models/olmo3-32b/     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_olmo3-32b_musique_2hop.log 2>&1 &

````


## olmo3+KBLaM
````bash

# 0. olmo3无KB推理
python src/kblam/models/olmo3/test_phase_5_1.py

# 1. olmo3带KB推理，查看KB部分注意力
  # 打开kblam_olmo3_attention.py和kblam_injector.py中的日志输出
python src/kblam/models/olmo3/test_phase_6.py


# squad数据集
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/squad/v4/train_datasets.json --dataset_type synthetic \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/squad/v4/test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000   --N 9999999 \
  --model_save_dir ./train/olmo-squad >> train_olmo3_squad.log 2>&1 &


# Musique, 推理精度不提高(kb_size 10~100, dataset size 10000)
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 100 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 5000 --path_attn  --N 9999999 \
  --model_save_dir ./train/olmo3_musique_2hop_v0 >> train_olmo3_musique_2hop_v0.log 2>&1 &

# 先用2wiki预训练
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 100 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets.json --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets.json \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000   --N 9999999 --path_attn \
  --model_save_dir ./train/olmo3_2wiki_1.2 \
  >> train_olmo3_2wiki_1.2.log 2>&1 &

# 再训练hotpot_2hop 
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/train_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 10000 --path_attn  --N 9999999   --model_dir_to_resume ./train/olmo3_2wiki_1.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_olmo3_step_5600 \
  --model_save_dir ./train/olmo3_hotpot_2hop_v1 >> train_olmo3_hotpot_2hop_v1.log 2>&1 &

# 再用2wiki预训练后的模型，用musique_2hop数据集继续预训练
nohup python train.py \
  --seed 1 --B 10  --lr 5e-4 \
  --sep_query_head --use_cached_embd --use_lr_decay  --save_period 500 --keep_old_checkpoints --duplicate_true_kb \
  --dynamic_kb_size 10 50 --outlier_num -999999 \
  --encoder_spec qwen-embedding-0.6B --key_embd_src key \
  --train_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples.jsonl --dataset_type 2wiki \
  --train_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --train_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/train_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --hf_model_spec /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
  --gradient_accm_step 20  --kb_token_layer_frequency 3 \
  --verbose \
  --test_data_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples.jsonl \
  --test_precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --test_precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 10000 --path_attn  --N 9999999   --model_dir_to_resume ./train/olmo3_2wiki_1.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_olmo3_step_5600 \
  --model_save_dir ./train/olmo3_musique_2hop_v1 >> train_olmo3_musique_2hop_v1.log 2>&1 &

````

## 推理
````bash

# olmo3_musique_2hop_v1: 2wiki+Musique_2hop
nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/olmo3_musique_2hop_v1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_olmo3_step_9999_encoder/encoder.pt  \
    --model_dir ./train/olmo3_musique_2hop_v1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_olmo3_step_9999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop \
    --test_dataset test_datasets_triples.jsonl \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_olmo3_musique_2hop_v1.log 2>&1 &

# 精度比训练时候还下降一截
#---------------------------------------------------------------------------
# 回退一点，先测试olmo3_2wiki的推理精度

# 使用测试环境（torch更高）给BERT用
# conda activate kblam-rag


nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/olmo3_2wiki_1.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_olmo3_step_5600_encoder/encoder.pt \
    --model_dir ./train/olmo3_2wiki_1.2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_olmo3_step_5600 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_olmo3_2wiki_1.2.log 2>&1 &

# √√√： 发现是模型装载问题，先初始化olmo3模型，再替换attention（加入seperate query head），但是没有装载对应头参数。使用专门的load_kblam_olmo3_model之后推理精度回归


# 同样2wiki数据集，测试没有“--path_attn”和RAG
nohup python llama_rag_v2.py    --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/merge_data/merged_dev.json     --dataset-type 2wiki     --model-path /home/sdu/zhu/models/olmo3-7b/     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_olmo3_2wiki_1.2.log 2>&1 &


# 测试squad数据集
nohup python llama_rag_v2.py     --dataset-path /mnt/n0/datasets/squad/plain_text/validation_merged.json     --dataset-type squad     --model-path /home/sdu/zhu/models/olmo3-7b/     --embedding-model sentence-transformers/all-MiniLM-L6-v2     --n-samples 100 --kb-size 100     --similarity-top-k 10 >> eval_olmo3_squad.log 2>&1 &

nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/olmo3-7b/ --llm_type olmo3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir  ./train/olmo-squad/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_olmo3_step_4000_encoder/encoder.pt \
    --model_dir  ./train/olmo-squad/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_olmo3_step_4000 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/v4/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type squad --query_size 100 --seed 1 --path_attn --save_dir ./gen_tmp >> eval_olmo3_squad.log 2>&1 &
````




# 向量检索

````bash
conda activate kblam-rag


nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/squad_2.7_stage_2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/squad_2.7_stage_2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/v4/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --query_size 100 --seed 1 >> eval_squad2.7.log 2>&1 &

nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/squad_2.7_stage_2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/squad_2.7_stage_2/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_synthetic_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/v4/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/v4/test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --hnsw_index_path /mnt/n0/datasets/squad/v4/hnsw \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --query_size 100 --seed 1 >> eval_squad2.7.log 2>&1 &

# 单跳跑通，需要top-1+random-9
# 精度变化：0.81 -> 0.8

#-----------------------------------------------------------------------------

# 多跳，2wiki
## 使用纯向量检索
nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --dataset_type 2wiki --query_size 100 --seed 1 --save_dir ./gen_tmp --path_attn >> eval_2wiki_1.1.log 2>&1 &

nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_hnsw \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --dataset_type 2wiki --query_size 100 --seed 1 --save_dir ./gen_tmp --path_attn >> eval_2wiki_1.1.log 2>&1 &

# 精度变化： 0.84 -> 0.83

### 多跳检索的召回率分析
nohup python eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_hnsw \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --dataset_type 2wiki --query_size 100 --seed 1 --save_dir ./gen_tmp --path_attn >> eval_2wiki_1.1.log 2>&1 &

##------------------------------------------------
# 性能分析：
 # 单条操作，retrieve_top时间：0.5ms，embedding时间: 60ms (cpu), 17ms (cuda)
 # 批量操作， Embedding： 5000/7s=1.3ms (批量+cuda)， retrieve： 5000/0.14s=0.028ms. 精度不受影响。

# top-1, 命中率0.95， top-10，命中率0.99

## 实现两种重排算法，轻量重排，语义重排
nohup python eval_generation.py debug \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999_encoder/encoder.pt \
    --model_dir ./train/2wiki_1.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_2wiki_llama3_step_7999 \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique \
    --test_dataset 2wiki_test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
    --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_hnsw \
    --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B \
    --dataset_type 2wiki --query_size 100 --seed 1 --path_attn >> eval_retrieve.log 2>&1 &


````


# 图增强的RAG检索
## 使用PathWeaver中的多跳检索

````bash
conda activate kblam-rag
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
export DASHSCOPE_API_KEY=sk-459cec30805e4538ac2c086a65d32b16

nohup python graph_rag.py \
  --n-samples 100 \
  --model-path /home/sdu/zhu/models/llama3_8B_instruct \
  --similarity-top-k 10 \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/graph_rag \
  --test_dataset 2wiki.json \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/2wiki_hnsw \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B >> graph_rag.log 2>&1 &

nohup python graph_rag.py \
  --n-samples 100 \
  --model-path /home/sdu/zhu/models/llama3_8B_instruct \
  --similarity-top-k 10 \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/graph_rag \
  --test_dataset hotpot_2hop.json \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/hotpot_2hop/test_datasets_qwen-embedding-0.6B_embd_value.npy \
  --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/hotpot_2hop_hnsw \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B >> graph_rag.log 2>&1 &

nohup python graph_rag.py \
  --n-samples 100 \
  --model-path /home/sdu/zhu/models/llama3_8B_instruct \
  --similarity-top-k 10 \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/graph_rag \
  --test_dataset musique_2hop.json \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/musique_2hop/test_datasets_triples_qwen-embedding-0.6B_embd_value.npy \
  --hnsw_index_path /mnt/n0/datasets/wiki_hotspot_musique/musique_2hop_hnsw \
  --base_embeder_path /home/sdu/zhu/models/qwen-embedding-0.6B >> graph_rag.log 2>&1 &

````