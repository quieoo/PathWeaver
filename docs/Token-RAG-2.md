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
    --eval_mode kb --kb_size=10 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/debug_squad/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4999_encoder/encoder.pt  \
    --model_dir ./train/debug_squad/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4999  \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed 1 >> eval_squad.log 2>&1 &



nohup python eval.py generation \
    --eval_mode kb --kb_size=10 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_4999_encoder/encoder.pt \
    --model_dir ./train/squad-2/stage1_lr_0.0005KBTokenLayerFreq1UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_all-MiniLM-L6-v2_squad_llama3_step_4999 \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir /mnt/n0/datasets/squad/ \
    --test_dataset test_datasets.json \
    --precomputed_embed_keys_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path /mnt/n0/datasets/squad/squad_test_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 100 --seed 1 --save_dir "./gen_tmp" >> eval_squad.log 2>&1 &


````

## squad训练性能优化-1

### 训练时随机化正确键位置
在训练时，为了增加模型的鲁棒性，我们可以在每个批次中随机化正确键的位置。这样可以防止模型依赖于键的固定位置，提高其泛化能力。



### 评估与解码对齐“短答案”

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

### 放大短答案的监督强度
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


### 把答案定位的掩码做稳
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


## Squad数据集上的语义对齐

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

## squad的刻度问题

不同的kb_scale_factor对结果的影响
````bash
nohup python eval.py generation \
    --eval_mode kb --kb_size=10 \
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
    --eval_mode kb --kb_size=2 \
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



## squad数据集重构

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