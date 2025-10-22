
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
````
生成位于dataset_path同目录下的新文件，文件名包含"triples"


## 生成训练用数据集
````bash
python datasets_gen.py -t musique -p1 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl -p2 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train_triple_llama3_8B_instruct_inst6_num_sample6000.json -p3 ../datasets/musique_train_6000/train_datasets.json
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
````
功能：
    - 生成KB-Embedding，存储在dataset_path同目录下
    - 在原始数据集中添加每个样本、每个段落的三元组偏移量（训练时候需要）

# 训练

训练脚本：experiments/train.py

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
  --model_save_dir ./train/musique_kbsize1_6000 \
  --gradient_accm_step 10 \
  --save_period 100 \
  --total_steps 6000 \
  --kb_size 1 \
  --verbose > train_musique_kbsize1_v4.log 2>&1 &
````

关键参数：
    - dataset_dir: 存储训练数据集和KB-Embedding的目录
    - dataset_type: 数据集类型
    - kb_size: 在训练时如何包含KB Tokens, 例如，在数据集类型为musique时，kb_size=-1，包含当前样本的所有段落的KB Tokens；kb_size=1，包含当前样本中与答案相关的段落的KB Tokens

