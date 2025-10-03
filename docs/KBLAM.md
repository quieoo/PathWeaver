-1. 环境准备
````bash
# 激活环境
conda activate kblam
# 限制GPU使用
export CUDA_VISIBLE_DEVICES=1
# 需要先登录wandb: wandb login
# 开启镜像all-MiniLM-L6-v2
export HF_ENDPOINT=https://hf-mirror.com
````


0. 生成三元组
    json文件（例如enron.json)，每条包含以下内容:
     name, description_type, description
     Q, A
     key_string

1. 生成KV embedding (generate_kb_embedding.py)
    输入：三元组文件，embedding_encoder
    输出：kv embedding文件

    例子:
````bash
# 使用在线模型
cd dataset_generation

python generate_kb_embeddings.py --dataset_name synthetic --dataset_path ../datasets/synthetic.json --output_path ../datasets/ --model_name text-embedding-v4 --api_key sk-459cec30805e4538ac2c086a65d32b16

# 使用本地模型
python generate_kb_embeddings.py \
    --model_name all-MiniLM-L6-v2 \
    --dataset_name synthetic --dataset_path ../datasets/synthetic.json \
    --output_path ../datasets/     
````

1.1 训练集/测试集分割
````bash
python create_train_test_split.py \
    --data_path ../datasets/synthetic.json \
    --embedding_keys_path ../datasets/synthetic_all-MiniLM-L6-v2_embd_key.npy \
    --embeddings_values_path ../datasets/synthetic_all-MiniLM-L6-v2_embd_value.npy \
    --output_path ../datasets/synthetic_embd \
    --split_index 120000

# 训练集大小：120000
# 测试集大小：12072
````


2. 训练Adapter (train.py)
    例如：
````bash

# 2000步大约需要5个小时，12个小时大约设置4800步
# B=20会带来训练时显存不足的问题，修改B为10
# 每隔100步保存一次(每次保存会将旧的checkpoint文件删除)
# 设置model_dir_to_resume从checkpoint恢复训练
nohup python train.py \
  --seed 1 --N 120000 --B 10  --lr 5e-4 --total_steps 4800 \
  --sep_query_head --use_cached_embd --use_data_aug --use_lr_decay --duplicate_true_kb \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 --outlier_num 2 --multi_entities 2 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --dataset_dir ../datasets/ --train_dataset synthetic \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/synthetic1 \
  --gradient_accm_step 10 --save_period 100 \
  --model_dir_to_resume ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4 \
  --verbose \
  > train.log 2>&1 &

# 差不多2400步时候收敛，到0.5左右的loss
# 最后生成两个文件夹：
    *_step_* 为模型保存的文件夹
    *_step_*_encoder 为encoder保存的文件夹
````
    参数：
        train_dataset: 训练数据集名称
        dataset_dir：数据集路径，数据集完整路径应该为 dataset_dir/train_dataset.json
        N: 从dataset中取前N条训练
        B：batch size
        lr：学习率
        use_lr_decay：是否使用学习率衰减，默认True
        sep_query_head：是否分离query head，默认True
        use_cached_embd：是否使用预先计算的kv embedding，默认True
        encoder_spec: 指定embedding encoder的名称或路径
        key_embd_src：默认'key'，表示使用key的embedding作为检索的embedding
        total_steps：训练总步数
        gradient_accm_step：梯度累积步数
        save_period：每隔多少步保存一次模型
        use_data_aug: 使用数据增强，默认True
        hf_model_spec：指定大语言模型的名称或路径(llama3-8b-instruct)
        llm_type：大语言模型类型，llama3或phi3
        model_save_dir：模型训练输出路径
        dynamic_kb_size：动态知识库大小范围，10-100默认
        duplicate_true_kb：未实现
        length_invariance：未实现
        outlier_num：outlier_num/batch_size，控制不存在于知识库中的问题比例
        multi_entities：多个相同Token的问题占比，multi_entities/batch_size
        use_extended_qa: 是否使用开放式问题，需要数据集支持
        kb_token_layer_frequency：知识库token插入频率

    输出：
        训练后的模型，adapter

3. 推理(eval.py)
    例如：
````bash
python eval.py generation  --dataset_dir ../datasets/synthetic_embd  --encoder_dir /home/sdu/KBLaM/KBLaM/experiments/train/synthetic/stage1_lr_0.0005KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_18000_encoder/encoder.pt  --encoder_spec all-MiniLM-L6-v2  --llm_base_dir /home/sdu/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B --llm_type llama3  --model_dir /home/sdu/KBLaM/KBLaM/experiments/train/synthetic/stage1_lr_0.0005KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_18000 --save_dir ./gen_output0  --seed 42  --test_dataset test_synthetic_augmented.json  --precomputed_embed_keys_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_key.npy  --precomputed_embed_values_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_value.npy  --eval_mode kb --kb_size=100

# 运行似乎需要开启代理:Downloading package wordnet to /home/sdu/nltk_data...

python eval.py generation \
    --eval_mode kb --kb_size=100 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700_encoder/encoder.pt \
    --model_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700 \
    --kb_layer_frequency 1 --kb_scale_factor 100 \
    --dataset_dir ../datasets/synthetic_embd \
    --test_dataset test_synthetic.json \
    --precomputed_embed_keys_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 10 \
    --save_dir ./gen_output0
````

    参数：
        dataset_dir: 数据集路径文件夹
        test_dataset：测试数据集文件名
        precomputed_embed_keys_path：K embedding 路径
        precomputed_embed_values_path：B embedding 路径
        kb_size: 使用的知识库大小
        seed: 随机种子选择知识库
        encoder_dir：训练好的adapter路径
        encoder_spec：embedding encoder名称或路径
        llm_base_dir：原始模型路径
        llm_type：大语言模型类型，llama3或phi3
        model_dir：训练好的模型路径
        query_head_path ：可选，分离query head时使用
        eval_mode: 评估模式，'kb'表示使用知识库
        kb_layer_frequency: 知识库token插入频率，和训练时保持一致
        kb_scale_factor： 知识库token缩放因子，设置值应该参考知识库的总大小(kb_size)
        multi_entites: 
        no_outlier：
        remove_sorry: 
        save_dir：结果保存路径（所有回答结果，benchmark结果文件）
        
        
        

    



