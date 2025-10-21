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


python tools/1.embedding.py \
    --dataset_name musique_train \
    --model_name all-MiniLM-L6-v2 \
    --dataset_path ./datasets/musique/train_6000.json
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



nohup python train.py \
  --seed 1 --N 1160 --B 10  --lr 5e-4 --total_steps 4800 \
  --sep_query_head --use_cached_embd --use_lr_decay \
  --dynamic_kb_size 10 100 --kb_token_layer_frequency 1 \
  --encoder_spec all-MiniLM-L6-v2 --key_embd_src key \
  --dataset_dir ../datasets/ --train_dataset multi_wiki_qa_train \
  --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
  --model_save_dir ./train/multi_wiki_qa \
  --gradient_accm_step 10 --save_period 100 \
  --verbose \
  > train.log 2>&1 &

# train musique
python train.py   --seed 1 --B 1  --lr 5e-4 --total_steps 4800   --sep_query_head --use_cached_embd --use_lr_decay   --kb_size -1 --kb_token_layer_frequency 1   --encoder_spec all-MiniLM-L6-v2 --key_embd_src key   --dataset_dir ../datasets/ --train_dataset musique_6000   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3   --model_save_dir ./train/musique   --gradient_accm_step 10 --save_period 1000   --verbose

export CUDA_VISIBLE_DEVICES=3
nohup python train.py   --seed 1 --B 1  --lr 5e-4 --total_steps 4800   --sep_query_head --use_cached_embd --use_lr_decay   --kb_size 1 --kb_token_layer_frequency 1   --encoder_spec all-MiniLM-L6-v2 --key_embd_src key   --dataset_dir ../datasets/ --train_dataset musique_6000   --hf_model_spec /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3   --model_save_dir ./train/musique_kbsize1   --gradient_accm_step 10 --save_period 1000   --verbose > train_musique_kbsize1.log 2>&1 &
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
# 运行似乎需要开启代理:Downloading package wordnet to /home/sdu/nltk_data...
# 新增参数 query_size, seed (-1代表不随机)
python eval.py generation \
    --eval_mode kb --kb_size=100 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700_encoder/encoder.pt \
    --model_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700 \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir ../datasets/synthetic_embd \
    --test_dataset test_synthetic.json \
    --precomputed_embed_keys_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 10 --seed -1 \
    --save_dir ./gen_output1_kbscale1
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
        
        
        
# 准确率分析
````bash
# 新增参数 query_size=10, seed (-1代表不随机)
python eval.py generation \
    --eval_mode kb --kb_size=20 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700_encoder/encoder.pt \
    --model_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700 \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir ../datasets/synthetic_embd \
    --test_dataset test_synthetic.json \
    --precomputed_embed_keys_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 10 --seed -1 \
    --save_dir ./gen_output_kbscale1
````
## seperate query head
结论：开启或者关闭query head分离，准确率基本无变化

开启query head分离之后，输出中会包含很多"sorry。。。"，如下：
````
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The description of Boltzmann's Bastion is a luxury yacht offering private charters and premium services.
-------
Model output: The objectives of Boltzmann's Bastion is provide comfortable and unique accommodations, promote sustainability, and attract guests.
True answer: The objectives of Boltzmann's Bastion is provide exceptional service, ensure guest comfort, and offer unique experiences.
-------
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The purpose of Boltzmann's Bastion is to create an unforgettable maritime adventure.
````

关闭query head分离之后，输出中虽然没有"sorry"，但是会输出一些与正确结果差异较大的结果，如下：
````
Model output:  The purpose of Boltzmann’s Bastion is a vibrant cultural hub.
True answer: The description of Boltzmann's Bastion is a luxury yacht offering private charters and premium services.
-------
Model output: The objectives of the Viking Harbor is to offer a rich maritime experience.
True answer: The objectives of Boltzmann's Bastion is provide exceptional service, ensure guest comfort, and offer unique experiences.
-------
Model output:  The objectives of the Heritage Site is to showcase the beauty of the Mediterranean.
True answer: The purpose of Boltzmann's Bastion is to create an unforgettable maritime adventure.
````

## kb scale factor

结论： 
0的话相当于关闭知识库，使用原始模型进行推理，因此会输出”can't find information"
1是允许输入的最小整数，相对的准确率最高
之后随着kbsf的值增大，准确率会下降，当达到10000的时候，输出基本变得没有意义
(不过实验使用的kb size为20，事实上随着kb size增大，整体精度都会下降，此时kbsf的影响会怎样暂时未知)
kbsf    f1_score    recall (1-sorry_ratio)
0         0.47       I don't have information
1         0.85       0.9
10        0.77       0.7
100       0.74       0.7
1000      0.71       0.6
10000     0.28       nonesense output


## unstructured answer
````bash
python eval.py generation \
    --eval_mode kb --kb_size=20 \
    --llm_base_dir /mnt/n0/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec all-MiniLM-L6-v2 \
    --encoder_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700_encoder/encoder.pt \
    --model_dir ./train/synthetic1/stage1_lr_0.0005KBTokenLayerFreq1MultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_4700 \
    --kb_layer_frequency 1 --kb_scale_factor 1 \
    --dataset_dir ../datasets/synthetic_embd \
    --test_dataset test_synthetic.json \
    --precomputed_embed_keys_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path ../datasets/synthetic_embd/test_synthetic_all-MiniLM-L6-v2_embd_value.npy \
    --query_size 10 --seed -1 \
    --save_dir ./gen_output_kbscale1_natural
````

结论：使用非结构化问题，准确率会轻微下降；输出结果仍旧有原来的结构化输出格式风格为主


使用原本的结构化的问题：
````
Question: What is the description of Boltzmann's Bastion?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The description of Boltzmann's Bastion is a luxury yacht offering private charters and premium services.
-------
Question: What is the objectives of Boltzmann's Bastion?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The objectives of Boltzmann's Bastion is provide exceptional service, ensure guest comfort, and offer unique experiences.
-------
Question: What is the purpose of Boltzmann's Bastion?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The purpose of Boltzmann's Bastion is to create an unforgettable maritime adventure.
-------
Question: What is the description of Ampère's Alcove?
Model output: The description of Ampère's Alcove is a cozy café with a variety of artisanal coffees and pastries.
True answer: The description of Ampère's Alcove is a cozy café with a focus on artisanal coffee and pastries.
-------
Question: What is the objectives of Ampère's Alcove?
Model output: The objectives of Ampère's Alcove is provide delicious food, support local vendors, and create a welcoming atmosphere.
True answer: The objectives of Ampère's Alcove is provide delicious food and drinks, ensure a welcoming atmosphere, and support local suppliers.
-------
Question: What is the purpose of Ampère's Alcove?
Model output: The purpose of Ampère's Alcove is to create a relaxing environment for guests to unwind and enjoy.
True answer: The purpose of Ampère's Alcove is to create a relaxing and enjoyable café experience.
-------
Question: What is the description of Gauss' Gallery?
Model output: The description of Gauss' Gallery is a high-end fashion boutique with exclusive designer collections.
True answer: The description of Gauss' Gallery is a high-end fashion boutique with exclusive designer collections.
-------
Question: What is the objectives of Gauss' Gallery?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The objectives of Gauss' Gallery is provide exceptional service, offer unique products, and ensure customer satisfaction.
-------
Question: What is the purpose of Gauss' Gallery?
Model output: The purpose of Gauss' Gallery is to cater to fashion enthusiasts and luxury shoppers.
True answer: The purpose of Gauss' Gallery is to cater to fashion enthusiasts and provide a premium shopping experience.
-------
Question: What is the description of Euler's Edifice?
Model output: The description of Euler's Edifice is a luxury resort with spa services and outdoor activities. Relax and recharge.
True answer: The description of Euler's Edifice is a luxury spa resort with a focus on wellness and relaxation.
-------

````
转换成非结构化问题之后：
````
Question: Can you tell me what Boltzmann’s Bastion is like?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The description of Boltzmann's Bastion is a luxury yacht offering private charters and premium services.
-------
Question: What are the main goals or objectives of Boltzmann’s Bastion?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The objectives of Boltzmann's Bastion is provide exceptional service, ensure guest comfort, and offer unique experiences.
-------
Question: What’s the main purpose of Boltzmann’s Bastion?
Model output: The main purpose of Boltzmann's Bastion is to offer a unique and adventurous experience.
True answer: The purpose of Boltzmann's Bastion is to create an unforgettable maritime adventure.
-------
Question: How would you describe Ampère’s Alcove?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The description of Ampère's Alcove is a cozy café with a focus on artisanal coffee and pastries.
-------
Question: What are the objectives or aims of Ampère’s Alcove?
Model output: The objectives of Ampère’s Alcove is provide delicious food, support local vendors, and create a welcoming atmosphere.
True answer: The objectives of Ampère's Alcove is provide delicious food and drinks, ensure a welcoming atmosphere, and support local suppliers.
-------
Question: What’s the purpose of Ampère’s Alcove?
Model output: The purpose of Ampère’s Alcove is to create a relaxing and enjoyable dining experience.
True answer: The purpose of Ampère's Alcove is to create a relaxing and enjoyable café experience.
-------
Question: What kind of place is Gauss’ Gallery?
Model output: The description of Gauss’ Gallery is a high-end fashion boutique with exclusive designer collections.
True answer: The description of Gauss' Gallery is a high-end fashion boutique with exclusive designer collections.
-------
Question: What are Gauss’ Gallery’s main objectives or goals?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The objectives of Gauss' Gallery is provide exceptional service, offer unique products, and ensure customer satisfaction.
-------
Question: What’s the main purpose of Gauss’ Gallery?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The purpose of Gauss' Gallery is to cater to fashion enthusiasts and provide a premium shopping experience.
-------
Question: How would you describe Euler’s Edifice?
Model output: I am sorry I cannot find relevant information in the KB.
True answer: The description of Euler's Edifice is a luxury spa resort with a focus on wellness and relaxation.
-------

````