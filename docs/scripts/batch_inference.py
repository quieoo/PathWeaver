import os
import subprocess

ckpt_dir = "/home/sdu/zhu/kblam/train/atfb_2wiki_v4/"
dataset_dicts = [
    {
        "id": "without_silver",
        "dir": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples",
        "name": "AT2QA_2wiki_test_2hop_compositional_gold.json",
        "key_path": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_key.npy",
        "value_path": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold_qwen-embedding-0.6B_embd_value.npy",
    },
    {
        "id": "with_silver",
        "dir": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples",
        "name": "ATFB_2wiki_test_2hop_compositional_silver.json",
        "key_path": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_key.npy",
        "value_path": "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/ATFB_2wiki_test_2hop_compositional_silver_qwen-embedding-0.6B_embd_value.npy",
    }
]

t_step=8000

# 命令模版
command_template = """
python /mnt/n0/KBLAM/KBLaM/experiments/eval_generation.py generation \
    --kb_size=10 \
    --llm_base_dir /home/sdu/zhu/models/llama3_8B_instruct/ --llm_type llama3 \
    --encoder_spec qwen-embedding-0.6B \
    --encoder_dir {encoder_path} \
    --model_dir {model_path} \
    --kb_layer_frequency 3 --kb_scale_factor 1 \
    --dataset_dir {dataset_dir} \
    --test_dataset {dataset_name} \
    --precomputed_embed_keys_path {key_path} \
    --precomputed_embed_values_path {value_path} \
    --step {step} --t_step 8000 \
    --dataset_type at2qa_2wiki --query_size 100 --seed 1 --path_attn 
"""

# 遍历文件夹获得所有模型与encoder文件夹对
model_encoder_pairs = []
for item in os.listdir(ckpt_dir):
    if item.endswith("encoder"):
        encoder_path = os.path.join(ckpt_dir, item, "encoder.pt")   
        model_path = os.path.join(ckpt_dir, item.split("_encoder")[0])
        step=model_path.split("_step_")[-1]
        
        model_encoder_pairs.append((step, model_path, encoder_path))

# for step, model_path, encoder_path in model_encoder_pairs:
#     print(f"===== evaluate model with step {step} =====")
#     for dataset_dict in dataset_dicts:
#         dataset_id = dataset_dict["id"]
#         dataset_dir = dataset_dict["dir"]
#         dataset_name = dataset_dict["name"]
#         key_path = dataset_dict["key_path"]
#         value_path = dataset_dict["value_path"]
#         print(f"------- {dataset_id} -------")
        
#         command = command_template.format(encoder_path=encoder_path, model_path=model_path, dataset_dir=dataset_dir, dataset_name=dataset_name, key_path=key_path, value_path=value_path, step=step)
#         print(command)
#         try:
#             os.system(command)
#         except Exception as e:
#             print(f"执行命令失败，错误信息：{e}")
#             continue

print("####################################################")

# fixed_step_ratio=[0.3, 0.6, 1.0]
fixed_step_ratio=[0.4, 1.0]

fixed_step=[int(t_step * r) for r in fixed_step_ratio]
for step in fixed_step:
    print(f"===== Evaluate Model With Step {step} =====")
    for model_id, model_path, encoder_path in model_encoder_pairs:
        print(f"-------- Chechkpoint ID: {model_id} --------")
        for dataset_dict in dataset_dicts:
            dataset_id = dataset_dict["id"]
            dataset_dir = dataset_dict["dir"]
            dataset_name = dataset_dict["name"]
            key_path = dataset_dict["key_path"]
            value_path = dataset_dict["value_path"]
            print(f"------- {dataset_id} -------")
            
            command = command_template.format(encoder_path=encoder_path, model_path=model_path, dataset_dir=dataset_dir, dataset_name=dataset_name, key_path=key_path, value_path=value_path, step=step)
            print(command)


            # try:
            #     os.system(command)
            # except Exception as e:
            #     print(f"执行命令失败，错误信息：{e}")
            #     continue

            # 2. 执行命令并捕捉输出
            try:
                result = subprocess.run(
                    command,  # 要执行的命令
                    shell=True,  # 允许执行shell命令（支持管道、路径空格等）
                    capture_output=True,  # 捕捉stdout和stderr
                    text=True,  # 输出以字符串（str）格式返回，而非字节流（bytes）
                    encoding="utf-8",  # 指定编码，避免中文等特殊字符乱码
                    timeout=None  # 不设置超时（若命令执行时间长，可指定具体秒数）
                )
                
                # 3. 处理捕捉到的输出
                stdout_output = result.stdout  # 命令正常输出内容
                stderr_output = result.stderr  # 命令错误/警告输出内容
                return_code = result.returncode  # 命令退出码（0=成功，非0=失败）
                
                # 4. 打印输出（终端实时反馈）
                print("===== 命令执行正常输出 =====")
                print(stdout_output)
                if stderr_output:  # 只有当有错误输出时才打印
                    print("===== 命令执行错误/警告输出 =====")
                    print(stderr_output)
            except subprocess.TimeoutExpired:
                print(f"命令执行超时！")
            except Exception as e:
                print(f"执行命令失败，错误信息：{e}\n")
                continue