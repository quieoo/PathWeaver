import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_local_llama3_model():
    """
    从本地路径加载Llama3模型并执行推理的传统方法
    """
    # 本地模型路径
    local_model_path = "/mnt/n0/models/llama3_8B_instruct"
    
    # 检查模型路径是否存在
    if not os.path.exists(local_model_path):
        print(f"模型路径 {local_model_path} 不存在，请确认路径正确")
        return
    
    print(f"正在从本地路径加载模型: {local_model_path}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        print("模型加载成功!")
        print(f"模型类型: {type(model)}")
        
        # 准备测试输入
        test_input = "Hello, how are you today?"
        print(f"\n测试输入: {test_input}")
        
        # 编码输入
        inputs = tokenizer(test_input, return_tensors="pt", padding=True).to(model.device)
        
        # 执行推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成文本: {generated_text}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"加载或执行模型时出错: {e}")
        raise e

def test_conversational_model():
    """
    测试对话模式下的模型推理
    """
    # 本地模型路径
    local_model_path = "/mnt/n0/models/llama3_8B_instruct"
    
    # 检查模型路径是否存在
    if not os.path.exists(local_model_path):
        print(f"模型路径 {local_model_path} 不存在，请确认路径正确")
        return
    
    print(f"正在从本地路径加载模型: {local_model_path}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        print("模型加载成功!")
        
        # 构造对话格式的输入 (Llama3的对话模板)
        messages = [
            {"role": "user", "content": "Hello, how are you today?"},
        ]
        
        # 应用聊天模板
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"\n格式化输入: {input_text}")
        
        # 编码输入
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        
        # 执行推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成文本: {generated_text}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"加载或执行模型时出错: {e}")
        raise e

if __name__ == "__main__":
    print("=== 测试本地Llama3模型加载和执行 (传统方法) ===")
    # 运行基本测试
    test_local_llama3_model()
    
    print("\n=== 测试对话模式下的推理 ===")
    # 运行对话模式测试
    test_conversational_model()