import os
import torch
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------
# 核心配置：替换为你的本地模型路径/名称
# --------------------------
MODEL_NAME = "/home/sdu/zhu/models/llama3_8B_instruct/"  # 本地模型路径
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Hugging Face Hub模型名
DEVICE = "cpu"  # CPU运行；有GPU改为"cuda"
CONTEXT_WINDOW = 2048
MAX_NEW_TOKENS = 512

# 1. 初始化本地Hugging Face LLM（修复torch_dtype警告）
def init_huggingface_llm():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    # 加载模型（修复torch_dtype弃用警告）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,  # 替换torch_dtype为dtype
        device_map=DEVICE
    )
    
    # 封装为LlamaIndex兼容的LLM
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=CONTEXT_WINDOW,
        max_new_tokens=MAX_NEW_TOKENS,
        system_prompt="你是一个专业的多跳推理助手，基于给定的知识图谱回答问题，步骤清晰、准确。",
        generate_kwargs={
            "temperature": 0.0,
            "do_sample": False
        }
    )
    return llm

# 2. 构建知识图谱三元组数据
graph_data = [
    ("Alice", "同事", "Bob"),
    ("Bob", "直属上级", "Charlie"),
    ("Charlie", "部门", "研发部"),
    ("Bob", "部门", "研发部"),
    ("Alice", "部门", "研发部"),
    ("Charlie", "直属上级", "David"),
    ("David", "职位", "CTO")
]

# 3. 初始化图存储和LLM（适配新版本API）
graph_store = SimpleGraphStore()
# 新版本需要先创建StorageContext
storage_context = StorageContext.from_defaults(graph_store=graph_store)
llm = init_huggingface_llm()

# 4. 创建图索引（修复from_triples不存在的问题）
# 新版本中KnowledgeGraphIndex.from_triples需指定storage_context
kg_index = KnowledgeGraphIndex.from_triples(
    graph_data,
    llm=llm,
    storage_context=storage_context,  # 新增：必须指定storage_context
    show_progress=True,
    include_embeddings=False
)

# 5. 创建多跳问答引擎
multi_hop_engine = MultiStepQueryEngine(
    index=kg_index,
    llm=llm,
    max_steps=3
)

# 6. 执行多跳查询
if __name__ == "__main__":
    # 测试1：2跳问题
    query1 = "谁是Alice的同事的直属上级？"
    response1 = multi_hop_engine.query(query1)
    print(f"问题1：{query1}")
    print(f"回答1：{response1}\n")

    # 测试2：3跳问题
    query2 = "Alice的同事的老板的老板是谁？他的职位是什么？"
    response2 = multi_hop_engine.query(query2)
    print(f"问题2：{query2}")
    print(f"回答2：{response2}")