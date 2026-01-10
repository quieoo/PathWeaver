import os
import torch
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate

# --------------------------
# 100%本地配置（无OpenAI依赖）
# --------------------------
MODEL_PATH = "/home/sdu/zhu/models/llama3_8B_instruct/"  # 你的本地Llama3模型路径
EMBED_MODEL_NAME = "BAAI/bge-small-en"  # 本地轻量英文嵌入模型
DEVICE = "cpu"  # 有GPU改为"cuda"
CONTEXT_WINDOW = 4096
MAX_NEW_TOKENS = 512

# 1. 全局配置：替换默认的OpenAI嵌入/LLM为本地模型
# 1.1 本地嵌入模型（无API，纯本地）
local_embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_NAME,
    # model_kwargs={"device": DEVICE},
    # encode_kwargs={"normalize_embeddings": True},
)
Settings.embed_model = local_embed_model
Settings.chunk_size = 512

# 1.2 本地Llama3-8B-Instruct LLM（适配官方模板）
def init_llama3_llm():
    # 加载Tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        # CPU可选：开启8位量化降低内存占用
        # load_in_8bit=True if DEVICE == "cpu" else False,
    )
    
    # Llama3官方Prompt模板（无需额外包）
    llama3_template = PromptTemplate(
        template=(
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{query_str}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    )
    
    # 初始化LLM（core内置，无需question-gen-openai）
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=CONTEXT_WINDOW,
        max_new_tokens=MAX_NEW_TOKENS,
        system_prompt=(
            "You are a professional multi-hop reasoning assistant. "
            "Break down complex questions into clear sub-questions, answer each step accurately, "
            "and synthesize a concise final answer with logical consistency."
        ),
        query_wrapper_prompt=llama3_template,
        generate_kwargs={
            "temperature": 0.0,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        },
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True, "padding_side": "right"},
    )
    Settings.llm = llm
    return llm

# 2. 示例英文文档（多跳推理数据源）
def get_english_documents():
    return [
        Document(text="Alice is an engineer in the R&D Department. Her colleague is Bob."),
        Document(text="Bob's direct supervisor is Charlie, who oversees the entire R&D Department."),
        Document(text="Charlie's direct supervisor is David, who serves as the company's CTO and manages all technical departments."),
        Document(text="All employees in the R&D Department report to Charlie, and Charlie reports to David, the CTO.")
    ]

# 3. 主流程：纯本地子问题分解多跳问答
if __name__ == "__main__":
    # 加载本地Llama3模型
    print("Loading Llama3-8B-Instruct (local only)...")
    llm = init_llama3_llm()
    
    # 构建本地向量索引（无OpenAI Embedding）
    documents = get_english_documents()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=local_embed_model,
        llm=llm,
        show_progress=True
    )
    
    # 封装查询工具（core内置，无需额外包）
    query_tool = QueryEngineTool(
        query_engine=index.as_query_engine(llm=llm),
        metadata=ToolMetadata(
            name="company_org_tool",
            description="Query company organizational structure (employee relationships, hierarchy, job titles)"
        )
    )
    
    # 核心：子问题分解引擎（core内置，无需question-gen-openai）
    sub_q_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[query_tool],
        llm=llm,
        use_async=False,
        verbose=True,  # 打印拆解的子问题，便于调试
    )
    
    # 测试多跳问答
    print("\n===== Local Llama3 Multi-Hop QA (No OpenAI) =====")
    # 2跳问题
    q1 = "Who is the direct supervisor of Alice's colleague?"
    print(f"\nQ1: {q1}")
    print(f"A1: {sub_q_engine.query(q1)}")
    
    # 3跳问题
    q2 = "Who is the supervisor of Alice's colleague's supervisor? What is their job title?"
    print(f"\nQ2: {q2}")
    print(f"A2: {sub_q_engine.query(q2)}")