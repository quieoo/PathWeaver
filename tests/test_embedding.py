from openai import OpenAI
import os
import time

os.environ["DASHSCOPE_API_KEY"] = "sk-459cec30805e4538ac2c086a65d32b16"

client = OpenAI(
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 以下是北京地域base-url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

start_time=time.time()
resp = client.embeddings.create(
    model="text-embedding-v4",
    input=["喜欢，以后还来这里买"] * 10,
    # 将向量维度设置为 256
    dimensions=1024
)
end_time=time.time()
print(f"单次请求耗时: {end_time-start_time}秒")
print(f"批量推理向量维度: {len(resp.data[0].embedding)}")

gpt=
