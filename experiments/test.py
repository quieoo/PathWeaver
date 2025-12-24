import hnswlib
import numpy as np

# 假设你已有的知识库向量：
kb_size = 12072      # 知识库中有12072个向量
d = 384              # 每个嵌入向量的维度
kb_embeddings = np.random.rand(kb_size, d).astype(np.float32)

# ---------- Step 1: 构建 HNSW 索引 ----------
# 创建索引对象
p = hnswlib.Index(space='l2', dim=d)  # 'l2' 表示欧氏距离，也可以是 'cosine'

# 初始化索引
# 参数说明：
# - max_elements: 最大可容纳向量数
# - ef_construction: 构建时的搜索宽度（越大越准但建得越慢）
# - M: 每个节点的连接数（控制内存与性能的平衡）
p.init_index(max_elements=kb_size, ef_construction=200, M=16)

# 可选：设置运行时参数
p.set_num_threads(8)  # 使用多线程
p.set_ef(50)          # 查询时的搜索范围，越大越准但查询慢

# ---------- Step 2: 添加知识库向量 ----------
ids = np.arange(kb_size)  # 每个向量对应的唯一ID
p.add_items(kb_embeddings, ids)
print(f"HNSW 索引构建完成，共 {kb_size} 个知识向量。")

# ---------- Step 3: 保存索引（可选） ----------
p.save_index("kb_index.bin")

# ---------- Step 4: 查询 ----------
# 假设有一个查询向量 q
q = np.random.rand(d).astype(np.float32)

# 查询最相似的 top-k 向量
top_k = 5
labels, distances = p.knn_query(q, k=top_k)

print(f"查询结果ID: {labels}")
print(f"对应距离: {distances}")

# ---------- Step 5: 批量查询 ----------
queries = np.random.rand(10, d).astype(np.float32)  # 10个查询
labels, distances = p.knn_query(queries, k=top_k)
print(f"批量查询结果 shape: {labels.shape}")
