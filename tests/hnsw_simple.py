# test_hnsw_large.py
import hnswlib
import numpy as np
import time

# ---------- 参数 ----------
dim = 256
num_elements = 10_000_000          
k = 10
M = 32                            # 图出度，召回优先
ef_construction = 200             # 建库宽度
ef = 200                          # 查询宽度
q=1000
np.random.seed(42)


# ---------- 1. 模拟大数据集（流式 add，避免一次性占满内存） ----------
index = hnswlib.Index(space='l2', dim=dim)   # 用 L2 距离
index.init_index(max_elements=num_elements,
                 ef_construction=ef_construction,
                 M=M)
index.set_num_threads(0)          # 0 表示用所有 CPU 核

print(f'Start adding {num_elements} vectors (256D)...')
chunk_size=int(num_elements/100)

for start in range(0, num_elements, chunk_size):
    vectors = np.random.randn(chunk_size, dim).astype('float32')
    ids = np.arange(start, start + chunk_size, dtype=np.int32)
    index.add_items(vectors, ids)
    print(f'  added {(start + chunk_size) // 1000} k / {num_elements // 1000} k')
print('Index build complete.')

# ---------- 2. 查询参数 ----------
index.set_ef(ef)                  # 越大召回越高
query = np.random.randn(dim).astype('float32')


print('\nSingle query (top-10):')
t0 = time.time()
labels, distances = index.knn_query(query, k=k)
latency = (time.time() - t0) * 1000
print(f'Latency = {latency:.2f} ms')
print('IDs   :', labels[0])
print('Dist  :', ['%.3f' % d for d in distances[0]])

# ---------- 4. 可选：落盘 ----------
file_name= f'hnsw_{num_elements // 1000_000}m_{dim}d.bin'
index.save_index(file_name)
