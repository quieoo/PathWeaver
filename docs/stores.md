# Quick Start Commands

## 建图
```bash
export CUDA_VISIBLE_DEVICES=0
cd /mnt/n0/PathWeaver

python tools/build_pathweaver_stores.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --dataset-id 2wiki-dev-v5 \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --hnsw-embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --kv-embedding-model /mnt/n0/models/qwen-embedding-0.6B/ \
  --hnsw-embedding-batch-size 1024 \
  --kv-embedding-batch-size 1024 \
  --kv-encoding-profile qwen3-embedding-v2 \
  --build-hnsw
```

## 离线检索
```bash
DATA_DIR=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set

/mnt/n0/uv_envs/kblam/bin/python tools/retrieve_pathweaver_dags.py \
  --input "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl" \
  --output "$DATA_DIR/store/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_store_dag_aa.jsonl" \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --model-ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st-model /mnt/n0/models/bge-en-v1.5/ \
  --entity-top-k 1 \
  --subgraph-hops 3 \
  --search-backend hnsw \
  --infer-batch-size 1024 \
  --topic-top-k 8 \
  --dde-hops 3 \
  --mention-bonus 0.2 \
  --seed-edge-topk 18 \
  --expansion-hops 2 \
  --per-src-cap 3 \
  --max-nodes 30 \
  --max-edges 40 \
  --max-sinks 3 \
  --answer-aware \
  --keep-score \
  --answerable-only \
  --reverse-sink-edge-topk 2 \
  --reverse-sink-hops 4 \
  --reverse-sink-beam-width 4 \
  --reference-output "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl"
```
输出

```
Answer recall: 0.8101
Graph  recall: 0.9325
None-sink recall: 0.8101
Answer + supported sink ratio: 0.7975
Sink relevance rank stats (merged across final sink counts):
  all_samples=192 answer_sink_samples=192 top-1=0.7396 top-2=0.8958 top-3=1.0000
[DONE] input=237 output=192 saved_to=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/store/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_store_dag_aa.jsonl
{
  "comparison": {
    "generated_samples": 192,
    "reference_samples": 208,
    "shared_samples": 188,
    "exact_dag_ratio": 0.0,
    "generated_nonempty_ratio": 1.0,
    "reference_nonempty_ratio": 1.0,
    "generated_answer_coverage": 1.0,
    "reference_answer_coverage": 1.0,
    "mean_generated_kv_nodes": 13.196808510638299,
    "mean_reference_kv_nodes": 12.063829787234043,
    "macro_kv_precision": 0.7816134116551021,
    "macro_kv_recall": 0.8275338604080121,
    "macro_kv_f1": 0.795896511369803
  }
}
```

# KVStore 与 GraphStore
、
第一版 Store 实现采用本地、可增量追加的知识管理方案：

- `KVStore` 为每个唯一的 key/value 对分配稳定的、从 0 开始的 offset。
  KV 文本和来源信息保存在 SQLite 中；基础 embedding 或最终 KV tensor
  可以在之后以 NumPy 数组的形式写入，并与 offset 严格对齐。
- `GraphStore` 保存规范化后的实体、别名、完整三元组、图邻接关系、
  数据来源，以及每条三元组对应的 KV offsets。
- 关系三元组的第三元作为实体；属性三元组的第三元作为 literal 图节点。
  literal 可以参与局部图恢复，但默认不参与实体向量检索。如果某个 literal
  后续出现在关系三元组的第一元或第三元中，它会被提升为实体。

## 构建 Store

使用经过三元组提取、但尚未经过 DAG-Retriever 剪枝的完整数据集构建 Store：

```bash
python tools/build_pathweaver_stores.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --dataset-id 2wiki-dev-v5 \
  --store-dir experiments/stores/2wiki-dev-v5
```

该命令是幂等的。重复执行不会为已有 KV 对分配新 offset，也不会重复创建
相同三元组。

向同一个 Store 追加其它完整三元组数据集时，保持 `--store-dir` 不变，
并为新数据集指定不同的 `--dataset-id`：

```bash
python tools/build_pathweaver_stores.py \
  --dataset-path /path/to/another_tripled_dataset.jsonl \
  --dataset-id another-dataset \
  --store-dir experiments/stores/2wiki-dev-v5
```

已有 KV offset 保持不变，新 KV 会从当前数组尾部继续分配 offset。

## 实体提取与消解

实体提取规则如下：

- `RELATION` 三元组：第一元和第三元作为实体。
- `ATTRIBUTE` 三元组：只有第一元作为实体，第三元作为 literal。
- 第二元始终作为关系或属性名称，不作为实体。

默认实体消解器执行确定性的字符串规范化，包括空白、标点、大小写和括号内容
规范化。也可以在首次导入实体前提供人工整理的 alias 映射：

```json
{
  "Maung Wunna": "Maung Wunna",
  "Wunna": "Maung Wunna"
}
```

构建时通过 `--alias-file` 指定：

```bash
python tools/build_pathweaver_stores.py \
  --dataset-path /path/to/triples.jsonl \
  --dataset-id another-dataset \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --alias-file /path/to/entity_aliases.json
```

alias 映射应在实体第一次进入 Store 前提供。第一版实现不会自动合并已经分别
存在于 Store 中的历史节点。

## 实体向量索引

实体 embedding 与明确的实体 `node_id` 对齐，不依赖 SQLite 的隐含行顺序。
使用以下命令生成实体 embedding，并可选构建 HNSW 索引：

```bash
python tools/build_pathweaver_stores.py \
  --dataset-path /path/to/triples.jsonl \
  --dataset-id another-dataset \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --hnsw-embedding-model /path/to/entity-retrieval-model \
  --build-hnsw
```

`--entity-embedding-model` 仍作为 `--hnsw-embedding-model` 的兼容别名保留。

当 HNSW 索引存在且环境安装了 `hnswlib` 时，
`GraphStore.search_entities()` 使用 HNSW 检索；否则使用同一组实体向量执行
精确余弦相似度检索。精确检索适合单元测试和较小的 Store。

新增实体，或者将 literal 提升为实体时，已有实体 embedding 和 HNSW 索引会
自动失效。扩展 GraphStore 后必须重新生成实体 embedding 和索引，避免新实体
被检索过程静默遗漏。

## 分离 HNSW 与 KV 编码模型

实体检索和 KV 编码可以使用两个完全不同的本地 embedding 模型：

```bash
python tools/build_pathweaver_stores.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --dataset-id 2wiki-dev-v5 \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --hnsw-embedding-model /path/to/entity-retrieval-model \
  --kv-embedding-model /path/to/kv-embedding-model \
  --hnsw-embedding-batch-size 256 \
  --kv-embedding-batch-size 128 \
  --kv-encoding-profile qwen3-embedding-v2 \
  --build-hnsw
```

两个参数的职责不同：

- `--hnsw-embedding-model` 只编码实体名称，输出
  `entity_vector_ids.npy` 和 `entity_vectors.npy`，并供 HNSW 建索引。
- `--kv-embedding-model` 按全局 KV offset 顺序分别编码 `key_text` 和
  `value_text`，输出 `key_tensors.npy` 和 `value_tensors.npy`。

如果本地模型要求特定的 SentenceTransformer prompt，可分别指定：

```bash
--hnsw-prompt-name passage \
--kv-prompt-name query
```

两个模型的目录和 prompt 会分别记录在：

```text
graph/entity_vectors.json
kv/tensor_metadata.json
```

HNSW 模型只影响实体召回，可以独立替换。KV 编码模型的输出会继续输入训练好的
`KBEncoder`，因此它必须与训练 adapter 时使用的 embedding 模型、维度和编码
方式保持一致。当前写入 KVStore 的是基础 key/value embedding，不是经过
`KBEncoder` 投影后的最终模型 KV cache tensor。

`--kv-encoding-profile` 默认是 `qwen3-embedding-v2`，用于对齐
`docs/scripts/embedding_v2.py --model_name qwen3-embedding-0.6B` 的主要编码路径：

- tokenizer 使用 left padding；
- 模型使用 `device_map="auto"` 加载；
- 模型移动到 CUDA（无 CUDA 时使用 CPU）；
- SentenceTransformer 第一层转换为 FP16；
- 未显式设置 `--kv-prompt-name` 时默认使用 `query`；
- 按 `key_texts + value_texts` 拼接后统一分批编码；
- OOM 时自动缩小 batch size；
- 最终向量归一化并保存为 FP32。

对于不需要上述 Qwen3 兼容行为的模型，可以改用：

```bash
--kv-encoding-profile sentence-transformer
```

## 写入 KV Tensor

按照 KV offset 顺序编码全部 KV 文本后，将 tensor 写入 KVStore：

```python
from kblam.stores import KVStore

with KVStore("experiments/stores/2wiki-dev-v5/kv") as store:
    records = list(store.iter_records())
    # 必须严格按照 records 中的 offset 顺序编码 key_text 和 value_text。
    store.write_tensors(key_tensors, value_tensors)
```

追加新数据集后，只需编码新分配的 offsets，并将对应的新行传给
`append_tensors()`。已有 offset 及其 tensor 不会发生变化。

## 检查 Store

使用以下命令在终端打印摘要，并生成完整 JSON 报告：

```bash
python tools/inspect_pathweaver_store.py \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --output experiments/stores/2wiki-dev-v5/report.json \
  --top-k 20 \
  --max-hops 2
```

报告包括：

- 图规模及三元组去重情况；
- 只考虑关系边时的实体度数和连通性；
- 高频关系和属性；
- 实体消解冲突及可疑实体；
- KV offset 完整性、越界引用和孤立 KV；
- 每个数据集的 sample、triple 和 KV 来源统计；
- 实体 embedding 与 HNSW 索引覆盖情况；
- 1-hop 和 2-hop 局部候选图的实体、三元组及 KV 规模。

局部图膨胀统计默认最多抽样 1,000 个实体。使用 `--max-seeds 0` 可以检查
全部实体：

```bash
python tools/inspect_pathweaver_store.py \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --max-hops 2 \
  --max-seeds 0

python tools/inspect_pathweaver_store.py \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --max-seeds 0

python tools/inspect_pathweaver_store.py \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --max-seeds 0 \
  --top-k 50 \
  --output experiments/stores/2wiki-dev-v5/report.json
```

## 从 Store 在线恢复 DAG

`tools/retrieve_pathweaver_dags.py` 实现以下查询链路：

1. 使用构建实体索引时相同的 embedding 模型编码问题；
2. 通过 HNSW 定位 top-k 个实体；
3. 分别从每个实体执行指定 hop 数的局部图扩展，再按 `triple_id` 合并去重；
4. 根据三元组关联的 KV offsets 从 KVStore 恢复原始 key/value 文本；
5. 将候选图交给 `DAG_KV_SubgraphRAG_trainable_v5_2.py` 的原始模型加载、
   特征构造、边和终点打分、DAG 剪枝及导出函数；
6. 保留输入 tripled 样本的原始字段，仅新增 `dag` 字段。

完整测试命令如下：

```bash
DATA_DIR=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set

python tools/retrieve_pathweaver_dags.py \
  --input "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl" \
  --output "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_store_dag_aa.jsonl" \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --model-ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st-model /mnt/n0/models/bge-en-v1.5/ \
  --entity-top-k 1 \
  --subgraph-hops 2 \
  --infer-batch-size 1024 \
  --topic-top-k 8 \
  --dde-hops 3 \
  --mention-bonus 0.2 \
  --seed-edge-topk 18 \
  --expansion-hops 2 \
  --per-src-cap 3 \
  --max-nodes 30 \
  --max-edges 40 \
  --max-sinks 3 \
  --answer-aware \
  --keep-score \
  --answerable-only \
  --reverse-sink-edge-topk 2 \
  --reverse-sink-hops 4 \
  --reverse-sink-beam-width 4 \
  --reference-output "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl"
```

`--entity-top-k` 和 `--topic-top-k` 属于不同阶段：前者控制全局实体索引返回多少个
seed，默认是 1；后者保留原 DAG 提取器的含义，在已经恢复的候选子图内选择 topic
节点。`--subgraph-hops` 控制每个 seed 的 GraphStore 扩展半径；可以通过
`--max-triples-per-seed` 单独限制每张局部子图的规模。

默认 `--search-backend auto`：安装 `hnswlib` 且索引文件存在时使用 HNSW，否则
回退到精确余弦检索。`--search-backend hnsw` 会强制使用 HNSW，并在依赖或索引缺失
时立即报错，适合确认正式实验没有静默回退。

工具默认在 `dag.meta.retrieval` 中记录 seed 实体名称、全局 node ID、检索分数、
候选节点数和三元组数。使用 `--no-retrieval-meta` 可以输出与旧脚本更接近的字段。
提供 `--reference-output` 后会按样本 ID 对比旧 `_dag_aa`，报告完整 DAG 相同率、
非空图比例以及 KV 集合的 macro precision、recall 和 F1。

`--answer-aware` 和 `--answerable-only` 会读取评测文件中的标准答案，仅适用于离线
复现实验。真实在线 query 没有标准答案时应去掉这两个参数；其余 DAG 打分和剪枝
逻辑保持不变。

旧 DAG 脚本中基于特定 2Wiki 输入文件名自动开启 `--local_propogation` 的实验遗留
逻辑已经移除。现在旧脚本和新工具都只在显式传入该参数时进入完整图无环导出路径；
普通 `--mode infer` 会真正加载并调用模型 checkpoint。

## 实体 Top-K 与子图 Hop 扫描

使用 2Wiki dev 的 237 个问题扫描 `entity_top_k={1,2,4,8}` 和
`subgraph_hops={1,2,3}`：

```bash
/mnt/n0/uv_envs/kblam/bin/python tools/benchmark_pathweaver_retrieval.py \
  --input "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl" \
  --store-dir experiments/stores/2wiki-dev-v5 \
  --st-model /mnt/n0/models/bge-en-v1.5/ \
  --entity-top-k-values 1,2,4,8 \
  --subgraph-hop-values 1,2,3 \
  --search-backend hnsw \
  --query-batch-size 128 \
  --warmup-queries 10 \
  --output experiments/stores/2wiki-dev-v5/retrieval_sweep.json
```

测试使用 `/mnt/n0/uv_envs/kblam` 中的 `hnswlib`。每组参数先预热 10 个 query，
下表采用第二次完整运行的数据；两次运行的答案命中数和候选图规模完全一致。
候选图 recall 表示在 DAG 剪枝之前，候选 KV 的 value 是否包含标准答案。
延迟包含 HNSW 查询、GraphStore 扩展和 KVStore 文本回读，不包含 query embedding
和后续 MLP DAG 提取。

| Entity top-k | Hops | Answer recall | 节点 mean/p95 | 三元组 mean/p95 | KV mean/p95 | 延迟 mean/p95/p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.59% | 9.2 / 18.0 | 9.2 / 17.2 | 18.5 / 36.0 | 0.89 / 1.26 / 1.55 |
| 1 | 2 | 88.61% | 21.9 / 37.0 | 24.5 / 43.0 | 49.2 / 86.0 | 1.74 / 2.68 / 3.33 |
| 1 | 3 | 93.25% | 39.0 / 104.2 | 45.5 / 115.4 | 91.2 / 230.8 | 2.86 / 6.15 / 10.62 |
| 2 | 1 | 23.63% | 13.4 / 26.0 | 13.5 / 30.2 | 27.3 / 60.4 | 1.12 / 1.83 / 2.18 |
| 2 | 2 | 91.14% | 32.9 / 56.2 | 36.8 / 66.0 | 73.9 / 132.0 | 2.55 / 4.08 / 4.62 |
| 2 | 3 | 94.51% | 59.1 / 129.8 | 68.9 / 159.0 | 138.0 / 318.0 | 4.51 / 9.18 / 13.40 |
| 4 | 1 | 35.86% | 21.3 / 42.0 | 21.1 / 43.0 | 42.5 / 86.0 | 1.51 / 2.42 / 2.77 |
| 4 | 2 | 92.41% | 56.1 / 97.0 | 62.9 / 109.4 | 126.0 / 218.8 | 4.19 / 6.55 / 7.17 |
| 4 | 3 | 94.94% | 102.5 / 191.8 | 120.6 / 230.0 | 241.6 / 460.0 | 8.03 / 14.70 / 19.09 |
| 8 | 1 | 44.30% | 37.6 / 65.2 | 36.8 / 69.6 | 73.8 / 139.2 | 2.27 / 3.65 / 4.35 |
| 8 | 2 | 94.09% | 101.9 / 151.2 | 114.7 / 174.2 | 229.7 / 348.4 | 8.06 / 11.04 / 13.15 |
| 8 | 3 | 95.78% | 184.7 / 313.0 | 219.2 / 382.4 | 439.0 / 764.8 | 15.00 / 25.22 / 30.11 |

Query embedding 在 CPU 上以 batch size 128 编码，总计约 2.16 秒，摊销约
9.09 ms/query。该时间对所有参数组合相同，因此没有计入表中的 retrieval latency。
真实单请求 embedding 延迟不能直接用此批处理摊销值代替。

结果表明：

- `hops=1` 不适合该多跳数据集。即使 top-k 增加到 8，候选答案召回仍只有
  44.30%；从 1-hop 增加到 2-hop 才是主要召回增益。
- `(top_k=1, hops=2)` 是低开销配置：召回 88.61%，平均 24.5 条三元组，
  retrieval 平均延迟 1.74 ms。
- `(top_k=2, hops=2)` 是推荐的均衡配置：召回提高到 91.14%，平均候选三元组
  增加到 36.8，平均延迟为 2.55 ms。
- 若更重视召回，`(top_k=1, hops=3)` 达到 93.25%，平均只有 45.5 条三元组；
  它比 `(top_k=4, hops=2)` 的召回更高，同时候选图和延迟都更小。
- `(top_k=8, hops=3)` 的最高召回为 95.78%，但平均候选三元组达到 219.2，
  平均延迟为 15.00 ms。相对 `(1,3)` 只增加 2.53 个百分点，不适合作为默认值。

因此下一轮完整 DAG 和推理精度实验建议至少比较 `(1,2)`、`(2,2)` 和 `(1,3)`
三组配置，分别代表低开销、均衡和召回优先。

`--reference-output` 用于将新生成的 DAG 与已有 _dag_aa 文件进行离线对比，不参与检索或 DAG 生成。
它会按 _id/id 对齐共同样本，并统计：
完整 DAG 相同率
非空 DAG 比例
答案覆盖率
平均 KV 节点数
KV 集合的 macro precision、recall 和 F1
