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

## 在线检索

当前最优配置为 top-1 / 2-hop、hybrid entity seed、heuristic terminal reranker
和 `max_sinks=3`。下面的命令在数据集前 100 条样本上同时执行实体检索、局部图
恢复、V8 answer-blind DAG 提取、KVStore 读取、KBEncoder 和 Qwen3-14B 推理：

```bash
export CUDA_VISIBLE_DEVICES=0
source /mnt/n0/uv_envs/kblam/bin/activate
cd /mnt/n0/PathWeaver

DATA_DIR=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set
MODEL_DIR=/mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800
ENCODER=/mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800_encoder/encoder.pt

python experiments/eval_generation_dag_kv.py \
  --data_path "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl" \
  --model_path "$MODEL_DIR" \
  --encoder_path "$ENCODER" \
  --encoder_spec qwen-embedding-0.6B \
  --base_model_name_or_path /mnt/n0/models/qwen3-14B-Instruct \
  --llm_type qwen3 \
  --kb_layer_frequency 3 \
  --kb_scale_factor 4 \
  --path_attn \
  --path_attn_mix_ratio 0.8 \
  --step 7999 \
  --t_step 8000 \
  --max_samples 100 \
  --seed 0 \
  --eval_batch_size 10 \
  --online_store_dir experiments/stores/2wiki-dev-v5 \
  --online_dag_script docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v8_infer_only.py \
  --online_dag_model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --online_st_model /mnt/n0/models/bge-en-v1.5/ \
  --online_entity_top_k 1 \
  --online_subgraph_hops 2 \
  --online_search_backend hnsw \
  --online_seed_strategy hybrid \
  --online_mention_min_chars 8 \
  --online_infer_batch_size 1024 \
  --online_topic_top_k 8 \
  --online_dde_hops 3 \
  --online_mention_bonus 0.2 \
  --online_seed_edge_topk 18 \
  --online_expansion_hops 2 \
  --online_per_src_cap 3 \
  --online_max_nodes 30 \
  --online_max_edges 40 \
  --online_max_sinks 3 \
  --online_reverse_sink_edge_topk 2 \
  --online_reverse_sink_hops 4 \
  --online_reverse_sink_beam_width 4 \
  --online_selection_mode legacy \
  --online_terminal_reranker heuristic \
  --save_json experiments/results/online_dag_eval/online_store_v8_hybrid_top1_hop2_sink3_heuristic_100.json
```

实测结果（GPU 0，本地 EM/F1/ROUGE，不调用外部 LLM judge）：

```text
EM:                         0.4400
F1-overlap:                 0.5925
ROUGE-L:                    0.6118
Model TTFT:                 81.26 ms
Online retrieval:           80.18 ms
End-to-end TTFT:           161.44 ms
Average request latency:   431.85 ms
QPS:                         2.316

Retrieval breakdown:
  entity + candidate graph: 22.09 ms
  V8 DAG extraction:        57.14 ms
  KV read + projection:      0.90 ms
```

结果保存于
`experiments/results/online_dag_eval/online_store_v8_hybrid_top1_hop2_sink3_heuristic_100.json`。
去掉 `--max_samples 100` 即可评测完整输入文件。更详细的精度差距分析和消融结果见
本文“在线精度差距分析与修复”一节。

# KVStore 与 GraphStore

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

## Answer-Blind V8 DAG Retriever

V5.2 即使没有启用 `--answer-aware`，terminal KV 限额仍会保护与标准答案匹配的
边，因此不满足真实在线查询的要求。V8 inference-only 实现不读取 `answer` 或
`supporting_facts`，但保持原 checkpoint 的特征维度、边打分和节点打分兼容。

实现文件：

```text
docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v8_infer_only.py
```

`TrainableDAGExtractor` 根据脚本接口选择后端：V5.2 使用 `load_model()` 和
`create_dag_with_model()`，V8 使用 `load_models()` 和 `infer()`。在线模式强制要求
V8 后端，并只向 DAG Retriever 传递 `question` 与 GraphStore 恢复的三元组。

候选样本会清空原始 `context`，避免数据集内残留三元组绕过在线 GraphStore。
V8 导出的每个 KV node 携带 `kv_offset`，后续可以直接读取 KVStore tensor。

在相同的 2Wiki store 候选图上，V8 与 V5.2 NAA 的 223/237 张 DAG 结构完全一致，
macro KV-set Jaccard 为 0.9839；V8 的 answer-any recall 仅低 0.42 个百分点。详细的
答案泄漏分析和独立运行方法见
`docs/scripts/graph_gen/DAG_KV_SubgraphRAG_V8_ANSWER_BLIND.md`。

## 在线 DAG-KV 查询流程

在线链路只接入 `experiments/eval_generation_dag_kv.py`，不修改通用的
`experiments/eval_generation.py`：

1. 使用实体索引对应的 embedding 模型编码 query；
2. hybrid seed 优先匹配问题中的长实体名称，其余名额由 HNSW 补齐；
3. 从 GraphStore 恢复 top-1 实体的 2-hop 局部图；
4. V8 answer-blind DAG Retriever 对候选图剪枝；
5. 根据 DAG node 的 `kv_offset` 从 KVStore 批量读取 key/value embedding；
6. KBEncoder 投影 tensor，并根据 DAG adjacency 构造 `kb_adj`；
7. 将 query、KV tensors 和 `kb_adj` 输入 Qwen3 DAG-KV 模型。

在线输入使用原始 `tripled_v5` 文件，不依赖预生成的 `dag_naa` 或逐样本 embedding
数组。完整命令位于本文开头的“在线检索”Quick Start。

## 在线结果

当前保留配置为 `entity_top_k=1`、`subgraph_hops=2`、hybrid seed、heuristic
terminal reranker 和 `max_sinks=3`。GPU 0 上评测前 100 条 2Wiki dev 样本：

| 配置 | EM | F1 | ROUGE-L | 检索 | 端到端 TTFT | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 离线 `dag_naa` | 0.510 | 0.661 | 0.675 | 0 ms | 70.93 ms | 320.12 ms |
| 在线 Store + V8 | 0.440 | 0.592 | 0.612 | 80.18 ms | 161.44 ms | 431.85 ms |

在线检索耗时包含 query embedding、HNSW、局部图恢复、V8 DAG、KVStore 读取和
KBEncoder。端到端 TTFT 为在线检索与模型 prefill 之和。本次结果使用本地
EM/F1/ROUGE 指标，没有调用外部 LLM judge。
