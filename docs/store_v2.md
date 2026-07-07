# Store V2

本文记录当前已经实现的 PathWeaver Store V2。当前版本先只围绕三个现实问题收敛，
不考虑 1B 级知识库的长期形态：

1. KV tensor 追加不能再依赖单个 `.npy` 全量重写；
2. hybrid mention 不能再扫描全体 entity；
3. 在线局部图恢复不能继续走 SQLite + Python object 热路径。

当前实现对应代码：

- [src/kblam/stores/kv_store_v2.py](/mnt/n0/PathWeaver/src/kblam/stores/kv_store_v2.py)
- [src/kblam/stores/graph_store_v2.py](/mnt/n0/PathWeaver/src/kblam/stores/graph_store_v2.py)
- [src/kblam/dag_store_retriever_v2.py](/mnt/n0/PathWeaver/src/kblam/dag_store_retriever_v2.py)
- [tools/build_pathweaver_store_v2.py](/mnt/n0/PathWeaver/tools/build_pathweaver_store_v2.py)
- [tools/benchmark_pathweaver_store_v2.py](/mnt/n0/PathWeaver/tools/benchmark_pathweaver_store_v2.py)

## 当前 V2 架构

V2 暂时不推翻现有 V1 ingest，而是采用“两层”：

1. V1 canonical store 继续负责离线去重和构建；
2. V2 从 V1 export 出 serving snapshot，在线检索只读 V2。

当前 V2 的物理目录：

```text
store_v2/
  manifest.json
  graph_v2/
    manifest.json
    entity_node_ids.npy
    entity_names.json
    entity_aliases.json
    forward_index.npy
    forward_triples.npy
    reverse_index.npy
    reverse_triples.npy
    triple_ids.npy
    triple_subject_pos.npy
    triple_object_node_id.npy
    triple_object_kind.npy
    triple_type.npy
    triple_predicates.json
    triple_object_names.json
    triple_titles.json
    triple_kv_index.npy
    triple_kv_offsets.npy
    entity_vector_ids.npy
    entity_vectors.npy
    entity_hnsw.bin
    entity_hnsw.json
  kv_v2/
    kv_store.sqlite3
    tensor_manifest.json
    key/seg_*.npy
    value/seg_*.npy
```

当前 V2 只做了必要重构：

- `kv_store.sqlite3` 仍保留，用于按 offset 读取 KV 文本；
- graph SQLite 不再进入在线检索热路径；
- key/value base embedding 改成 segment 化；
- mention 从“全表扫描”改成“ANN shortlist + lexical rerank”；
- local subgraph 从 graph snapshot 的邻接数组恢复。

## 三个问题在 V2 里的对应改法

### 1. KV append 重写问题

V1 的问题是 `append_tensors()` 扩容时会重写完整 key/value `.npy`。

V2 的做法是：

- 保持 offset 单调递增不变；
- key / value 改成多个 segment 文件；
- `tensor_manifest.json` 记录每个 segment 覆盖的 offset 范围；
- 追加时只新增 segment，不重写历史 segment。

这一步解决的是“追加写放大”和“额外完整数组峰值空间”，不是稳定态容量。

### 2. mention scan 线性扫描问题

V1 当前 hybrid 会扫描所有 entity name。

V2 当前先不做全量 alias index，而是用一个更小改动的过渡版：

1. ANN 先召回较大的 `entity_candidate_top_k`；
2. 只在这批 shortlist 上做 lexical rerank；
3. lexical 命中加 bonus，最终保留 `entity_top_k`。

这个版本已经把复杂度从 `O(all entities)` 降成了：

```text
ANN + O(candidate_top_k)
```

### 3. Graph SQLite 热路径问题

V1 在线局部图恢复依赖：

- SQLite 查 `subject_id` / `object_id`
- 再逐 triple inflate 成 `GraphTriple`
- 再额外查 `triple_kvs`

V2 当前把图导出成几组数组：

- entity 列表；
- forward adjacency；
- reverse adjacency；
- triple table；
- triple -> kv_offsets 扁平数组。

在线 1~2 hop expansion 不再访问 SQLite。

## 当前实现的边界

这版 V2 还不是最终形态，当前明确保留了几个简化：

- 仍依赖 V1 store 作为 export source；
- 仍保留 `kv_store.sqlite3` 做 KV 文本读取；
- mention 还是 shortlist rerank，不是正式 alias index；
- graph snapshot 里字符串还用了 JSON，没有进一步做 string pool 压缩；
- 目前只先验证 base 2Wiki store。

也就是说，这版重点是把三个主要瓶颈从热路径拿掉，而不是追求最终极致空间压缩。

## Base 2Wiki：构建与导出命令

数据集：

```text
/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl
```

### 1. 先构建 V1 base store

如果本地还没有 `experiments/stores/2wiki-dev-v5`，先跑：

```bash
cd /mnt/n0/PathWeaver

export CUDA_VISIBLE_DEVICES=0

/mnt/n0/uv_envs/kblam/bin/python tools/build_pathweaver_stores.py \
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

### 2. 从 V1 导出 V2

```bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/build_pathweaver_store_v2.py \
  --source-store-dir experiments/stores/2wiki-dev-v5 \
  --output-store-dir experiments/stores/2wiki-dev-v5-store-v2 \
  --segment-rows 4096
```

`--segment-rows` 当前只是控制 segment 粒度，方便先验证 V2 读写路径；不是最终推荐值。

## Base 2Wiki：V1 / V2 对比命令

```bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/benchmark_pathweaver_store_v2.py \
  --store-v1 experiments/stores/2wiki-dev-v5 \
  --store-v2 experiments/stores/2wiki-dev-v5-store-v2 \
  --queries /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --limit 100 \
  --warmup-queries 10 \
  --repeats 3 \
  --entity-top-k 1 \
  --entity-candidate-top-k 64 \
  --subgraph-hops 2 \
  --seed-strategy hybrid \
  --search-backend hnsw \
  --output experiments/stores/2wiki-dev-v5-store-v2/benchmark_v1_vs_v2.json
```

结果文件：

```text
experiments/stores/2wiki-dev-v5-store-v2/benchmark_v1_vs_v2.json
```

## Base 2Wiki：接入在线 DAG-KV 评测

`OnlineDAGKBRetriever` 现在已经支持通过 `--online_store_version v2` 切到 V2 store。
当前需要配合：

- `--online_store_dir experiments/stores/2wiki-dev-v5-store-v2`
- `--online_store_version v2`
- 如果使用 hybrid，还可以显式指定 `--online_entity_candidate_top_k`

注意：对 `llm_type=qwen3`，当前 `experiments/eval_generation_dag_kv.py` 里的两个模型参数语义是：

- `--base_model_name_or_path`：Qwen3 base model 目录；
- `--model_path`：KBLaM-Qwen3 checkpoint 目录，里面必须包含 `q_proj_new` 权重。

因此这里的 `--model_path` 不能再填纯 base model `/mnt/n0/models/qwen3-14B-Instruct`，否则会在
`q_proj_new missing after state_dict load` 处失败。

示例命令：

```bash
source /mnt/n0/uv_envs/kblam/bin/activate
cd /mnt/n0/PathWeaver

DATA_DIR=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set
MODEL_DIR=/mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800
ENCODER=/mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800_encoder/encoder.pt

python experiments/eval_generation_dag_kv.py \
  --data_path "$DATA_DIR/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl" \
  --model_path "$MODEL_DIR" \
  --base_model_name_or_path /mnt/n0/models/qwen3-14B-Instruct \
  --encoder_path "$ENCODER" \
  --encoder_spec qwen-embedding-0.6B \
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
  --online_store_dir experiments/stores/2wiki-dev-v5-store-v2 \
  --online_store_version v2 \
  --online_dag_script docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v8_infer_only.py \
  --online_dag_model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --online_st_model /mnt/n0/models/bge-en-v1.5/ \
  --online_entity_top_k 1 \
  --online_entity_candidate_top_k 64 \
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
  --use_multihop_adj \
  --max_hops 10 \
  --hop_decay 1.0 \
  --dynamic_hops_by_longest_path \
  --save_json experiments/results/online_dag_eval/online_store_v2_hybrid_top1_hop2_sink3_100.json
```

```
{
  "performance": {
    "queries": 100,
    "generated_samples": 100,
    "empty_kb_samples": 0,
    "qps": 2.4001448102940457,
    "average_latency_seconds": 0.41664152750745415,
    "average_model_ttft_seconds": 0.08035006248392165,
    "average_retrieval_seconds": 0.07116825531236827,
    "average_end_to_end_ttft_seconds": 0.15151831779628994,
    "average_tpot_seconds": 0.06638826633440391
  }
}
[OnlineDAG] samples=100 empty_dags=0 candidate=8.88ms dag=61.11ms tensor=1.14ms total=71.13ms
===== First 5 Samples =====
Sample 0:
  Model output: Mannai Award
  Ground truth: Myanmar Motion Picture Academy Awards
  --------------------------------------------------
Sample 1:
  Model output: Zuzanna Brzóska
  Ground truth: Małgorzata Braunek
  --------------------------------------------------
Sample 2:
  Model output: 12 June 1516
  Ground truth: 12 June 1516
  --------------------------------------------------
Sample 3:
  Model output: Alain Poiré
  Ground truth: Alain Poiré
  --------------------------------------------------
Sample 4:
  Model output: Julius Caesar
  Ground truth: Pompey
  --------------------------------------------------
Calculating ROUGE scores...
✅ ROUGE computed successfully.
✅ Exact Match (EM): 0.4600
✅ F1-Overlap: 0.5962
🔸 Large input detected (100 samples), splitting into batches of 20...
✅ Faithfulness01: 0.5800
{
  "rouge1": 0.6134577922077922,
  "rouge2": 0.2569166666666667,
  "rougeL": 0.6110119047619047,
  "rougeLsum": 0.6112283549783549,
  "exact_match": 0.46,
  "f1_overlap": 0.5961825396825396,
  "faithfulness01": 0.58
}
```

## Base 2Wiki：比较结果

本节数据来自上面的 benchmark 命令，query 为 2Wiki 前 100 条，warmup 10 条，重复 3 次。

### 空间

这里只比较 serving 相关文件，不额外把可选 `canonical/` 副本算进去。

| 版本 | 总空间 |
| --- | ---: |
| V1 | 132.04 MiB |
| V2 | 130.44 MiB |

稳定态空间只小幅下降，原因很直接：

- V2 去掉了 graph SQLite 热路径，换成 snapshot arrays；
- 但仍保留了 `kv_store.sqlite3`；
- 2Wiki base store 规模较小，segment 化主要改善的是 append 写放大，不是稳定态容量。

所以当前 V2 的核心收益不是“base store 立刻变得很小”，而是：

- 追加不再全量重写；
- mention 不再全表扫；
- graph 不再查 SQLite。

### 检索

同一批 query 下，候选图规模没有变化：

- candidate triple mean: `21.04 -> 21.04`
- candidate KV mean: `42.40 -> 42.40`
- answer recall: `1.00 -> 1.00`

说明当前 V2 没有通过“缩小候选图”换速度，而是主要靠实现路径变轻。

| 阶段 | V1 p50 | V2 p50 | 改善 |
| --- | ---: | ---: | ---: |
| ANN | 0.481 ms | 0.515 ms | 基本持平 |
| Mention / rerank | 12.710 ms | 0.119 ms | 106.9x |
| 局部图 | 0.766 ms | 0.383 ms | 2.0x |
| KV text | 0.435 ms | 0.481 ms | 基本持平 |
| KV tensor | 0.991 ms | 0.165 ms | 6.0x |
| Total | 15.382 ms | 1.697 ms | 9.1x |

当前结论：

1. ANN 本来就不是 base 2Wiki 的主瓶颈；
2. 最大收益来自 hybrid mention 从全表扫改成 shortlist rerank；
3. graph snapshot 让局部图恢复大约再快一倍；
4. segmented KV tensor 的读取在这个规模下也比 V1 单大数组更快，说明小范围 gather 的局部性更好。

## 当前对比的理解

这版结果说明，围绕你提的三个问题做的 V2 改动，在 base store 上已经是有效的：

- 问题 1：append 重写
  - 目前已通过 segmented KV layout 解决写路径结构问题；
  - base benchmark 还没有单独量化 append 时间，但物理布局已经切换成功。

- 问题 2：mention scan
  - 已从 p50 `12.71 ms` 降到 `0.119 ms`；
  - 这是当前 V2 最大的收益来源。

- 问题 3：Graph SQLite 热路径
  - 已从 SQLite incident query 切到 snapshot adjacency；
  - 局部图 p50 从 `0.766 ms` 降到 `0.383 ms`。

## 下一步：64k 知识库命令

下一步建议先把已经存在的 64k V1 store 导出成 V2，再按同一批 2Wiki query 对比。

### 1. 导出 64k V2

```bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/build_pathweaver_store_v2.py \
  --source-store-dir experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train \
  --output-store-dir experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2 \
  --segment-rows 131072
```

### 2. 比较 64k V1 / V2

```bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/benchmark_pathweaver_store_v2.py \
  --store-v1 experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train \
  --store-v2 experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2 \
  --queries /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --limit 100 \
  --warmup-queries 10 \
  --repeats 3 \
  --entity-top-k 1 \
  --entity-candidate-top-k 64 \
  --subgraph-hops 2 \
  --seed-strategy hybrid \
  --search-backend hnsw \
  --output experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2/benchmark_v1_vs_v2.json
```

## 当前最值得继续观察的点

下一轮重点建议看三件事：

1. 64k 时 V2 的 shortlist rerank 是否还能保持 answer recall；
2. 64k 时 graph snapshot 的 tail latency 能压多少；
3. segmented KV 在大 candidate set 下的 `kv_tensor` 读取是否仍优于 V1。

如果这三点在 64k 上仍然成立，再继续往更大的 store 推，会更有把握。
