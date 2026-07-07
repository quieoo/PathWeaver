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

## 64k 知识库：命令与结果

以下结果基于已经存在的 `064000-with-train` V1 store 导出 V2 后，在同一批 2Wiki
前 100 条 query 上对比得到。

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

结果文件：

```text
experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2/benchmark_v1_vs_v2.json
```

### 空间

| 版本 | 总空间 |
| --- | ---: |
| V1 | 16267.27 MiB |
| V2 | 16009.13 MiB |

64k 档位下，V2 稳定态空间相比 V1 仅下降约 `258.14 MiB`，约 `1.59%`。这和 base 2Wiki
时的结论一致：当前 V2 的主要收益不是稳定态容量，而是在线检索路径变轻，以及 KV append
不再依赖单文件全量重写。

### 检索

这一档位下，V2 仍保持 `answer_recall = 1.0`，说明 shortlist rerank 至少在这批
2Wiki query 上没有带来召回损失。

不过一个重要变化是：V2 的候选图规模变大了。

- candidate triple mean: `1231.27 -> 2213.86`
- candidate KV mean: `2483.59 -> 4480.67`

也就是说，这一档位下 V2 不是靠“更小的候选图”换速度，而是：

- 把 mention scan 从全表线性扫描降成 shortlist rerank；
- 把 graph expansion 和 KV tensor 读取做轻；
- 即使候选图更大，整体延迟仍明显更低。

| 阶段 | V1 p50 | V2 p50 | 改善 |
| --- | ---: | ---: | ---: |
| ANN | 4.824 ms | 4.837 ms | 基本持平 |
| Mention / rerank | 1090.624 ms | 0.269 ms | 4060.8x |
| 局部图 | 11.006 ms | 2.717 ms | 4.1x |
| KV text | 6.240 ms | 4.505 ms | 1.4x |
| KV tensor | 6.307 ms | 1.707 ms | 3.7x |
| Total | 1121.164 ms | 13.848 ms | 81.0x |

tail latency 也显著改善：

- total p95: `1481.42 ms -> 661.92 ms`，约 `2.24x`
- total p99: `2097.91 ms -> 774.12 ms`，约 `2.71x`

### 结果解读

64k 结果说明三件事：

1. mention scan 仍然是 V1 的绝对主瓶颈，而 V2 已经把它几乎完全消掉；
2. graph snapshot 在大规模候选图下仍有效，局部图 p50 从 `11.01 ms` 降到 `2.72 ms`；
3. segmented KV tensor 在更大的候选集下仍优于 V1，`kv_tensor` p50 从 `6.31 ms`
   降到 `1.71 ms`。

但也暴露了一个新的现象：

- V2 当前 hybrid shortlist rerank 会改变 top-1 seed，从而把更多 dense neighborhood
  拉进候选图；
- 所以 candidate triple / KV 均值明显变大；
- 这说明 V2 下一步不能只看 latency，还要继续观察“更大候选图是否会推高 DAG 阶段和
  端到端 generation 成本”。

当前可以先得出的结论是：即使 64k 档位候选图膨胀，V2 在 store-only 检索上依然远快于
V1，最大的收益仍然来自去掉线性 mention scan，其次是 graph snapshot 和 segmented
KV tensor 读取。

### 为什么 64k 的候选图会膨胀

一个容易误判的点是：候选图大小并不是由“全局平均图密度”直接决定的，而是更接近由
“最终 seed entity 的 2-hop 局部邻域大小”决定。

对 V2 store 做实体级 KV 分布统计后，可以看到 64k 和 base 2Wiki 的中位数确实比较接近：

| Store | entities | mean | p50 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki base V2 | 3,006 | 6.66 | 4 | 18 | 25.5 | 44 | 98 |
| 64k V2 | 272,775 | 10.26 | 4 | 24 | 36 | 86 | 3091 |

这里的统计口径是 `incident`，也就是“一个实体作为图节点时，能碰到的唯一 KV 数”。

从这组数字可以看到：

- `p50` 都是 `4`，说明大多数普通实体并不胖；
- 但 64k 的 tail 明显更重，`p99` 从 `44` 提高到 `86`，`max` 从 `98` 直接拉到 `3091`；
- 所以 64k 的问题不是“全图平均更密很多”，而是“长尾 hub 节点重得多”。

这会在当前检索链路里被进一步放大，原因有三层：

1. 当前候选图来自 `top-1 seed + 2-hop expansion`，不是随机采样实体；
2. V2 的 hybrid shortlist rerank 会改变最终 top-1 seed，更容易把 query 落到一个“语义相关但局部邻域更胖”的实体上；
3. 2-hop expansion 是乘法效应：只要 seed 稍微更靠近 tail，它的一跳邻居往往也更 dense，
   展开到两跳后，candidate triple / KV 数就会快速膨胀。

因此，虽然 64k 和 base 在全局 `p50` 上几乎一样，但这并不能约束在线检索看到的候选图大小。
真正决定候选图大小的是：

- 这批 query 最终命中了哪些 seed；
- 这些 seed 在全局分布里落在什么分位；
- 它们的 1-hop / 2-hop 邻域是否位于长尾区域。

这也能解释为什么当前现象是：

- store-only benchmark 里，V2 仍然能靠更轻的实现把 `mention / graph / kv_tensor` 压得很低；
- 但一旦 seed 更常落到长尾 hub，candidate graph 还是会明显变大；
- 再往后进入 DAG extractor 时，这种膨胀会进一步转化成更高的 DAG 计算成本和更差的端到端效果。

所以目前对 64k 膨胀问题更准确的表述应当是：

- 不是“64k 全局图密度比 2Wiki 大很多”；
- 而是“64k 的长尾 hub 更重，V2 的 shortlist rerank 更容易把 query 带进这些 hub 的 2-hop
  neighborhood，于是候选图膨胀”。

下一步如果要把这个判断彻底坐实，最直接的诊断不是继续看全局分布，而是对 benchmark 的
100 条 query 逐条记录：

- 最终 seed entity；
- seed 本身的 incident triple / KV 数；
- seed 的 1-hop / 2-hop 子图大小；
- V1 / V2 是否选中了不同的 top-1 seed。

这会比全局图密度统计更直接地解释候选图为什么会变大。

### 进一步诊断实验：直接比较 top-1 seed 的局部扩图

为了验证上面的判断，这里进一步在同一批 2Wiki 前 100 条 query 上，对 `base 2Wiki V2`
和 `64k V2` 跑同一套检索配置：

- `entity_top_k=1`
- `entity_candidate_top_k=64`
- `subgraph_hops=2`
- `seed_strategy=hybrid`
- `search_backend=hnsw`

并逐条记录：

- 最终 top-1 seed 是谁；
- seed 的 1-hop triple / KV 数；
- seed 的 2-hop triple / KV 数；
- 最终 candidate triple / KV 数。

复现实验命令：

```bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/diagnose_store_seed_expansion.py \
  --store-a experiments/stores/2wiki-dev-v5-store-v2 \
  --store-b experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2 \
  --label-a base_2wiki_v2 \
  --label-b store_64k_v2 \
  --queries /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --st-model /mnt/n0/models/bge-en-v1.5/ \
  --limit 100 \
  --entity-top-k 1 \
  --entity-candidate-top-k 64 \
  --subgraph-hops 2 \
  --search-backend hnsw \
  --seed-strategy hybrid \
  --mention-min-chars 8 \
  --output experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2/seed_expansion_base_vs_64k.json
```

实验结果非常直接：

| Metric | base 2Wiki V2 | 64k V2 |
| --- | ---: | ---: |
| hop1 triples mean | 8.29 | 64.13 |
| hop1 triples p95 | 16.0 | 540.0 |
| hop2 triples mean | 20.67 | 2423.72 |
| hop2 triples p95 | 32.05 | 18311.0 |
| candidate triples mean | 20.67 | 2423.72 |
| candidate triples p95 | 32.05 | 18311.0 |
| candidate KV mean | 41.62 | 4906.14 |
| candidate KV p95 | 66.0 | 37140.0 |

还可以看到两个关键事实：

1. `same_seed_name_rate = 0.8`，说明 100 条 query 里有约 20 条的 top-1 seed 确实发生了切换；
2. 在候选图膨胀最严重的 top 10 query 里，`64k V2` 的 top-1 seed 全部变成了 `director`，
   而 `base 2Wiki V2` 仍然命中了具体电影实体。

例如下面这些 query：

- `What is the place of birth of the director of film Letter To The King?`
- `What is the place of birth of the director of film Aurora (2018 Film)?`
- `Where was the director of film The Half-Way Girl born?`

在 `base 2Wiki V2` 上，top-1 seed 分别是具体电影：

- `Letter to the King`
- `Aurora (2018 film)`
- `The Half-Way Girl`

但在 `64k V2` 上，top-1 seed 都漂到了同一个高频泛化实体：

- `director`

而这个 `director` seed 的局部图规模极大：

- `hop2_triples = 18311`
- `hop2_kvs = 37140`

这就把原本只有十几到几十条 triple 的候选图，直接放大成上万条 triple。

这个实验因此已经能直接证明：

- 候选图膨胀并不是“全局平均图密度稍高一点”这么简单；
- 更直接的原因是：在 64k store 里，hybrid shortlist 在一部分 query 上把 top-1 seed
  从具体电影实体切换成了泛化 hub 实体 `director`；
- 一旦 top-1 seed 变成这个 hub，`2-hop expansion` 会立刻把候选图放大到 `18k+ triples /
  37k+ KVs` 的量级；
- 由于当前 `entity_top_k=1`，最终 candidate graph 几乎就是这个 top-1 seed 的 2-hop
  子图，所以 seed 漂移会直接转化成候选图膨胀。

也就是说，64k 候选图膨胀的核心原因已经不是猜测，而是有实验直接支持的：

- 问题集中发生在 top-1 seed 选择；
- 具体表现为一批 query 的 seed 从具体电影实体漂到了高频 hub `director`；
- 然后被 `2-hop` 扩图机制放大。

### 规则修复实验：subject-only entity

针对上面的 seed 漂移问题，这里进一步落地了一条最小规则：

- 只要一个节点在任意三元组的第一元出现过，它就保留在 V2 的 entity 集里；
- 如果一个节点只在第三元出现、从来没有做过第一元，那么它不再进入 V2 的 entity 集；
- 这些节点仍然保留在 triple 的 object 文本里，也仍然保留对应 KV，只是不再参与 entity
  ANN 和 graph expansion。

最开始这条规则只在 V2 export 阶段生效，方便直接基于已有 V1 store 重建 serving snapshot。
现在这条规则也已经同步落到了 V1 ingest：

- relation triple 的第三元初始按 literal 落库；
- 只有当该节点后来在任意 triple 里作为第一元出现时，才升级成 entity；
- 升级时会同步修正历史 relation edge 的 `object_kind`，保证 V1 / V2 语义一致。

#### 复现命令

先导出一版 subject-only V2：

```bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/build_pathweaver_store_v2.py \
  --source-store-dir experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train \
  --output-store-dir experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2-subject-only \
  --segment-rows 131072 \
  --subject-only-entities
```

base 2Wiki 的 sanity check 导出命令：

```bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/build_pathweaver_store_v2.py \
  --source-store-dir experiments/stores/2wiki-dev-v5 \
  --output-store-dir experiments/stores/2wiki-dev-v5-store-v2-subject-only \
  --segment-rows 131072 \
  --subject-only-entities
```

#### base 2Wiki：sanity check

在 base 2Wiki 上，这条规则没有破坏检索行为，反而把 entity 集进一步压小：

- entities: `3006 -> 1377`
- candidate triples mean: `21.04 -> 19.92`
- candidate KV mean: `42.40 -> 40.16`
- answer recall: `1.00 -> 1.00`

延迟仍然保持和原 V2 同一量级：

- total p50: `1.52 ms`
- mention p50: `0.129 ms`
- graph p50: `0.310 ms`
- kv_tensor p50: `0.165 ms`

这说明 subject-only 规则在 base store 上至少是稳定的，没有明显伤害召回。

#### 64k：store-only benchmark

基于同一批 2Wiki 前 100 条 query、`top-1 / hybrid / hop-2 / hnsw` 配置，新的 64k
subject-only V2 结果如下：

| Metric | 旧 64k V2 | subject-only 64k V2 |
| --- | ---: | ---: |
| entities | 272,775 | 108,886 |
| candidate triples mean | 2213.86 | 626.18 |
| candidate KV mean | 4480.67 | 1260.04 |
| answer recall | 1.00 | 1.00 |
| total p50 | 13.85 ms | 10.21 ms |
| total p95 | 661.92 ms | 159.56 ms |
| total p99 | 774.12 ms | 252.66 ms |

对比 V1 时，subject-only V2 仍然明显更快：

- V1 total p50: `1286.88 ms`
- subject-only V2 total p50: `10.21 ms`

分阶段看，变化也很清楚：

- ANN p50: `4.84 -> 3.93 ms`
- mention / rerank p50: `0.269 -> 0.261 ms`
- graph p50: `2.72 -> 1.79 ms`
- kv_text p50: `4.51 -> 2.45 ms`
- kv_tensor p50: `1.71 -> 1.20 ms`

也就是说，这条规则在 64k 下既没有伤到 `answer_recall`，又显著压缩了候选图和 tail latency。

#### 64k：seed 膨胀修复前后直接对比

为了验证问题是否真的被这条规则打中，这里直接比较旧 64k V2 和 subject-only 64k V2 的
top-1 seed 局部扩图。

| Metric | 旧 64k V2 | subject-only 64k V2 |
| --- | ---: | ---: |
| hop1 triples mean | 64.13 | 13.70 |
| hop2 triples mean | 2423.72 | 672.80 |
| hop2 triples p95 | 18311.0 | 3735.4 |
| candidate KV mean | 4906.14 | 1355.23 |
| candidate KV p95 | 37140.0 | 7509.0 |

同时：

- `same_seed_name_rate = 0.81`
- 也就是说，大多数 query 的 seed 没变；
- 但正是那部分发生 seed 切换的 query，贡献了几乎全部的极端膨胀。

修复前，最严重的一批 query 往往会选到：

- `director`
- `performer`
- `legendary German director`
- `Polish actor and director`

这类只做第三元、从不做第一元的泛化 / 描述型节点。

修复后，这些节点被移出 entity 集，seed 会回到更具体的实体上，例如：

- `My Heidelberg, I Can Not Forget You`
- `Polish-Russian War`
- `Pharoahe Monch`

因此，这次实验可以直接支持下面这个结论：

- 64k 候选图膨胀的核心原因，确实和 object-only 节点被错误实体化有关；
- 只保留“至少做过一次第一元”的 subject-only entity 规则，可以在不伤 `answer_recall`
  的前提下，把这类坏 seed 大幅压掉；
- 候选图均值、tail、graph / kv 阶段延迟都随之明显下降。

### 量化问题

````bash
cd /mnt/n0/PathWeaver

PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/audit_relation_objects.py \
  --store-dir experiments/stores/2wiki-dev-v5-store-v2 \
  --top-k 50 \
  --sample-per-object 3 \
  --output experiments/stores/2wiki-dev-v5-store-v2/relation_object_audit.json


PYTHONPATH=src /mnt/n0/uv_envs/kblam/bin/python tools/audit_relation_objects.py \
  --store-dir experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2 \
  --top-k 50 \
  --sample-per-object 3 \
  --output experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2/relation_object_audit.json
````

## 64k 知识库：online generation 全路径

下面这条命令是在 64k subject-only V2 store 上跑完整在线链路：实体检索、局部图恢复、
V8 answer-blind DAG 提取、KV tensor 恢复、KBEncoder 投影，以及 Qwen3-14B generation。

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
  --online_store_dir experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2-subject-only \
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
  --save_json experiments/results/online_dag_eval/online_store_v2_64k_subject_only_hybrid_top1_hop2_sink3_heuristic_100.json
```

当前文档里还没有补入这条命令对应的 subject-only online generation 实测结果；跑完后建议把
`performance`、`[OnlineDAG]` stage breakdown，以及 `EM / F1 / Faithfulness01` 贴回本节。

结果建议单独写到：

```text
experiments/results/online_dag_eval/online_store_v2_64k_subject_only_hybrid_top1_hop2_sink3_heuristic_100.json
```

这样可以和 base 2Wiki V2 的在线结果分开保存，避免覆盖。

## KV per Entity 

````bash
cd /mnt/n0/PathWeaver

/mnt/n0/uv_envs/kblam/bin/python tools/entity_kv_stats.py \
  --store-dir experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2

/mnt/n0/uv_envs/kblam/bin/python tools/entity_kv_stats.py \
  --store-dir experiments/stores/2wiki-dev-v5-store-v2
````

```
{
  "config": {
    "store_dir": "experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2",
    "store_version": "v2",
    "scope": "incident"
  },
  "kv_per_entity": {
    "count": 272775,
    "mean": 10.264829988085419,
    "min": 2,
    "p50": 4.0,
    "p90": 24.0,
    "p95": 36.0,
    "p99": 86.0,
    "max": 3091
  }
}

{
  "config": {
    "store_dir": "experiments/stores/2wiki-dev-v5-store-v2",
    "store_version": "v2",
    "scope": "incident"
  },
  "kv_per_entity": {
    "count": 3006,
    "mean": 6.655355954757153,
    "min": 2,
    "p50": 4.0,
    "p90": 18.0,
    "p95": 25.5,
    "p99": 44.0,
    "max": 98
  }
}
```
