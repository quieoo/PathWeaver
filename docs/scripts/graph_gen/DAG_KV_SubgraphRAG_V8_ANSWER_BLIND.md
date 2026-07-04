# DAG-KV V5.2 answer 泄露分析与 V8 无答案推理

## 结论

V5.2 即使不传 `--answer_aware`，最终 DAG 仍会被当前样本的 gold
`answer` 改变。原因不是训练模型本身，而是推理后处理中的 terminal-KV
限额逻辑无条件保护 answer-matching edge。

V8 的图构建器不访问 answer 字段，也不提供依赖 gold label 的推理选项。
在 2Wiki、HotpotQA、MuSiQue 和 MintQA 上重跑后，相对 V5.2 NAA 的答案
sink 召回分别下降 2.11、6.67、9.00 和 0.74 个百分点。2Wiki/MintQA 的代价
较小，HotpotQA/MuSiQue 更明显；因此不能只依据 2Wiki 推广“只有微小下降”。

## V5.2 为什么仍然泄露 answer

V5.2 中有三类 answer 使用。

1. 训练监督（不是在线泄露）

   - `weak_label_edge()` 用 answer 标注正边（V5.2 第 491--497 行）。
   - `weak_label_node_end()` 用 answer 标注 end node（第 756--773 行）。

   这些逻辑用于训练 checkpoint。训练集 label 作为监督本身是合理的；只要
   checkpoint 的训练数据不包含当前测试样本，就不等同于测试时读取 gold
   answer。V8 保留原 checkpoint，因此无需重新训练，也保留了主要质量来源。

2. 会改变最终 DAG 的在线泄露

   - `answer_terminalization()` 确实由 `--answer_aware` 控制（第
     2617--2640、2863--2871 行）。
   - 但随后无条件调用 `enforce_max_terminal_kv_nodes(..., answer=answer)`
     （第 2873--2881 行）。
   - 该函数把 answer-matching terminal edge 放入保护集合，优先删除其他
     terminal edge（第 2532--2574 行）。

   所以 `--answer_aware=False` 只关闭了第一种显式 answer 后处理，没有关闭
   第二种 answer-aware sink 裁剪。这正是原命令仍泄露答案并影响 DAG 的根因。

3. 不改变普通 DAG、但仍读取 answer 的逻辑

   - batched infer 对每条样本执行 `sample.get('answer')`（第 3006--3029 行）；
   - 在线输出 answer recall、sink rank 等统计（第 2840、2908--2943 行）；
   - 写入 `dag.meta.goal_ids`（第 2950--2954 行）；
   - `--dis_out_path` 按 answer recall 选择 evaluated IDs（第
     3657--3672 行）；
   - `--verbose` 会按答案 `Rome` 选择调试样本；
   - `--supporting_only` 使用 gold `supporting_facts`，属于另一种 gold-evidence
     泄露，而不是 answer 字段泄露。

## V8 实现

推荐文件：`DAG_KV_SubgraphRAG_trainable_v8_infer_only.py`

`v8_infer_only` 是可独立执行的单文件 inference entrypoint，不导入 V5.2、
V6、V7 或旧 V8，也不包含 train mode、训练 dataset、optimizer、弱标签构造
和训练 cache。文件内部只保留 JSON/JSONL IO、图构造、特征、checkpoint 兼容
模型、批量打分、answer-free 后处理与导出。

- 图生成代码只显式读取 `question`、`context`、`triple_list` 及其图字段；
- 文件中不存在 `sample.get('answer')`、`value_matches_answer()`、
  `answer_terminalization()` 或 `supporting_facts` 读取；
- CLI 只有 infer mode，默认 checkpoint 已经存在；路径不存在时立即报错；
- JSONL 按 batch 流式读取和写出，避免大文件同时驻留全部输入与 DAG；
- 输出路径使用进程锁防止重复任务并发写入；`--resume` 会先验证已有 JSONL，
  再从对应输入行继续；
- `dag.meta.goal_ids=[]`，并写入
  `answer_free_inference=true`；
- 输出通过浅复制保留原始样本的其他字段（包括 answer），便于后续独立
  evaluator 使用；这些字段对图生成代码是不透明的，不参与任何访问或分支。

这里的“answer-free”严格指当前样本的 gold answer 不参与 DAG 的特征、打分、
选择、裁剪或在线统计。checkpoint 在训练阶段使用 answer 弱标签仍被允许。若
要求训练阶段也从未见过任何 answer，则必须重新定义自监督训练目标并重训，
无法期待直接保持当前 checkpoint 的质量。

## 运行命令

当前仓库目录为 `/mnt/n0/PathWeaver`，环境实际位于
`/mnt/n0/uv_envs/kblam/bin/activate`。使用绝对路径可以避免从仓库目录执行时
找不到 `uv_envs/kblam/bin/activate`。

```bash
export CUDA_VISIBLE_DEVICES=0
source /mnt/n0/uv_envs/kblam/bin/activate

python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v8_infer_only.py \
  --mode infer \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v8_answer_blind.jsonl \
  --model_ckpt /mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt \
  --st_model /mnt/n0/models/bge-en-v1.5/ \
  --infer_batch_size 1024 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --keep_score \
  --reverse_sink_edge_topk 2 \
  --reverse_sink_hops 4 \
  --reverse_sink_beam_width 4
```

独立版没有 `--answer_aware`、`--supporting_only`、`--dis_out_path`、train
mode 等参数。需要 answer recall、
sink recall 或 goal IDs 时，应在 DAG 文件写完后，用原始 dataset 的 answer
做独立评估，不能把评估 label 反馈给图生成过程。

## 全测试集重跑结果

V8 使用同一个 checkpoint、GPU 0 以及原命令的全部图参数。AA 和 NAA 是附件
中已有的 V5.2 输出；V8 是本次实际重跑生成的 `*_dag_v8_answer_blind.jsonl`。

本次按最终确认只统计测试集。178,027 条训练集推理因耗时较长被用户取消；
已停止训练集进程并删除 10,240 行未完成文件，因此下表不混入 partial 结果。

本次生成的完整测试文件：

- `2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_v8_answer_blind.jsonl`（237 行）；
- `hotpot_dev_tripled_v5-qwen3.5-27B_dag_v8_answer_blind.jsonl`（300 行）；
- `musique_dev_tripled_v5-qwen3.5-27B_dag_v8_answer_blind.jsonl`（300 行）；
- `mintqa_pruned64_hop2_dag_v8_answer_blind.jsonl`（808 行）。

AA 不是公平的在线基线：它使用 gold answer 强制 terminalization，并且
`--answerable_only` 最终只保留 answer sink 命中的样本。因此表中的比率都以
原输入样本数为分母，而不是以 AA 输出行数为分母。

### 输出行数与 answer recall

`any` 表示 answer 出现在任意导出 KV node；`sink` 表示 answer 出现在
terminal KV node；`supported sink` 进一步要求该 terminal KV node 有入边。

| 数据集 | 方法 | 输出/输入 | any | sink | supported sink |
|---|---|---:|---:|---:|---:|
| 2Wiki | V5.2 AA | 208/237 | 87.76% | 87.76% | 85.23% |
| 2Wiki | V5.2 NAA | 237/237 | 88.19% | 85.23% | 83.12% |
| 2Wiki | V8 blind | 237/237 | 86.50% | 83.12% | 81.01% |
| HotpotQA | V5.2 AA | 264/300 | 88.00% | 88.00% | 80.33% |
| HotpotQA | V5.2 NAA | 300/300 | 89.00% | 82.00% | 76.67% |
| HotpotQA | V8 blind | 300/300 | 83.67% | 75.33% | 70.00% |
| MuSiQue | V5.2 AA | 190/300 | 63.33% | 63.33% | 56.67% |
| MuSiQue | V5.2 NAA | 300/300 | 69.00% | 58.33% | 54.67% |
| MuSiQue | V8 blind | 300/300 | 62.00% | 49.33% | 45.33% |
| MintQA | V5.2 AA | 748/808 | 92.57% | 92.57% | 60.15% |
| MintQA | V5.2 NAA | 808/808 | 92.82% | 89.11% | 59.41% |
| MintQA | V8 blind | 808/808 | 92.08% | 88.37% | 58.17% |

### V8 相对 V5.2 NAA 的变化

| 数据集 | any 变化 | sink 变化 | supported sink 变化 | 完整 DAG 相同 | KV-set Jaccard |
|---|---:|---:|---:|---:|---:|
| 2Wiki | -1.69 pp | -2.11 pp | -2.11 pp | 218/237 (91.98%) | 0.9734 |
| HotpotQA | -5.33 pp | -6.67 pp | -6.67 pp | 247/300 (82.33%) | 0.9291 |
| MuSiQue | -7.00 pp | -9.00 pp | -9.33 pp | 253/300 (84.33%) | 0.9333 |
| MintQA | -0.74 pp | -0.74 pp | -1.24 pp | 775/808 (95.92%) | 0.9812 |

### 图规模

| 数据集 | 方法 | 非空 DAG | 平均 KV nodes | 平均 terminal KV nodes |
|---|---|---:|---:|---:|
| 2Wiki | NAA / V8 | 237 / 237 | 12.076 / 11.966 | 2.966 / 2.966 |
| HotpotQA | NAA / V8 | 300 / 300 | 11.010 / 11.073 | 2.953 / 2.940 |
| MuSiQue | NAA / V8 | 300 / 300 | 11.387 / 11.403 | 2.937 / 2.917 |
| MintQA | NAA / V8 | 808 / 808 | 8.686 / 8.649 | 2.947 / 2.946 |

其他结构对照：

- 修改前 8 条样本的 answer 为完全错误的随机字符串后重新运行 V8，8/8 的
  KV nodes（忽略浮点 score）和 adjacency 保持完全一致。
- standalone `v8_infer_only` 与原 answer-blind V8 在 237/237 条样本上的
  KV nodes、adjacency、edge score 和 joint score 完全一致，最大浮点差为 0。

## 在线使用

在线查询必须使用 answer-blind V8，并由 GraphStore 提供候选三元组。当前推荐参数：

```bash
--entity-top-k 1 \
--subgraph-hops 2 \
--seed-strategy hybrid \
--mention-min-chars 8 \
--max-sinks 3 \
--selection-mode legacy \
--terminal-reranker heuristic
```

完整的 store 构建、在线检索和 Qwen3 推理命令见 `docs/stores.md`。最终配置以
下游 QA EM/F1 为准，不能用测试答案参与 terminal 保护或在线参数选择。

差距来自移除 gold-answer terminal-edge 保护后的真实代价，不应通过另一个
读取 answer 的启发式把它“补回”。若要缩小 HotpotQA/MuSiQue 上的差距，合理
方向是使用独立训练/验证集改进 node-ending scorer，而不是在测试时保护答案边。
