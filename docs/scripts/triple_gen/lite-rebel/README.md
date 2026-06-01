# lite-rebel

`lite-rebel` 是一套基于 [Babelscape/rebel](https://github.com/Babelscape/rebel) 的轻量知识图谱抽取实现，目标是尽量对齐 [`build_knowledge_graph_v5.py`](/mnt/n0/PathWeaver/docs/scripts/triple_gen/build_knowledge_graph_v5.py) 的输入输出风格，但把 `stage1` 的抽取内核替换成 REBEL。

官方参考：

- REBEL GitHub：<https://github.com/Babelscape/rebel>
- Hugging Face 模型：<https://huggingface.co/Babelscape/rebel-large>

实现说明：

- `stage1`
  - 使用 REBEL 逐句生成 relation triplets
  - 解析官方线性化输出中的 `<triplet> / <subj> / <obj>` 标记
  - 生成 `entity_list + triples`
- `stage2`
  - 默认启用一个 REBEL 专用的启发式 answer-aware 修图逻辑
  - 优先保留 REBEL 的规范关系，再用轻量 regex + 目标导向补图补关键桥
  - 如果你只想保留 `stage1`，可以显式传 `--disable-answer-aware`
- `final`
  - 复用现有确定性 KV 生成逻辑
  - 输出 `triple_list / answer_sufficient / missing_links / revision_notes`

## 目录文件

- [extractor_rebel.py](/mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-rebel/extractor_rebel.py)
  - REBEL 抽取器与输出解析
- [build_knowledge_graph_rebel.py](/mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-rebel/build_knowledge_graph_rebel.py)
  - 只跑 `stage1`
- [build_knowledge_graph_v5_rebel.py](/mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-rebel/build_knowledge_graph_v5_rebel.py)
  - 完整 v5 风格 pipeline

## 输入格式

支持：

- `.json`
- `.jsonl`
- JSON 根节点为 `list`
- 或 `{ "data": [...] }`

推荐样本结构：

```json
{
  "_id": "sample-id",
  "question": "...",
  "answer": "...",
  "supporting_facts": [["Title A", 0], ["Title B", 1]],
  "context": [
    {
      "title": "Title A",
      "sentences": ["Sentence 1.", "Sentence 2."]
    }
  ]
}
```

也兼容：

- `paragraphs`
- `context` 中的 `[title, sentences]`
- `paragraph_text` / `text` / `context`

## 输出格式

### 1. stage1

`build_knowledge_graph_rebel.py` 输出：

```json
{
  "_id": "sample-id",
  "entity_list": ["Entity A", "Entity B"],
  "triples": [
    {
      "head": "Entity A",
      "relation": "located in",
      "tail": "Entity B",
      "triple_type": "RELATION"
    }
  ]
}
```

### 2. 完整 pipeline

`build_knowledge_graph_v5_rebel.py` 输出格式与 `build_knowledge_graph_v5.py` 对齐：

```json
{
  "_id": "sample-id",
  "question": "...",
  "answer": "...",
  "context": [
    {
      "title": "Title A",
      "sentences": ["Sentence 1.", "Sentence 2."]
    }
  ],
  "triple_list": [
    {
      "type": "RELATION",
      "name": "Entity A",
      "description_type": "located in",
      "description": "Entity B",
      "kv_lists": [
        {"key_string": "Entity A located in", "value_string": "Entity B"},
        {"key_string": "the entity that located in Entity B is", "value_string": "Entity A"}
      ]
    }
  ],
  "answer_sufficient": true,
  "missing_links": [],
  "revision_notes": ["added regex bridge: ..."]
}
```

## 依赖

最少需要：

```bash
pip install torch transformers sentencepiece
```

如果你希望模型文件固定到某个目录，可以用 `--hf-cache-dir`。

## 快速开始

### 1. 只跑 stage1

```bash
cd /mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-rebel
python3 build_knowledge_graph_rebel.py \
  --input /path/to/input.json \
  --output /path/to/stage1_output.jsonl \
  --supporting-pages-only
```

### 2. 跑完整 v5 风格输出

默认会启用更稳的启发式 stage2：

```bash
cd /mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-rebel
python3 build_knowledge_graph_v5_rebel.py \
  --input /path/to/input.json \
  --output /path/to/final_output.jsonl \
  --stage-cache-dir ./stage_cache
```

### 3. 如果你想关闭 answer-aware stage2

```bash
python3 build_knowledge_graph_v5_rebel.py \
  --input /path/to/input.json \
  --output /path/to/final_output.jsonl \
  --stage-cache-dir ./stage_cache \
  --disable-answer-aware
```

## 主要参数

- `--model-name`
  - 默认 `Babelscape/rebel-large`
- `--hf-cache-dir`
  - Hugging Face 模型缓存目录
- `--device`
  - `auto`、`cpu`、`cuda`、`cuda:0` 等
- `--batch-size`
  - 逐句生成时的 batch 大小
- `--max-input-length`
  - tokenizer 截断长度
- `--max-new-tokens`
  - decoder 最多生成 token 数
- `--num-beams`
  - beam search 大小
- `--use-all-context-pages`
  - 完整 pipeline 使用全部 context；默认只用 supporting pages
- `--supporting-pages-only`
  - stage1 脚本只看 supporting pages
- `--include-question-entities`
  - 把问题中的简易实体候选并入 `entity_list`
- `--stage-cache-dir`
  - 保存 `stage1/stage2/final` 缓存
- `--disable-answer-aware`
  - 跳过 REBEL 专用 stage2，只保留 stage1 图

## 兼容性说明

当前实现对齐的是：

- 输入数据格式
- 最终输出 JSONL 结构
- `stage_cache_dir / resume / overwrite` 这些主流程能力

没有完全复刻原始 LLM 版的地方：

- `stage1` 的事实抽取质量由 REBEL 和后处理规则决定
- `stage2` 仍然是启发式修图，不是基于 REBEL 的生成式图修复
- 不依赖 `model / api-base / api-key / concurrency` 这套 LLM 参数

## 建议

如果你先想要一个稳定、低改动的替代版本，建议从默认配置开始，也就是：

- REBEL 跑 `stage1`
- 保留默认启用的轻量 `stage2`
- 先观察你数据上的 `triple_list` 分布和召回情况

如果后面你觉得有必要，我可以继续帮你把：

- REBEL 的 relation label 做归一化映射
- 长段落切块策略从“逐句”升级成“滑窗”
- `stage2` 改成更贴近 REBEL 输出风格的补图逻辑
