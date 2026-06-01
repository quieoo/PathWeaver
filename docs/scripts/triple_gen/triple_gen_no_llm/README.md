# triple_gen_no_llm

`triple_gen_no_llm` 是一套不依赖 LLM 的知识图谱抽取 pipeline。

目录里现在有两个入口：

- `build_knowledge_graph_no_llm.py`
  - 只做 `stage1`
  - 输出 `entity_list + triples`
- `build_knowledge_graph_v5_no_llm.py`
  - 做完整 pipeline
  - 输出格式对齐 `build_knowledge_graph_v5.py`
  - 包含：
    - `stage1` 高召回图抽取
    - `stage2` 启发式 answer-aware 修图 / sufficiency 判定
    - 最终 `triple_list` 的确定性 KV 生成

默认模式只依赖 Python 标准库，可以直接运行。
如果安装了 `spacy` 和英文模型 `en_core_web_sm`，脚本会自动启用更强的实体和依存句法抽取能力。

## 目录结构

- `build_knowledge_graph_no_llm.py`
  - stage1 图抽取入口
- `build_knowledge_graph_v5_no_llm.py`
  - 完整 v5 风格 pipeline 入口
- `extractor.py`
  - 规则抽取、启发式 stage2、KV 生成主逻辑
- `requirements.txt`
  - 可选增强依赖

## 输入格式

输入支持：

- `.jsonl`
- `.json`
- JSON 根节点是 `list`
- 或 `{ "data": [...] }`

每条样本推荐包含：

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
- `context` 中的 `[title, sentences]` 形式
- `paragraph_text` / `text` / `context` 字段

## 输出格式

### 1. stage1 输出

`build_knowledge_graph_no_llm.py` 的输出是 `.jsonl`，每条记录长这样：

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
    },
    {
      "head": "Entity A",
      "relation": "population",
      "tail": "100000",
      "triple_type": "ATTRIBUTE"
    }
  ]
}
```

### 2. 完整 pipeline 输出

`build_knowledge_graph_v5_no_llm.py` 的输出格式对齐 `build_knowledge_graph_v5.py`，是“原始样本 + 最终结果”：

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
  "revision_notes": []
}
```

## 快速开始

### 1. 只跑 stage1

```bash
cd /mnt/n0/PathWeaver/docs/scripts/triple_gen/triple_gen_no_llm
python3 build_knowledge_graph_no_llm.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/no_llm/2wiki_dev_2hop_stage1.jsonl \
  --limit 10
```

### 2. 跑完整 pipeline

```bash
cd /mnt/n0/PathWeaver/docs/scripts/triple_gen/triple_gen_no_llm
python3 build_knowledge_graph_v5_no_llm.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/no_llm/2wiki_dev_2hop_no_llm.jsonl \
  --limit 10
```

### 3. 启用 spaCy 增强

安装依赖：

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

运行 stage1：

```bash
python3 build_knowledge_graph_no_llm.py \
  --input /path/to/input.jsonl \
  --output /path/to/stage1_output.jsonl \
  --use-spacy
```

运行完整 pipeline：

```bash
cd /mnt/n0/PathWeaver/docs/scripts/triple_gen/triple_gen_no_llm
python3 build_knowledge_graph_v5_no_llm.py \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/no_llm/2wiki_dev_2hop_no_llm.jsonl \
  --use-spacy \
  --include-question-entities \
  --limit 10 \
  --stage-cache-dir ./stage_cache
```

## 常用参数

- `--supporting-pages-only`
  - 只使用 supporting pages
- `--limit 100`
  - 只处理前 100 条
- `--resume`
  - 跳过已经写入输出文件的 `_id`
- `--overwrite`
  - 忽略已有输出，重跑
- `--use-spacy`
  - 使用 spaCy 增强抽取
- `--include-question-entities`
  - 将问题里的实体候选并入 `entity_list`
- `--max-triples-per-sentence 32`
  - 控制每句最多保留多少候选三元组
- `--stage-cache-dir ./stage_cache`
  - 为完整 pipeline 保存 `stage1/stage2/final` 缓存
- `--disable-answer-aware`
  - 跳过启发式 stage2，只保留 stage1 图
- `--max-hops 4`
  - 控制 answer bridge 搜索允许的关系跳数

## 推荐运行方式

如果你现在主要想做一个“尽量别漏”的高召回 stage1 baseline，建议：

```bash
python3 build_knowledge_graph_no_llm.py \
  --input /path/to/input.jsonl \
  --output /path/to/stage1_output.jsonl \
  --use-spacy \
  --include-question-entities \
  --max-triples-per-sentence 48
```

如果你想直接得到与 `build_knowledge_graph_v5.py` 相同最终格式的输出，建议：

```bash
python3 build_knowledge_graph_v5_no_llm.py \
  --input /path/to/input.jsonl \
  --output /path/to/final_output.jsonl \
  --use-spacy \
  --include-question-entities \
  --stage-cache-dir ./stage_cache
```

如果你更在意速度和零依赖：

```bash
python3 build_knowledge_graph_v5_no_llm.py \
  --input /path/to/input.jsonl \
  --output /path/to/final_output.jsonl
```

## Pipeline 说明

这套 baseline 采用“多路候选 + 后处理去重 + 启发式补图”的方式：

1. 从标题、句子中的大写短语、可选 NER 结果中收集实体候选
2. 通过规则抽取：
   - `X is/was Y`
   - `X in Y`
   - `X from Y`
   - `X by Y`
   - `X's attribute is value`
   - `<title> ... population/date/year/...`
3. 若启用 spaCy，再加入：
   - NER 实体
   - 基于依存关系的 `subject-verb-object`
4. 对候选结果做：
   - 关系规范化
   - `RELATION` / `ATTRIBUTE` 判定
   - 去重
   - 占位实体过滤
5. 完整 pipeline 再做启发式 stage2：
   - 从问题中找 anchor
   - 在图里查从 anchor 到 answer 的显式桥接链
   - 若不足，则在包含 anchor / answer 的句子上做局部补图
6. 最后确定性生成：
   - `triple_list`
   - 正向 / 反向 `kv_lists`

## 局限性

这是一个 baseline，不是语义上等价于 LLM 的替代品。常见局限：

- 多跳隐式关系不如 LLM
- 共指消解和 stage2 修图都只做了轻量启发式
- relation wording 可能不如 LLM 自然
- 复杂长句的漏抽和误抽会明显增加
- `answer_sufficient` 的判断是图搜索 + 规则补图，不等价于 LLM 推理

如果你后面想继续加强，优先建议补：

1. 更强的共指消解
2. OpenIE 集成
3. relation normalization 词典
4. 面向数据集的属性模板库
