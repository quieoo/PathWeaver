# lite-openIE

`lite-openIE` 是一套基于 Stanford CoreNLP OpenIE 的 LLM-Free 知识图谱抽取实现，目标是对齐 [`build_knowledge_graph_v5.py`](/mnt/n0/PathWeaver/docs/scripts/triple_gen/build_knowledge_graph_v5.py) 的输入输出风格，但把 `stage1` 抽取内核替换为官方 OpenIE。

官方参考：

- Stanford CoreNLP OpenIE 文档：<https://stanfordnlp.github.io/corenlp-docs-dev/openie.html>
- CoreNLP Server 文档：<https://stanfordnlp.github.io/CoreNLP/corenlp-server.html>

实现说明：

- `stage1`
  - 通过 CoreNLP Server 调用 `tokenize,ssplit,pos,lemma,depparse,natlog,openie`
  - 可选附加 `ner` / `coref`
  - 生成 `entity_list + triples`
- `stage2`
  - 复用 `triple_gen_no_llm/extractor.py` 里的启发式 answer-aware 修图逻辑
- `final`
  - 复用现有确定性 KV 生成逻辑
  - 输出 `triple_list / answer_sufficient / missing_links / revision_notes`

## 目录文件

- [build_knowledge_graph_openie.py](/mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-openIE/build_knowledge_graph_openie.py)
  - 只跑 `stage1`
- [build_knowledge_graph_v5_openie.py](/mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-openIE/build_knowledge_graph_v5_openie.py)
  - 完整 v5 风格 pipeline
- [extractor_openie.py](/mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-openIE/extractor_openie.py)
  - OpenIE 抽取器和 CoreNLP Server 客户端

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

`build_knowledge_graph_openie.py` 输出：

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

`build_knowledge_graph_v5_openie.py` 输出格式与 `build_knowledge_graph_v5.py` 对齐：

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

## 依赖与启动

这套脚本本身只依赖 Python 标准库，但需要一个可访问的 Stanford CoreNLP Server。

推荐做法是先在另一终端启动服务，例如：

```bash
cd /path/to/stanford-corenlp-4.x.x
java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
  -port 9000 \
  -timeout 120000
```

默认服务地址是 `http://localhost:9000`。

## 快速开始

### 1. 只跑 stage1

```bash
cd /mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-openIE
python3 build_knowledge_graph_openie.py \
  --input /path/to/input.json \
  --output /path/to/stage1_output.jsonl \
  --supporting-pages-only
```

### 2. 跑完整 v5 风格输出

```bash
cd /mnt/n0/PathWeaver/docs/scripts/triple_gen/lite-openIE
python3 build_knowledge_graph_v5_openie.py \
  --input /path/to/input.json \
  --output /path/to/final_output.jsonl \
  --stage-cache-dir ./stage_cache
```

### 3. 启用更激进的 OpenIE

```bash
python3 build_knowledge_graph_v5_openie.py \
  --input /path/to/input.json \
  --output /path/to/final_output.jsonl \
  --openie-no-strict \
  --openie-all-nominals \
  --with-ner \
  --include-question-entities
```

## 主要参数

- `--corenlp-url`
  - CoreNLP Server 地址，默认 `http://localhost:9000`
- `--use-all-context-pages`
  - 完整 pipeline 使用全部 context；默认只用 supporting pages
- `--supporting-pages-only`
  - stage1 脚本只看 supporting pages
- `--include-question-entities`
  - 把问题中的简易实体候选并入 `entity_list`
- `--stage-cache-dir`
  - 保存 `stage1/stage2/final` 缓存
- `--disable-answer-aware`
  - 跳过 stage2 修图
- `--openie-min-confidence`
  - 过滤低置信度 OpenIE triple
- `--openie-no-strict`
  - 对应 `openie.triple.strict=false`
- `--openie-all-nominals`
  - 对应 `openie.triple.all_nominals=true`
- `--openie-resolve-coref`
  - 对应 `openie.resolve_coref=true`
- `--with-ner`
  - 附带 NER token，辅助补充实体
- `--openie-max-entailments-per-clause`
  - 对应官方 `openie.max_entailments_per_clause`

## 兼容性说明

当前实现对齐的是：

- 输入数据格式
- 最终输出 JSONL 结构
- `stage_cache_dir / resume / overwrite / answer-aware` 这些主流程能力

没有完全复刻原始 LLM 版的地方：

- 不依赖 `model / api-base / api-key / concurrency` 等 LLM 参数
- `stage1` 的事实抽取质量由 OpenIE 和后处理规则决定
- `stage2` 仍然是启发式修图，不是生成式图修复

## 建议

如果你更重视召回率，可以尝试：

```bash
python3 build_knowledge_graph_v5_openie.py \
  --input /path/to/input.json \
  --output /path/to/output.jsonl \
  --openie-no-strict \
  --openie-all-nominals \
  --with-ner \
  --include-question-entities
```

如果你更重视稳定性，建议先从默认参数开始。
