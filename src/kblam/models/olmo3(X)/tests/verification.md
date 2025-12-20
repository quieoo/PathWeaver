下面给你一套**“由弱到强、逐步递增”的验证方案**，目标是：
👉 **在每一步都能明确判断：问题出在“权重 / 结构 / mask / RoPE / KBLaM 扩展”的哪一层级**
👉 **每一步都有明确的通过标准（pass / fail 信号）**

这套方案是 **工程验证路径**，不是论文式描述。

---

# 总体原则（先看这个）

1. **永远只验证一件事**
2. **永远有 reference（HF 官方或上一步）**
3. **从“无 KB、无 cache、短 prompt”开始**
4. **从 logits → hidden_states → attention 逐级深入**
5. **每一步失败，都能明确定位模块**

---

# Phase 0：环境与不变量验证（5 分钟）

### 0.1 确认 tokenizer 不变量（一次性）

**目标**：排除 tokenizer 差异导致的一切假问题

**操作**

```python
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
print(tok.encode("The capital of France is"))
print(tok.decode(tok.encode("The capital of France is")))
```

**通过标准**

* token id 序列与 HF 4.57 完全一致
* 无额外 `bos` / 重复 `system`

❌ 若失败 → 不进入后续步骤

---

### 0.2 权重完整性 sanity check

**操作**

```python
model, cfg = load_kblam_olmo3_from_local(...)
print("lm_head loaded:", ...)
```

**通过标准**

* `lm_head loaded: True`
* `unexpected missing keys = 0`

❌ 若失败 → loader / remap 问题，直接修

---

# Phase 1：最小 forward 正确性（无对齐）

> **不关心“对不对 HF”，只关心“逻辑是否自洽”**

### 1.1 单 batch / 短序列 forward

```python
x = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
out = model(input_ids=x, use_cache=False)
```

**通过标准**

* 无报错
* `out.logits.shape == (1, 8, vocab_size)`
* logits 非 NaN / Inf

📌 这一关只验证：

* mask shape
* sliding window 不炸
* QKV / RoPE / norm 能跑通

---

### 1.2 确定“无 KB 等价于纯 OLMo3”

**操作**

* 不传 `kb_kvs`
* 不传 `kb_adj`
* 不传 `kb_config`

```python
out1 = model(input_ids=x, use_cache=False)
out2 = model(input_ids=x, use_cache=False)
```

**通过标准**

* `out1.logits == out2.logits`（bitwise 或 atol=0）

❌ 若失败 → 模型存在 hidden state 污染（clone / in-place）

````bash
python test1.py
````

---

# Phase 2：与 HF 的 logits 对齐（最关键）

> 从这里开始，**每一步都是“硬对齐”**

---

## Phase 2.1：HF vs 你实现（最简单设置）

### 设置（非常重要）

| 项               | 值                            |
| --------------- | ---------------------------- |
| prompt          | `"The capital of France is"` |
| batch           | 1                            |
| use_cache       | False                        |
| max length      | 不 generate                   |
| dtype           | bf16                         |
| last_token_only | ✅                            |

---

### 2.1.1 导出 HF reference

在 **Transformers ≥4.57**：

```bash
conda activate olmo3

python export_hf_ref_logits.py \
  --model_dir /home/sdu/zhu/models/olmo3-7b/ \
  --text "The capital of France is" \
  --out ref_last.pt \
  --dtype bfloat16 \
  --device cuda \
  --last_token_only
```

---

### 2.1.2 对齐你自己的实现

```bash
conda activate kblam

python compare_logits.py \
  --model_dir /home/sdu/zhu/models/olmo3-7b/ \
  --text "The capital of France is" \
  --ref ref_last.pt \
  --last_token_only
```

---

### 通过标准（分级）

| 等级     | 标准                     |
| ------ | ---------------------- |
| ✅ 优秀   | MAE < 1e-4，Top1 = 1.0  |
| ⚠️ 可接受 | MAE < 1e-3，Top1 ≥ 0.99 |
| ❌ 失败   | Top1 < 0.95            |

❌ 若失败 → **不要进入 KB / cache / generate**

---

## Phase 2.2：逐层定位 logits 偏差（失败时）

如果 Phase 2.1 失败：

### 2.2.1 hook hidden states（最后一层）

在你模型里临时加：

```python
return CausalLMOutputWithPast(
    logits=logits,
    hidden_states=hidden_states,   # 只保留最后一层
    ...
)
```

对比：

* HF 最后一层 hidden_state
* 你最后一层 hidden_state

📌 若 hidden 对齐、logits 不对 → `lm_head` / dtype
📌 若 hidden 不对 → attention / RoPE / norm

---

### 2.2.2 层级二分法（强烈推荐）

* 只跑前 `k` 层（k=4,8,16,32）
* 比较最后 hidden

👉 **5 分钟内就能锁定是“前半段”还是“后半段”**

---

# Phase 3：cache 行为验证（prefill / decode）

> 这一步只在 Phase 2 完全通过后进行

---

### 3.1 prefill vs no-cache 等价性

```python
out_full = model(input_ids=x, use_cache=False)

out_prefill = model(input_ids=x[:, :-1], use_cache=True)
out_decode  = model(input_ids=x[:, -1:], past_key_values=out_prefill.past_key_values)
```

**通过标准**

```text
out_full.logits[:, -1] ≈ out_decode.logits[:, -1]
```

❌ 若失败 → cache_position / RoPE / KV update 问题

---

# Phase 4：KBLaM 注入验证（结构正确性）

> **只验证“行为正确”，不验证“效果好”**

---

### 4.1 注入 dummy KB（可控）

```python
kb_kvs = (
  torch.zeros(kb_len, hidden_size),
  torch.zeros(kb_len, hidden_size),
)
```

**通过标准**

* logits 与无 KB 时 **几乎一致**
* Top1 不发生系统性变化

👉 验证：
**KB 不应在 score_shift=off 时“凭空生效”**

---

### 4.2 Path attention 开 / 关对比

```python
kb_config.path_attn = False
out1 = model(...)

kb_config.path_attn = True
out2 = model(...)
```

**通过标准**

* logits 有变化
* 变化只在 kb_len 对应区域

---

# Phase 5：generate 行为验证（最后一步）

> **这是最容易误判的一步，必须最后做**

### 5.1 greedy generate 对齐（短生成）

```text
max_new_tokens = 5
do_sample = False
```

**通过标准**

* 生成 token 序列与 HF 完全一致

❌ 若失败但 Phase 2 全过 →
**问题 100% 在 `prepare_inputs_for_generation`**

---

# 一句话总结（给你一个“总控表”）

| Phase | 目标       | 若失败，问题一定在               |
| ----- | -------- | ----------------------- |
| 0     | 不变量      | tokenizer / 权重          |
| 1     | 能跑       | mask / shape            |
| 2     | 数值等价     | attention / RoPE / norm |
| 3     | cache 正确 | KV / position           |
| 4     | KB 安全    | KB 拼接 / mask            |
| 5     | generate | HF glue code            |

---

如果你愿意，下一步我可以直接帮你：

* ✅ **给你一个“Phase 2.2 逐层对齐的最小 hook 代码”**
* ✅ **帮你把 Phase 3 的 cache bug 一次性扫干净**
* ✅ **直接判断你现在卡在哪个 Phase（基于你已有日志）**

你只要告诉我：
**“我现在 Phase 2.1 的 MAE / Top1 是多少”**
