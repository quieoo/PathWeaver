from bert_score import score

cands = ["Paris"]
refs = ["Paris"]

# 生成100组测试数据
for i in range(100):
    cands.append(f"Paris {i}")
    refs.append(f"Paris {i}!")



P, R, F1 = score(
    cands,
    refs,
    model_type="microsoft/deberta-xlarge-mnli",
    device="cpu",
    batch_size=4,
    lang="en",
    verbose=True,
    # use_fast_tokenizer=True,
)

print(P, R, F1)