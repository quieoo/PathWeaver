import argparse
import json
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings",
                                 "text-embedding-v4", "qwen3-embedding-0.6B"])
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="synthetic_data")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_flash_attn2", action="store_true")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--progress", action="store_true", help="显示进度条与速率")
    parser.add_argument("--log_mem_every", type=int, default=0,
                    help="每 N 个批次打印一次显存占用（0 表示不打印）")

    return parser.parse_args()


def _fmt_mb(x):  # Bytes → MB
    return f"{x/1024/1024:.0f} MB"

def safe_encode(model, texts, batch_size, encode_fn, device="cuda",
                show_progress=True, log_mem_every=0):
    """
    自动调整 batch_size 并在 OOM 时缩小重试；
    show_progress=True 时显示 tqdm 进度；log_mem_every>0 时定期打印显存。
    """
    embeddings = []
    i = 0
    pbar = tqdm(total=len(texts), dynamic_ncols=True, disable=not show_progress)
    step = 0

    while i < len(texts):
        bs = batch_size
        success = False
        while not success and bs >= 8:
            try:
                batch = texts[i:i+bs]
                emb = encode_fn(batch)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                embeddings.append(emb.astype(np.float32))
                i += bs
                pbar.update(len(batch))
                step += 1

                # 可选：显存日志
                if log_mem_every > 0 and torch.cuda.is_available() and step % log_mem_every == 0:
                    try:
                        alloc = torch.cuda.memory_allocated()
                        resv  = torch.cuda.memory_reserved()
                        max_alloc = torch.cuda.max_memory_allocated()
                        pbar.write(f"[GPU] alloc={_fmt_mb(alloc)}, reserved={_fmt_mb(resv)}, max={_fmt_mb(max_alloc)}")
                    except Exception:
                        pass

                del emb, batch
                torch.cuda.empty_cache()
                success = True
            except torch.cuda.OutOfMemoryError:
                pbar.write(f"⚠️ OOM at batch={bs}, reducing to {bs//2} ...")
                bs = bs // 2
                torch.cuda.empty_cache()
        if not success:
            pbar.close()
            raise RuntimeError("❌ OOM even at batch=8.")
    pbar.close()
    return np.concatenate(embeddings, axis=0)

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """与模型卡保持一致的 last-token pooling"""
    left_padding = attention_mask[:, 0].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
        return last_hidden_states[batch_idx, seq_lens]


# -------------------------
# 主流程
# -------------------------
if __name__ == "__main__":
    args = parser_args()
    key_strings, value_strings = [], []

    # ---- Step 1: 数据重构 ----
    if args.dataset_type == "multi_wiki_qa_train":
        sid, reformatted_data = 0, []
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line.strip()) for line in f]
        for doc in dataset:
            for triple in doc["triples"]:
                key_strings.append(triple["key_string"])
                value_strings.append(triple["description"])
            reformatted_data.append(json.dumps({**doc, "start_id": sid, "num_triples": len(doc["triples"])}))
            sid += len(doc["triples"])
        output_path = args.dataset_path.replace(".json", "_reformatted.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(reformatted_data))
        args.dataset_path = output_path

    elif args.dataset_type == "musique":
        sid, reformatted_data = 0, []
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for sample in dataset:
            new_paras = []
            for paragraph in sample["paragraphs"]:
                num_triples = len(paragraph["triples"])
                for triple in paragraph["triples"]:
                    key_strings.append(f"The {triple['Relation']} of {triple['Head']} is")
                    value_strings.append(triple["Tail"])
                new_paras.append({**paragraph, "start_id": sid, "num_triples": num_triples})
                sid += num_triples
            reformatted_data.append({**sample, "paragraphs": new_paras})
        # output_path = args.dataset_path.replace(".json", "_reformatted.json")
        output_path=args.dataset_path
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reformatted_data, f, indent=2, ensure_ascii=False)
        args.dataset_path = output_path

    elif args.dataset_type == "synthetic":
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for sample in dataset:
            key_strings.append(str(sample["key_string"]))
            value_strings.append(str(sample["description"]))
    elif args.dataset_type == "2wiki":
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for sample in dataset:
            if len(sample["triple_lists"]) != 2:
                raise ValueError(f"Sample {sample['id']} has {len(sample['triple_lists'])} triple lists, expected 2.")
            for triple in sample["triple_lists"]:
                key_strings.append(triple["key_string"])
                value_strings.append(triple["description"])

    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    # ---- Step 2: 加载模型 ----
    print(f"Computing embeddings for {len(key_strings)} triples using {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def save_npys(base_name, model_tag, key_embeds, value_embeds):
        output_dir = os.path.dirname(args.dataset_path) or "."
        save_name = model_tag.replace("/", "-")
        key_path = os.path.join(output_dir, f"{base_name}_{save_name}_embd_key.npy")
        value_path = os.path.join(output_dir, f"{base_name}_{save_name}_embd_value.npy")
        np.save(key_path, key_embeds)
        np.save(value_path, value_embeds)
        print(f"✅ Saved embeddings → {key_path} / {value_path}")

    base_name = os.path.basename(args.dataset_path).rsplit(".", 1)[0]
    use_qwen = args.model_name == "qwen3-embedding-0.6B"
    model_id_or_path = args.local_model_path or ("Qwen/Qwen3-Embedding-0.6B" if use_qwen else args.model_name)

    try:
        from sentence_transformers import SentenceTransformer
        model_kwargs, tokenizer_kwargs = {}, {}
        if use_qwen:
            tokenizer_kwargs["padding_side"] = "left"
            if args.use_flash_attn2:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["device_map"] = "auto"

        model = SentenceTransformer(model_id_or_path, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs)
        model.to(device)
        try:
            if hasattr(model, "_first_module") and hasattr(model._first_module(), "half"):
                model._first_module().half()
        except Exception:
            print("ℹ️ 半精度加载略过。")

        print("✅ Loaded SentenceTransformer.")

        encode_fn = lambda s: model.encode(s, batch_size=args.batch_size,
                                           normalize_embeddings=True,
                                           convert_to_numpy=True,
                                           show_progress_bar=False,
                                           prompt_name=("query" if use_qwen else None))

        concat = key_strings + value_strings
        concat_embeds = safe_encode(model, concat, args.batch_size, encode_fn)
        key_embeds, value_embeds = concat_embeds[:len(key_strings)], concat_embeds[len(key_strings):]

    except Exception as e:
        print(f"⚠️ SBERT failed ({e}), falling back to transformers.")
        from transformers import AutoTokenizer, AutoModel

        model_kwargs = {}
        if args.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="left", trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id_or_path, trust_remote_code=True, **model_kwargs).to(device)
        model.eval()

        def add_query_instruct(q: str) -> str:
            return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{q}"

        key_inputs = [add_query_instruct(q) for q in key_strings] if use_qwen else key_strings
        value_inputs = value_strings

        def encode_hf(batch):
            batch_dict = tokenizer(batch, padding=True, truncation=True,
                                   max_length=args.max_length, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**batch_dict)
                sent = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                sent = torch.nn.functional.normalize(sent, p=2, dim=1)
            emb = sent.detach().cpu().numpy().astype(np.float32)
            del batch_dict, outputs, sent
            torch.cuda.empty_cache()
            return emb

        key_embeds = safe_encode(model, key_inputs, args.batch_size, encode_hf)
        value_embeds = safe_encode(model, value_inputs, args.batch_size, encode_hf)

    # ---- Step 3: 验证 ----
    print(f"✅ Computed {len(key_embeds)} key embeddings and {len(value_embeds)} value embeddings.")

    # ---- Step 4: 保存 ----
    tag = (args.local_model_path.rstrip("/").split("/")[-1]
           if args.local_model_path else ("Qwen3-Embedding-0.6B" if use_qwen else args.model_name))
    save_npys(base_name, tag, key_embeds, value_embeds)
