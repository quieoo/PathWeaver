
import os
import re
import gc
import json
import time
import argparse
import statistics
import sys
import hashlib
import random
from pathlib import Path
from typing import List, Optional, Set

import torch
from transformers import AutoTokenizer



# -------- LlamaIndex (RAG) --------
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.core import Settings
except ImportError:
    from llama_index import Settings

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from kblam.metrics_evaluator import full_evaluation, evaluate_model_outputs
from kblam.utils.dataset import (
    compute_retrieval_recall_stats,
    extract_oracle_contexts,
    get_answer,
    get_question,
    get_sample_id,
    get_supporting_fact_titles,
    iter_row_contexts,
    load_dataset,
    load_queryset,
)

import faiss
try:
    from llama_index.vector_stores.faiss import FaissVectorStore
except Exception:
    # older naming in some versions
    from llama_index.vector_stores.faiss import FAISSVectorStore as FaissVectorStore

# Force imports to resolve to the local experiments/lmcache/common.py helper.
_LOCAL_LMCACHE_DIR = Path(__file__).resolve().parent / "lmcache"
if str(_LOCAL_LMCACHE_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LMCACHE_DIR))

from common import (
    destroy_lmcache_engine,
    get_blend_separator,
    setup_lmcache_environment,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Llama RAG with MuSiQue (TTFT & TPOT)')
    # 数据集参数（建议直接给到 jsonl 路径）
    parser.add_argument('--dataset-path', type=str,
                        default='/mnt/n0/datasets/MuSiQue/musique_ans_v1.0_dev.jsonl',
                        help='MuSiQue 数据集（jsonl）本地路径')
    parser.add_argument('--queryset-path', type=str, default=None,
                        help='可选查询集路径；支持完整 json/jsonl 样本，或包含 evaluated_samples/sample_ids/ids 的 JSON 文件')
    parser.add_argument('--n-samples', type=int, default=10, help='测试样本数量')
    parser.add_argument('--dataset-type', type=str, default='musique', help='数据集类型（musique/squad）')

    # 数据集处理
    parser.add_argument('--mintqa_min_hop', type=int, default=None, help='MintQA 最小跳数')

    # 模型参数
    parser.add_argument('--model-path', type=str,
                        default='/mnt/n0/models/llama3_8B_instruct/',
                        help='vLLM 模型路径（llama/deepseek/qwen 等）')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='vLLM tensor parallel size，默认 1，适合本地 Qwen3-4B')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='vLLM GPU memory utilization')
    parser.add_argument('--max-model-len', type=int, default=8192,
                        help='vLLM max model length')
    parser.add_argument('--max-output-len', type=int, default=16,
                        help='vLLM max output length')

    # 检索参数
    parser.add_argument('--similarity-top-k', type=int, default=5, help='相似度检索 Top-K')
    parser.add_argument('--embedding-model', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2',
                        help='嵌入模型名称')
    parser.add_argument('--embedding-device', type=str, default='cuda', help='嵌入模型设备')
    parser.add_argument('--index-path', type=str,
                        default=None,
                        help='索引路径')
    parser.add_argument('--oracle-retrieval', action='store_true',
                        help='使用 oracle 检索：用 is_supporting 段落直接拼接为上下文')
    parser.add_argument('--kb-size', type=int, default=10, help='知识库段落数量')
    parser.add_argument('--without-knowledge', action='store_true',
                        help='不使用知识库')
    parser.add_argument('--dis_out_path', type=str, default=None, help='输出文件路径')

    parser.add_argument('--use-lmcache', action='store_true',
                        help='启用 LMCache blending')
    parser.add_argument('--lmcache-warmup-batch-size', type=int, default=32,
                        help='LMCache document warm-up batch size；设为 0 可关闭 warm-up')
    parser.add_argument('--lmcache-blend-special-str', type=str, default='# #',
                        help='LMCache blending chunk separator')
    parser.add_argument('--lmcache-chunk-size', type=int, default=256,
                        help='LMCache chunk size')
    parser.add_argument('--lmcache-max-local-cpu-size', type=float, default=50.0,
                        help='LMCache local CPU cache size in GB')
    parser.add_argument('--recompute-ratios', type=float, default=0.15,
                        help='LMCache recompute ratios for blending')
    parser.add_argument('--blend-check-layers', type=str, default='1',
                        help='LMCache blending check layers, comma-separated (e.g. "1" or "1,2,3")')
    parser.add_argument('--lmcache-warmup-mode', type=str, default='chunk', choices=('chunk', 'full', 'reuse'),
                        help='LMCache warmup mode: chunk keeps current segment warmup; full warms a synthetic full prompt with shuffled chunks; reuse warms the exact same prompt')



    args = parser.parse_args()

    return args

def _faiss_vector_store_for_dir(persist_dir: str):
    """Load a persisted FAISS vector store from a LlamaIndex persist directory."""
    if hasattr(FaissVectorStore, "from_persist_dir"):
        return FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
    if hasattr(FaissVectorStore, "from_persist_path"):
        # Some versions persist to a single file path; try the common filename.
        idx_path = os.path.join(persist_dir, "faiss.index")
        return FaissVectorStore.from_persist_path(persist_path=idx_path)
    raise RuntimeError("This LlamaIndex FaissVectorStore does not support loading from disk.")


def get_rag_retriever(docs, args):
    if args.index_path is None:
        raise ValueError("--index-path must be set when using Vector-RAG (FAISS backend).")

    if os.path.exists(args.index_path):
        print(f"Load FAISS index from {args.index_path}")
        faiss_store = _faiss_vector_store_for_dir(args.index_path)
        storage_context = StorageContext.from_defaults(
            persist_dir=args.index_path,
            vector_store=faiss_store,
        )
        index = load_index_from_storage(storage_context)
    else:
        if docs is None:
            raise ValueError("Documents are required to build a new FAISS index.")
        print(f"Index not found in {args.index_path}, create FAISS index from documents...")
        # Probe embedding dimension once (embed model is set in setup_models)
        dim = len(Settings.embed_model.get_text_embedding("dimension probe"))
        # Inner-product flat index; works well if embeddings are normalized.
        # If you prefer L2 distance, switch to faiss.IndexFlatL2(dim).
        print(f"Create FAISS index with dimension {dim}")
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=faiss_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        index.storage_context.persist(persist_dir=args.index_path)
        print(f"FAISS index saved to {args.index_path}")

    retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)
    return retriever


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    deduped_docs: List[Document] = []
    seen = set()
    for doc in docs:
        key = (
            str(getattr(doc, "text", "")).strip(),
            str(getattr(doc, "metadata", {}).get("title", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_docs.append(doc)
    return deduped_docs

def setup_retriever(args, dataset, queryset=None):
    retriever = None
    print(f"setting embed model to {args.embedding_model}")
    # embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
    if os.path.isdir(args.embedding_model):
        # 本地 embedding 模型
        embed_model = HuggingFaceEmbedding(
            model_name=args.embedding_model,
            trust_remote_code=True,
            device=args.embedding_device,
        )
    else:
        # HuggingFace Hub 模型
        embed_model = HuggingFaceEmbedding(
            model_name=args.embedding_model,
            device=args.embedding_device,
        )

    Settings.embed_model = embed_model
    Settings.llm = None

    index_exists = (
        not args.oracle_retrieval
        and not args.without_knowledge
        and args.index_path is not None
        and os.path.exists(args.index_path)
    )

    docs = []
    oracle_contexts_per_sample = []
    if not index_exists and not args.oracle_retrieval and not args.without_knowledge:
        for row in dataset:
            for ctx in iter_row_contexts(row):
                docs.append(
                    Document(
                        text=ctx["text"],
                        metadata={
                            "title": ctx.get("title", ""),
                            "idx": ctx.get("idx"),
                        },
                    )
                )

    oracle_source = queryset if queryset is not None else dataset
    for row in oracle_source:
        oracle_contexts, _ = extract_oracle_contexts(row)
        oracle_contexts_per_sample.append(oracle_contexts)

    if not args.oracle_retrieval and not args.without_knowledge:
        if index_exists:
            print(f"Index exists at {args.index_path}; skip loading documents from dataset.")
        else:
            raw_doc_count = len(docs)
            docs = _dedupe_docs(docs)
            deduped_doc_count = len(docs)
            print(
                f"Loaded {raw_doc_count} documents before dedup; "
                f"{deduped_doc_count} remain after dedup "
                f"(removed {raw_doc_count - deduped_doc_count})."
            )
        # index = VectorStoreIndex.from_documents(docs)
        # retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)
        retriever = get_rag_retriever(docs, args)
    
    return retriever, oracle_contexts_per_sample, docs


def _get_blend_separator(args) -> str:
    return get_blend_separator(args.lmcache_blend_special_str)


def _normalize_chunk_text(text: str) -> str:
    return str(text).strip()


def _join_context_chunks(chunks: List[str], args) -> str:
    clean_chunks = []
    for chunk in chunks:
        clean_chunk = _normalize_chunk_text(chunk)
        if clean_chunk:
            clean_chunks.append(clean_chunk)
    if not clean_chunks:
        return ""
    if args.use_lmcache:
        separator = _get_blend_separator(args)
        return separator + separator.join(clean_chunks) + separator
    return "\n\n".join(clean_chunks)


def _dataset_context_instruction(args) -> str:
    if str(args.dataset_type).lower() != "mintqa":
        return ""
    return (
        "The context is a collection of knowledge-graph triples. "
        "Each context item uses this explicit format:\n"
        "Head: <subject>\n"
        "Relation: <predicate>\n"
        "Tail: <object>\n"
        "Interpret each item as one factual triple, and answer using these triples only.\n\n"
    )


def _collect_unique_warmup_prompts(
    chunks,
    warmed_texts: Optional[Set[str]] = None,
) -> List[str]:
    prompts: List[str] = []
    seen = set()

    for chunk in chunks:
        text = _normalize_chunk_text(getattr(chunk, "text", chunk))
        if not text or text in seen or (warmed_texts is not None and text in warmed_texts):
            continue
        seen.add(text)
        prompts.append(text)

    return prompts


def _load_prompt_tokenizer(model_path: str) -> Optional[AutoTokenizer]:
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[WARN] Failed to load prompt tokenizer for {model_path}: {exc}")
        return None


def _encode_prompt_tokens(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _encode_lmcache_separator_tokens(tokenizer, text: str) -> List[int]:
    token_ids = tokenizer.encode(text)
    return token_ids[1:] if len(token_ids) > 1 else token_ids


def _build_blended_prompt_token_ids(
    prefix_token_ids: List[int],
    context_chunks: List[str],
    suffix_token_ids: List[int],
    args,
    tokenizer,
) -> List[int]:
    prompt_token_ids = list(prefix_token_ids)
    separator_token_ids = _encode_lmcache_separator_tokens(
        tokenizer, _get_blend_separator(args)
    )

    for chunk in context_chunks:
        clean_chunk = _normalize_chunk_text(chunk)
        if not clean_chunk:
            continue
        prompt_token_ids.extend(separator_token_ids)
        prompt_token_ids.extend(_encode_prompt_tokens(tokenizer, clean_chunk))

    if context_chunks:
        prompt_token_ids.extend(separator_token_ids)
    prompt_token_ids.extend(suffix_token_ids)
    return prompt_token_ids


def _pack_short_context_chunks(
    chunks: List[str],
    args,
    tokenizer,
) -> List[str]:
    if not args.use_lmcache or tokenizer is None or len(chunks) <= 1:
        return chunks

    min_chunk_tokens = max(1, int(args.lmcache_chunk_size)*3)
    packed_chunks: List[str] = []
    pending_parts: List[str] = []
    pending_token_count = 0

    for chunk in chunks:
        clean_chunk = _normalize_chunk_text(chunk)
        if not clean_chunk:
            continue

        chunk_token_count = len(_encode_prompt_tokens(tokenizer, clean_chunk))
        if chunk_token_count >= min_chunk_tokens:
            if pending_parts:
                packed_chunks.append("\n".join(pending_parts))
                pending_parts = []
                pending_token_count = 0
            packed_chunks.append(clean_chunk)
            continue

        pending_parts.append(clean_chunk)
        pending_token_count += chunk_token_count
        if pending_token_count >= min_chunk_tokens:
            packed_chunks.append("\n".join(pending_parts))
            pending_parts = []
            pending_token_count = 0

    if pending_parts:
        if packed_chunks:
            packed_chunks[-1] = packed_chunks[-1] + "\n" + "\n".join(pending_parts)
        else:
            packed_chunks.append("\n".join(pending_parts))

    return packed_chunks


def _stable_seed_from_parts(*parts: str) -> int:
    payload = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _build_full_warmup_question(label: str, chunk_count: int) -> str:
    question_id = hashlib.sha256(f"{label}|{chunk_count}".encode("utf-8")).hexdigest()[:8]
    return (
        f"Warm-up query {question_id}: based only on the reordered context chunks, "
        "what is the single most supported entity, fact, or title?"
    )


def _build_shuffled_warmup_chunks(chunks: List[str], label: str) -> List[str]:
    clean_chunks = [_normalize_chunk_text(chunk) for chunk in chunks if _normalize_chunk_text(chunk)]
    if len(clean_chunks) <= 1:
        return clean_chunks

    rng = random.Random(_stable_seed_from_parts(label, *clean_chunks))
    shuffled_chunks = list(clean_chunks)
    rng.shuffle(shuffled_chunks)
    return shuffled_chunks


def _generate_from_token_ids(
    llm,
    prompt_token_ids: List[int],
    sampling_params: Optional[SamplingParams] = None,
):
    prompt = {"prompt_token_ids": prompt_token_ids}
    if sampling_params is None:
        return llm.generate(prompt)
    return llm.generate(prompt, sampling_params=sampling_params)


def _split_prompt_token_ids_by_separator(
    prompt_token_ids: List[int],
    separator_token_ids: List[int],
) -> List[List[int]]:
    if not separator_token_ids or len(prompt_token_ids) < len(separator_token_ids):
        return [list(prompt_token_ids)]

    segments: List[List[int]] = []
    start = 0
    sep_len = len(separator_token_ids)
    i = 0
    limit = len(prompt_token_ids) - sep_len

    while i <= limit:
        if prompt_token_ids[i : i + sep_len] == separator_token_ids:
            segments.append(prompt_token_ids[start:i])
            start = i + sep_len
            i = start
            continue
        i += 1

    segments.append(prompt_token_ids[start:])
    return segments


def _warmup_exact_prompt_segments(
    args,
    llm,
    prompt_tokenizer,
    prompt_token_ids: List[int],
    warmed_lmcache_segments: Set[tuple[int, ...]],
    label: str,
):
    if not args.use_lmcache or prompt_tokenizer is None or args.lmcache_warmup_batch_size <= 0:
        return

    separator_token_ids = _encode_lmcache_separator_tokens(
        prompt_tokenizer, _get_blend_separator(args)
    )
    segments = _split_prompt_token_ids_by_separator(prompt_token_ids, separator_token_ids)

    pending_segments: List[List[int]] = []
    for segment in segments:
        if not segment:
            continue
        segment_key = tuple(segment)
        if segment_key in warmed_lmcache_segments:
            continue
        warmed_lmcache_segments.add(segment_key)
        pending_segments.append(segment)

    if not pending_segments:
        return

    batch_size = max(1, args.lmcache_warmup_batch_size)
    total_batches = (len(pending_segments) + batch_size - 1) // batch_size
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    print(
        f"LMCache warm-up: caching {len(pending_segments)} exact prompt segment(s) "
        f"for {label} in {total_batches} batch(es)"
    )

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(len(pending_segments), start + batch_size)
        for segment_token_ids in pending_segments[start:end]:
            _generate_from_token_ids(
                llm,
                segment_token_ids,
                sampling_params=sampling_params,
            )


def _warmup_full_prompt(
    args,
    llm,
    prompt_tokenizer,
    warmup_chunks: List[str],
    prompt_prefix: str,
    prompt_suffix: str,
    question: str,
    label: str,
) -> None:
    if not args.use_lmcache or prompt_tokenizer is None or args.lmcache_warmup_batch_size <= 0:
        return

    shuffled_chunks = _build_shuffled_warmup_chunks(warmup_chunks, label)
    if not shuffled_chunks:
        return

    warmup_question = _build_full_warmup_question(label, len(shuffled_chunks))
    if question and question in prompt_suffix:
        warmup_suffix = prompt_suffix.replace(question, warmup_question, 1)
    else:
        warmup_suffix = prompt_suffix + f"\nWarm-up Question: {warmup_question}\n"

    warmup_prompt_token_ids = _build_blended_prompt_token_ids(
        _encode_prompt_tokens(prompt_tokenizer, prompt_prefix),
        shuffled_chunks,
        _encode_prompt_tokens(prompt_tokenizer, warmup_suffix),
        args,
        prompt_tokenizer,
    )

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    print(
        f"LMCache warm-up: caching 1 synthetic full prompt for {label} "
        f"with {len(shuffled_chunks)} shuffled chunk(s)"
    )
    _generate_from_token_ids(
        llm,
        warmup_prompt_token_ids,
        sampling_params=sampling_params,
    )


def _warmup_reuse_prompt(
    args,
    llm,
    prompt_token_ids: List[int],
    label: str,
) -> None:
    if not args.use_lmcache or args.lmcache_warmup_batch_size <= 0:
        return
    if not prompt_token_ids:
        return

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    print(f"LMCache warm-up: caching 1 exact full prompt for {label}")
    _generate_from_token_ids(
        llm,
        prompt_token_ids,
        sampling_params=sampling_params,
    )


def _generate_with_lmcache_segments(
    args,
    llm,
    prompt_tokenizer,
    warmed_lmcache_segments: Set[tuple[int, ...]],
    warmup_chunks: List[str],
    context_text: str,
    prompt_prefix: str,
    prompt_suffix: str,
    question: str,
    label: str,
    sampling_params: Optional[SamplingParams] = None,
):
    prompt_text = prompt_prefix + context_text + prompt_suffix

    if not args.use_lmcache or prompt_tokenizer is None:
        gen_start = time.perf_counter()
        if sampling_params is None:
            resp = llm.generate(prompt_text)
        else:
            resp = llm.generate(prompt_text, sampling_params=sampling_params)
        gen_elapsed = time.perf_counter() - gen_start
        return resp[0], gen_start, gen_elapsed, prompt_text

    prefix_token_ids = _encode_prompt_tokens(prompt_tokenizer, prompt_prefix)
    suffix_token_ids = _encode_prompt_tokens(prompt_tokenizer, prompt_suffix)

    # Only warm the shareable prompt prefix so the question-specific suffix
    # remains uncached and does not get spuriously reused across queries.
    warmup_prompt_token_ids = _build_blended_prompt_token_ids(
        prefix_token_ids,
        warmup_chunks,
        [],
        args,
        prompt_tokenizer,
    )

    prompt_token_ids = _build_blended_prompt_token_ids(
        prefix_token_ids,
        warmup_chunks,
        suffix_token_ids,
        args,
        prompt_tokenizer,
    )

    if args.lmcache_warmup_mode == "chunk":
        _warmup_exact_prompt_segments(
            args,
            llm,
            prompt_tokenizer,
            warmup_prompt_token_ids,
            warmed_lmcache_segments,
            label=f"{label} shareable prefix",
        )
    elif args.lmcache_warmup_mode == "full":
        _warmup_full_prompt(
            args,
            llm,
            prompt_tokenizer,
            warmup_chunks,
            prompt_prefix,
            prompt_suffix,
            question,
            label=f"{label} synthetic full prompt",
        )
    elif args.lmcache_warmup_mode == "reuse":
        _warmup_reuse_prompt(
            args,
            llm,
            prompt_token_ids,
            label=f"{label} exact full prompt",
        )
    else:
        raise ValueError(f"Unsupported LMCache warmup mode: {args.lmcache_warmup_mode}")

    gen_start = time.perf_counter()
    
    if warmup_prompt_token_ids == prompt_token_ids:
        resp = _generate_from_token_ids(
            llm,
            prompt_token_ids,
            sampling_params=sampling_params,
        )
    else:
        resp = _generate_from_token_ids(
            llm,
            prompt_token_ids,
            sampling_params=sampling_params,
        )

    gen_elapsed = time.perf_counter() - gen_start
    return resp[0], gen_start, gen_elapsed, prompt_text


def warmup_lmcache_docs(
    args,
    llm,
    chunks,
    tokenizer=None,
    warmed_texts: Optional[Set[str]] = None,
    label: str = "documents",
):
    if not args.use_lmcache or args.lmcache_warmup_batch_size <= 0:
        return

    prompts = _collect_unique_warmup_prompts(chunks, warmed_texts=warmed_texts)
    if not prompts:
        return

    batch_size = max(1, args.lmcache_warmup_batch_size)
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    print(f"LMCache warm-up: caching {len(prompts)} {label} in {total_batches} batch(es)")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(len(prompts), start + batch_size)
        for prompt_text in prompts[start:end]:
            if tokenizer is not None:
                _generate_from_token_ids(
                    llm,
                    _encode_prompt_tokens(tokenizer, prompt_text),
                    sampling_params=sampling_params,
                )
            else:
                llm.generate(prompt_text, sampling_params=sampling_params)
        if warmed_texts is not None:
            warmed_texts.update(prompts[start:end])


def _is_qwen_model(model_path: str) -> bool:
    return "qwen" in model_path.lower()


def _resolve_tensor_parallel_size(args) -> int:
    if args.tensor_parallel_size and args.tensor_parallel_size > 0:
        return args.tensor_parallel_size
    return max(1, torch.cuda.device_count())


def setup_models(args):
    print(f"Loading model from {args.model_path}...")
    prompt_tokenizer: Optional[AutoTokenizer] = None
    base_llm_kwargs = {
        "model": args.model_path,
        "enforce_eager": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
    }
    if args.use_lmcache:
        base_llm_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        )
        prompt_tokenizer = _load_prompt_tokenizer(args.model_path)

    if 'llama' in args.model_path:
    # 使用VLLM加载模型
        llm = LLM(**base_llm_kwargs)
    elif _is_qwen_model(args.model_path):
        tp_size = _resolve_tensor_parallel_size(args)
        print(f"Using tensor_parallel_size={tp_size} for Qwen model")
        qwen_kwargs = {
            **base_llm_kwargs,
            "dtype": "bfloat16",
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "trust_remote_code": True,
            "max_model_len": args.max_model_len,
        }
        llm = LLM(**qwen_kwargs)
        if prompt_tokenizer is None:
            prompt_tokenizer = _load_prompt_tokenizer(args.model_path)
    elif 'deepseek' in args.model_path:
        n_gpus = torch.cuda.device_count()
        print(f"Detected {n_gpus} GPUs")
        deepseek_kwargs = {
            **base_llm_kwargs,
            "dtype": "bfloat16",
            "tensor_parallel_size": n_gpus,
            "gpu_memory_utilization": 0.8, # save some memory for bert model
            "trust_remote_code": True,
            "max_model_len": 16384,
        }
        llm = LLM(**deepseek_kwargs)
    elif 'olmo3-7b' in args.model_path.lower():
        # === 新增：OLMo-3-7B-Instruct ===
        llm = LLM(
            **base_llm_kwargs,
            dtype="bfloat16",
            trust_remote_code=True,   # 必须
            max_model_len=8192,
            # disable_log_stats=True, 
        )
    elif "olmo3-32b" in args.model_path.lower():
        llm = LLM(
            **base_llm_kwargs,
            # quantization="awq",          # ⭐ 关键
            dtype="float16",             # ⭐ 不要用 bfloat16
            trust_remote_code=True,
            # enforce_eager=True,
            max_model_len=8192,          # 可后续调大
            # gpu_memory_utilization=0.90,
            # disable_log_stats=True,
        )


    print("Setup models done")
    return llm, prompt_tokenizer


def normalize_text(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return re.sub(r"[^\w\s]", " ", text)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

SPECIAL_TOKEN_THRESHOLD = 128000  # 排除如 <|eot_id|>=128009 等特殊 token


def _count_non_special_token_ids(token_ids) -> int:
    if not token_ids:
        return 0
    return sum(1 for tid in token_ids if isinstance(tid, int) and tid < SPECIAL_TOKEN_THRESHOLD)


def _extract_num_output_tokens(req_out) -> int:
    outputs = getattr(req_out, "outputs", None) or []
    total = 0
    for output in outputs:
        token_ids = getattr(output, "token_ids", None) or []
        if not token_ids:
            continue

        # vLLM 的 CompletionOutput.token_ids 已经是“生成输出”的 token 序列，
        # 优先直接按长度统计；若全被特殊 token 过滤掉，再回退到原始长度。
        non_special = _count_non_special_token_ids(token_ids)
        total += non_special if non_special > 0 else len(token_ids)
    return total


def _count_prompt_input_tokens(tokenizer, prompt_text: str) -> int:
    if tokenizer is None or not prompt_text:
        return 0

    try:
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(prompt_text)

    non_special = _count_non_special_token_ids(token_ids)
    return non_special if non_special > 0 else len(token_ids)


def _extract_latency_from_request_output(req_out):
    out = {
        "ttft_model_ts": None,
        "prefill_model": None,
        "decode_time": None,
        "num_output_tokens": 0,
        "has_request_metrics": False,
        "ttft_exact": False,
        "decode_exact": False,
    }

    first_token_time = getattr(req_out, "first_token_time", None)
    if first_token_time is not None:
        out["ttft_model_ts"] = float(first_token_time)
        out["ttft_exact"] = True

    out["num_output_tokens"] = _extract_num_output_tokens(req_out)

    m = getattr(req_out, "metrics", None)
    if m is None:
        return out

    out["has_request_metrics"] = True

    # ===== 新版 vLLM (RequestStateStats) =====
    if hasattr(m, "first_token_ts"):
        if out["ttft_model_ts"] is None:
            out["ttft_model_ts"] = float(m.first_token_ts)
            out["ttft_exact"] = True

        # prefill: schedule -> first token
        if hasattr(m, "first_token_latency"):
            out["prefill_model"] = float(m.first_token_latency)
        elif hasattr(m, "scheduled_ts"):
            out["prefill_model"] = float(m.first_token_ts - m.scheduled_ts)

        # decode: first token -> last token
        if hasattr(m, "last_token_ts"):
            out["decode_time"] = max(
                0.0, float(m.last_token_ts - m.first_token_ts)
            )
            out["decode_exact"] = True

    # ===== token 数 =====
    if hasattr(m, "num_generation_tokens"):
        out["num_output_tokens"] = int(m.num_generation_tokens)

    if out["prefill_model"] is None or out["decode_time"] is None:
        print(f"Warning: incomplete metrics: {m}")

    return out


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[k]


def _summarize(name: str, values: List[float], unit: str = "s"):
    if not values:
        print(f"[WARN] No values for {name}")
        return
    mean_v = statistics.fmean(values)
    p50_v = _percentile(values, 50)
    p95_v = _percentile(values, 95)
    print(f"{name}: mean={mean_v:.4f} {unit}, p50={p50_v:.4f} {unit}, p95={p95_v:.4f} {unit}")


def run_vector_rag(args, llm, retriever, oracle_contexts_per_sample, dataset, prompt_tokenizer=None):

    print("Create RAG engine done")
    predictions: List[str] = []
    answers: List[str] = []

    # 统计容器
    stat_retrieval: List[float] = []
    stat_prefill: List[float] = []
    stat_ttft: List[float] = []
    stat_tpot: List[float] = []
    stat_input_tokens: List[int] = []
    stat_tokens: List[int] = []
    stat_e2e: List[float] = []
    stat_retrieval_recall: List[float] = []
    stat_retrieval_hit: List[float] = []
    stat_retrieval_all_hit: List[float] = []
    warmed_lmcache_segments: Set[tuple[int, ...]] = set()
    any_exact_ttft = False
    any_estimated_ttft = False
    any_exact_decode = False
    any_estimated_decode = False

    if args.n_samples is not None and args.n_samples > 0 and len(dataset) > args.n_samples:
        dataset=dataset[:args.n_samples]
    
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_output_len,
        stop=["<|im_end|>", "<|end|>", "<|endoftext|>"],
    )
    if prompt_tokenizer is None:
        prompt_tokenizer = _load_prompt_tokenizer(args.model_path)
    dataset_context_instruction = _dataset_context_instruction(args)

    for i, row in enumerate(dataset):
        question = get_question(row)
        answer = get_answer(row)
        if question is None or answer is None:
            raise KeyError(f"Sample is missing question/answer fields: {get_sample_id(row, i)}")
        answers.append(answer)
        gold_titles = get_supporting_fact_titles(row)
        retrieved_titles: List[str] = []
        warmup_chunks: List[str] = []

        time_start=time.perf_counter()
        oracle_context_found = False
        if args.oracle_retrieval:
            sample_oracle_contexts = oracle_contexts_per_sample[i] if i < len(oracle_contexts_per_sample) else []
            oracle_context_found = bool(sample_oracle_contexts)
            warmup_chunks = _pack_short_context_chunks(
                sample_oracle_contexts,
                args,
                prompt_tokenizer,
            )
            context = _join_context_chunks(warmup_chunks, args)
            retrieved_titles = list(gold_titles)
            if not context:
                sample_id = row.get("id", row.get("_id", i))
                fallback_contexts = [
                    str(ctx.get("text", "")).strip()
                    for ctx in iter_row_contexts(row)
                    if str(ctx.get("text", "")).strip()
                ]
                warmup_chunks = _pack_short_context_chunks(
                    fallback_contexts,
                    args,
                    prompt_tokenizer,
                )
                context = _join_context_chunks(warmup_chunks, args)
                print(
                    f"[WARN] Empty oracle context for sample {i} ({sample_id}); "
                    f"no supporting paragraphs were found. Falling back to all sample contexts."
                )
        elif args.without_knowledge:
            context = ""
        else:
            nodes = retriever.retrieve(question)
            context_chunks = [_normalize_chunk_text(node.get_content()) for node in nodes]
            warmup_chunks = _pack_short_context_chunks(
                context_chunks,
                args,
                prompt_tokenizer,
            )
            context = _join_context_chunks(warmup_chunks, args)
            retrieved_titles = [
                str(getattr(node, "metadata", {}).get("title", "")).strip()
                for node in nodes
            ]
        retrieval_time=time.perf_counter()-time_start

        if args.oracle_retrieval:
            oracle_hit = float(oracle_context_found)
            recall_stats = {"recall": oracle_hit, "hit": oracle_hit, "all_hit": oracle_hit}
        elif args.without_knowledge:
            recall_stats = {"recall": 0.0, "hit": 0.0, "all_hit": 0.0}
        else:
            recall_stats = compute_retrieval_recall_stats(gold_titles, retrieved_titles)


        gen_start = None
        prompt = ""

        if "llama" in args.model_path.lower():
            prompt_prefix = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"You answer questions with ONLY the exact answer phrase. "
                f"Never add explanations, prefixes, or punctuation. "
                f"{dataset_context_instruction}"
                f"Examples:\n"
                f"Question: Who wrote '1984'? -> George Orwell\n"
                f"Question: Capital of France? -> Paris\n"
                f"Question: When was Einstein born? -> 1879\n"
                f"<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Context:\n"
            )
            prompt_suffix = (
                f"\n\nQuestion: {question}\n"
                f"Answer exactly: <|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
            req_out, gen_start, gen_elapsed, prompt = _generate_with_lmcache_segments(
                args,
                llm,
                prompt_tokenizer,
                warmed_lmcache_segments,
                warmup_chunks,
                context,
                prompt_prefix,
                prompt_suffix,
                question,
                label=f"sample {i} final prompt",
                sampling_params=sp,
            )
            pred = str(req_out.outputs[0].text).strip()

        elif "deepseek" in args.model_path.lower():
            prompt_prefix = (
                f"<|system|>\n"
                f"Answer the question using ONLY the given context. "
                f"Output ONLY the exact answer phrase in English. "
                f"Do NOT add any explanations, prefixes, suffixes, or punctuation.\n\n"
                f"{dataset_context_instruction}"
                f"--- Examples ---\n"
                f"Question: Who is the spouse of the Green performer? → Miquette Giraudy\n"
                f"Question: Capital of France? → Paris\n"
                f"Question: When was Einstein born? → 1879\n"
                f"--- End Examples ---\n"
                f"<|user|>\n"
                f"Context:\n"
            )
            prompt_suffix = (
                f"\n\n"
                f"Question: {question}\n"
                f"Answer:\n"
                f"<|assistant|>\n"
            )
            
            req_out, gen_start, gen_elapsed, prompt = _generate_with_lmcache_segments(
                args,
                llm,
                prompt_tokenizer,
                warmed_lmcache_segments,
                warmup_chunks,
                context,
                prompt_prefix,
                prompt_suffix,
                question,
                label=f"sample {i} final prompt",
                sampling_params=sp,
            )
            pred = str(req_out.outputs[0].text).strip()
            # 清理
            try:
                parts = pred.split("</think>")
                if len(parts) > 1:
                    pred = parts[1].strip()
            except Exception:
                pass
            if "Answer: " in pred:
                pred = pred.split("Answer: ")[-1].strip()

        elif "qwen" in args.model_path.lower():
            system_prompt = (
                "You answer questions with ONLY the exact answer phrase. "
                "Never add explanations, prefixes, or punctuation. "
                f"{dataset_context_instruction}"
                "Examples:\n"
                "Question: Who wrote '1984'? → George Orwell\n"
                "Question: Capital of France? → Paris\n"
                "Question: When was Einstein born? → 1879\n"
            )
            prompt_prefix = (
                "<|system|>\n" + system_prompt + "\n<|end|>\n"
                "<|user|>\n"
                "CONTEXT:\n"
            )
            prompt_suffix = (
                f"\n\nQUESTION:\n{question}\n\nAnswer:\n"
                "<|end|>\n"
                "<|assistant|>\n"
            )
            req_out, gen_start, gen_elapsed, prompt = _generate_with_lmcache_segments(
                args,
                llm,
                prompt_tokenizer,
                warmed_lmcache_segments,
                warmup_chunks,
                context,
                prompt_prefix,
                prompt_suffix,
                question,
                label=f"sample {i} final prompt",
                sampling_params=sp,
            )
            pred = str(req_out.outputs[0].text).strip()
            if "<|end|>" in pred:
                pred = pred.split("<|end|>")[0].strip()
            if "Explanation: " in pred:
                pred = pred.split("Explanation: ")[0].strip()
            pred = pred.split("\n")[0].strip()
            if i==0:
                print("===========================")
                print(prompt)
                print("===========================")
                print(req_out)
                print("===========================")
                print(pred)
                print("===========================")                
                
        elif "olmo3-7b" in args.model_path.lower():
            prompt_prefix = (
                "<|system|>\n"
                "You are a helpful assistant. "
                "Answer the question using ONLY the given context. "
                "Output ONLY the exact answer phrase. "
                "Do NOT add explanations or extra text.\n"
                f"{dataset_context_instruction}"
                "<|user|>\n"
                "Context:\n"
            )
            prompt_suffix = (
                "\n\n"
                f"Question: {question}\n"
                "Answer:\n"
                "<|assistant|>\n"
            )
            req_out, gen_start, gen_elapsed, prompt = _generate_with_lmcache_segments(
                args,
                llm,
                prompt_tokenizer,
                warmed_lmcache_segments,
                warmup_chunks,
                context,
                prompt_prefix,
                prompt_suffix,
                question,
                label=f"sample {i} final prompt",
                sampling_params=sp,
            )
            pred = req_out.outputs[0].text.strip()
        elif "olmo3-32b" in args.model_path.lower():
            # ===== 1. 评测专用硬约束 Prompt =====
            prompt_prefix = (
                "You must answer with ONLY the final answer.\n"
                "Rules:\n"
                "- Use ONLY the information in the context.\n"
                "- Output a single short phrase or name.\n"
                "- Do NOT explain.\n"
                "- Do NOT add any extra words.\n"
                "- Do NOT repeat the question.\n"
                "- If the answer is not explicitly stated in the context, output: UNKNOWN.\n"
                f"{dataset_context_instruction}"
                "Context:\n"
            )
            prompt_suffix = (
                "\nQuestion:\n"
                f"{question}\n"
                "Answer:\n"
            )

            req_out, gen_start, gen_elapsed, prompt = _generate_with_lmcache_segments(
                args,
                llm,
                prompt_tokenizer,
                warmed_lmcache_segments,
                warmup_chunks,
                context,
                prompt_prefix,
                prompt_suffix,
                question,
                label=f"sample {i} final prompt",
                sampling_params=sp,
            )
            print(f"==========================\n Prompts: {prompt} \nOutput: {req_out}\n==========================")
            pred = req_out.outputs[0].text.strip()

        else:
            print("Unknown model path (llama/deepseek/qwen).")
            req_out = None
            gen_elapsed = 0.0
            pred = ""

        # ---- 解析 vLLM 指标 & 计算 TTFT/TPOT ----
        metrics = _extract_latency_from_request_output(req_out) if req_out is not None else {}
        ttft_model_ts = metrics.get("ttft_model_ts", None)
        prefill_model = metrics.get("prefill_model", None)
        decode_time = metrics.get("decode_time", None)
        num_out = metrics.get("num_output_tokens", 0)
        num_input = _count_prompt_input_tokens(prompt_tokenizer, prompt)

        ttft_model = None
        if ttft_model_ts is not None and gen_start is not None:
            ttft_model = max(0.0, ttft_model_ts - gen_start)

        # 优先使用真实 TTFT；prefill 仅在底层 metrics 可用时单独记录。
        if prefill_model is None:
            prefill_model = ttft_model

        # 回退策略：decode_time 仍然尽量从总生成时长推导。
        if decode_time is None:
            if ttft_model is not None:
                decode_time = max(0.0, gen_elapsed - ttft_model)
            elif prefill_model is not None:
                decode_time = max(0.0, gen_elapsed - prefill_model)
            else:
                decode_time = max(0.0, gen_elapsed)

        if ttft_model is None:
            print(f"Warning: accurate TTFT unavailable, fallback to prefill_model={prefill_model}")
            ttft_model = max(0.0, prefill_model if prefill_model is not None else gen_elapsed)
            any_estimated_ttft = True
        else:
            any_exact_ttft = True

        if metrics.get("decode_exact", False):
            any_exact_decode = True
        else:
            any_estimated_decode = True

        # TTFT = 检索 + 模型 TTFT
        ttft = retrieval_time + ttft_model
        # TPOT = decode_time / 输出 token 数
        tpot = decode_time / max(1, num_out)

        # ---- 记录统计 ----
        stat_retrieval.append(retrieval_time)
        stat_prefill.append(prefill_model if prefill_model is not None else ttft_model)
        stat_ttft.append(ttft)
        stat_tpot.append(tpot)
        stat_input_tokens.append(num_input)
        stat_tokens.append(num_out)
        stat_e2e.append(retrieval_time + gen_elapsed)
        stat_retrieval_recall.append(recall_stats["recall"])
        stat_retrieval_hit.append(recall_stats["hit"])
        stat_retrieval_all_hit.append(recall_stats["all_hit"])


        predictions.append(pred)

    # ---- 汇总统计 ----
    print("\n========== Latency & Throughput ==========")
    _summarize("Retrieval time", stat_retrieval, "s")
    _summarize("Retrieval recall", stat_retrieval_recall, "")
    _summarize("Prefill time", stat_prefill, "s")
    _summarize("TTFT (retrieval+model TTFT)", stat_ttft, "s")
    _summarize("Num input_tokens", stat_input_tokens, "")
    _summarize("Num output_tokens", stat_tokens, "")
    _summarize("TPOT (time per output token)", stat_tpot, "s/token")
    if any_exact_ttft and not any_estimated_ttft:
        print("TTFT source: exact request first-token timestamp")
    elif any_exact_ttft:
        print("TTFT source: mixed exact first-token timestamp and fallback estimate")
    else:
        print("TTFT source: fallback estimate")
    if any_exact_decode and not any_estimated_decode:
        print("Decode/TPOT source: exact request decode timestamps")
    elif any_exact_decode:
        print("Decode/TPOT source: mixed exact decode timestamps and end-to-end estimate")
    else:
        print("Decode/TPOT source: end-to-end estimate (vLLM offline generate hides per-request decode timing in FINAL_ONLY mode)")
    if stat_retrieval_hit:
        print(f"Retrieval hit@{args.similarity_top_k}: mean={statistics.fmean(stat_retrieval_hit):.4f}")
    if stat_retrieval_all_hit:
        print(f"Retrieval all-support-hit@{args.similarity_top_k}: mean={statistics.fmean(stat_retrieval_all_hit):.4f}")
    if len(stat_e2e) > 0 and sum(stat_e2e) > 0:
        overall_tps = len(stat_e2e) / sum(stat_e2e)  # 含检索在内的整体 tokens/sec
        print(f"Throughput (QPS): {overall_tps:.2f}")
        print(f"Average E2E time: {sum(stat_e2e)/len(stat_e2e):.2f} s")
    if stat_input_tokens:
        print(f"Average input tokens: {statistics.fmean(stat_input_tokens):.2f}")
    print("==========================================\n")

    return predictions, answers

def clean_model(llm, use_lmcache: bool = False):
    if llm is not None:
        if hasattr(llm, "engine") and llm.engine is not None:
            try:
                llm.engine.shutdown()
                print("✅ vLLM engine shutdown complete.")
            except Exception as e:
                print(f"⚠️ engine shutdown failed: {e}")
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        print(f"✅ CUDA memory cleared. Remaining allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    if use_lmcache:
        try:
            destroy_lmcache_engine()
        except Exception:
            pass



def main():
    args=parse_args()
    if args.use_lmcache:
        setup_lmcache_environment(
            chunk_size=args.lmcache_chunk_size,
            blend_special_str=args.lmcache_blend_special_str,
            local_cpu_size_gb=args.lmcache_max_local_cpu_size,
            recompute_ratios=args.recompute_ratios,
            blend_check_layers=args.blend_check_layers,
        )
    dataset=load_dataset(args.dataset_path)
    queryset=load_queryset(args.queryset_path)
    eval_rows = queryset if queryset is not None else dataset
    selected_eval_indices = None

    if args.dataset_type == "mintqa" and args.mintqa_min_hop is not None:
        original_len = len(eval_rows)
        selected_eval_indices = [
            i for i, sample in enumerate(eval_rows)
            if sample.get("metadata", {}).get("support_hops", -1) >= args.mintqa_min_hop
        ]
        eval_rows = [eval_rows[i] for i in selected_eval_indices]
        print(f"Filtering to {len(eval_rows)} samples with hop >= {args.mintqa_min_hop} (original {original_len})")

    retriever, oracle_contexts_per_sample, _ = setup_retriever(args, dataset, queryset=queryset)
    if selected_eval_indices is not None:
        oracle_contexts_per_sample = [
            oracle_contexts_per_sample[i]
            for i in selected_eval_indices
            if i < len(oracle_contexts_per_sample)
        ]
    llm, prompt_tokenizer = setup_models(args)

    predictions, answers = run_vector_rag(
        args,
        llm,
        retriever,
        oracle_contexts_per_sample,
        eval_rows,
        prompt_tokenizer=prompt_tokenizer,
    )

    clean_model(llm, use_lmcache=args.use_lmcache)
    if args.dis_out_path is not None:
        metrics, faith_01_scores = evaluate_model_outputs(predictions, answers)
        num_scored = min(len(predictions), len(faith_01_scores))
        if len(faith_01_scores) != len(predictions):
            print(
                f"Warning: got {len(faith_01_scores)} faithfulness scores for "
                f"{len(predictions)} predictions; only scored samples will be exported."
            )

        #输出前五对 pred-answer 以及对应的 faith_01 分数
        print("\n=== Sample Predictions & Faithfulness Scores ===")
        for i in range(min(5, num_scored)):
            print(f"Q: {get_question(eval_rows[i])}")
            print(f"A: {answers[i]}")
            print(f"P: {predictions[i]}")
            print(f"Faithfulness01 Score: {faith_01_scores[i]}")
            print("-----------------------------------")

        evaluated_samples_ids=[]
        for i in range(num_scored):
            sample_id = get_sample_id(eval_rows[i], i)
            if faith_01_scores[i] == 0:
                evaluated_samples_ids.append(sample_id)
        with open(args.dis_out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluated_samples": evaluated_samples_ids,
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Evaluation {len(evaluated_samples_ids)} results saved to {args.dis_out_path}")
    else:
        comparison_str, metrics = full_evaluation(predictions, answers)
        print(metrics)


if __name__ == "__main__":
    main()
