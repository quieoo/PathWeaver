"""Script for evaluating KB models"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.olmo3.kblam_olmo3_attention import load_kblam_olmo3_model, replace_attention_with_kblam
from kblam.models.qwen3.kblam_qwen3_attention import (
    load_kblam_qwen3_model,
    load_qwen3_query_head,
    resolve_runtime_device,
)
from kblam.models.qwen3.kblam_qwen3_moe_attention import (
    load_kblam_qwen3_moe_model,
    load_qwen3_moe_query_head,
)
from kblam.kblam_attention.kblam_path import (
    enable_path_attn_trace,
    set_path_attn_trace_context,
    backfill_path_attn_trace_records,
    dump_path_attn_trace,
    clear_path_attn_trace,
)

from transformers import AutoModelForCausalLM


# from kblam.utils.data_utils import aug_row, generate_multi_entity_qa
from kblam.utils.data_utils import generate_multi_entity_qa

from kblam.utils.eval_utils import (
    instruction_prompts,
    instruction_prompts_multi_entities,
    zero_shot_prompt,
    zero_shot_prompt_multi_entities,
    format_Q_llama,
    format_Q_phi3,
    answer_question,
    answer_question_deterministic,
    softmax,
    format_output_for_synthetic,
    strip_generation_prefix,
    output_dag_sample,
)
from kblam.utils.train_utils import get_kb_embd
from kblam.kb_retriever import KBRetriever, StageRetriever, get_question_type_sampled_T1, get_question_type_sampled_T2
from kblam.metrics_evaluator import full_evaluation, simple_evaluation
import time
from kblam.models.llama3_model import kblam_profile_get, kblam_profile_reset
from kblam.autoschemakg_kb_retriever import AutoSchemaKGKBRetriever
from kblam.dag_kv_retriever import DAGKVKBRetriever
logging.set_verbosity_warning()
# Preserve the original CUDA default. Set DAG_KV_DEVICE=npu explicitly to
# activate the Ascend-specific execution path.
RUNTIME_DEVICE = resolve_runtime_device(os.getenv("DAG_KV_DEVICE", "cuda"))


def _get_kb_encoder_out_dim(model_config, kb_layer_frequency: int) -> int:
    slots = model_config.num_hidden_layers // kb_layer_frequency + 1
    head_dim = getattr(model_config, "head_dim", None)
    num_heads = getattr(model_config, "num_attention_heads", None)

    if head_dim is not None and num_heads is not None:
        per_slot_dim = head_dim * num_heads
    else:
        per_slot_dim = model_config.hidden_size

    return per_slot_dim * slots

def _prepare_models(
    encoder_spec,
    encoder_path,
    llm_type,
    llm_base_dir,
    model_path,
    query_head_path,
    kb_layer_frequency,
    kb_scale_factor,
):
    is_qwen_family = llm_type in {"qwen3", "qwen3_moe", "qwen_moe"}
    is_qwen_moe = llm_type in {"qwen3_moe", "qwen_moe"}

    tokenizer = AutoTokenizer.from_pretrained(
        llm_base_dir, trust_remote_code=True, padding_side="left"
    )
    if is_qwen_family:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
    else:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "^"

    if llm_type == "llama3":
        if query_head_path:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            model.load_query_head(query_head_path)
        else:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
    elif llm_type == "olmo3":
        model = load_kblam_olmo3_model(
            base_model_dir=llm_base_dir,   # 原始 olmo3-7b
            checkpoint_dir=model_path,      # 这个 stage1_lr_..._step_5600 目录
            device="cuda",
        )
    elif llm_type == "qwen3":
        model = load_kblam_qwen3_model(
            base_model_dir=llm_base_dir,
            checkpoint_dir=model_path,
            device=RUNTIME_DEVICE,
        )
        if query_head_path:
            load_qwen3_query_head(model, query_head_path)
    elif is_qwen_moe:
        model = load_kblam_qwen3_moe_model(
            base_model_dir=llm_base_dir,
            checkpoint_dir=model_path,
            device=None,
            device_map="auto",
        )
        if query_head_path:
            load_qwen3_moe_query_head(model, query_head_path)
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    # config = model.config.to_dict()
    kb_config = KBLaMConfig(
        sep_query_head=True,
        kb_layer_frequency=kb_layer_frequency,
        kb_scale_factor=kb_scale_factor,
    )
    # config.update(kb_config.to_dict())
    # new_config = KBLaMConfig(**config)
    # model.config = new_config

    encoder = KBEncoder(
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=_get_kb_encoder_out_dim(model.config, kb_layer_frequency),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=RUNTIME_DEVICE,
    )
    # print(f"encoder out_dim: {model.config.hidden_size
        # * (model.config.num_hidden_layers // kb_layer_frequency + 1)}")
    if RUNTIME_DEVICE.type == "npu":
        # A CUDA-saved checkpoint cannot be deserialized directly by
        # torch_npu. Load its storage on CPU, then copy it into the NPU model.
        encoder_state_dict = torch.load(
            encoder_path,
            map_location="cpu",
            weights_only=True,
        )
    else:
        # Preserve the original CUDA/CPU checkpoint-loading behavior.
        encoder_state_dict = torch.load(encoder_path, weights_only=True)
    encoder.load_state_dict(encoder_state_dict)
    return tokenizer, encoder, model, kb_config



def write_to_json(
    data: Any, filepath: str, indent: int = 4, encoding: str = "utf-8"
) -> bool:
    """
    Write a dictionary to a JSON file with error handling and formatting options.

    Args:
        data: Dictionary to write to JSON file
        filepath: Path where the JSON file should be saved
        indent: Number of spaces for indentation (default: 4)
        encoding: File encoding (default: 'utf-8')

    Raises:
        TypeError: If data is not a dictionary
    """

    try:
        # Convert string path to Path object
        file_path = Path(filepath)

        # Write the JSON file
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=True,  # For consistent output
                default=str,  # Handle non-serializable objects by converting to string
            )

    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")



def move_true_kb_to_front_inference(kb_embedding, true_index: int):
    """
    适用于推理阶段的版本：
    kb_embedding: (N, D)
    true_index: 正确 key 的索引
    """
    if true_index == 0:
        return kb_embedding  # 已在开头

    kb_keys, kb_vals = kb_embedding

    # 可选，全部置零
    # kb_keys = torch.zeros_like(kb_keys)
    # kb_vals = torch.zeros_like(kb_vals)
    # return (kb_keys, kb_vals)

    # 复制以避免原地修改
    new_keys = kb_keys.clone()
    new_vals = kb_vals.clone()

    true_key = new_keys[true_index].clone()
    true_val = new_vals[true_index].clone()

    # 把第0位和true_index交换
    new_keys[true_index], new_keys[0] = new_keys[0], true_key
    new_vals[true_index], new_vals[0] = new_vals[0], true_val

    # 可选：将true_key和true_val设为空，观察模型结果，可以看到模型精度大幅下降，说明正确答案的位置是对的，并且被掩盖
    # new_keys[0] = torch.zeros_like(true_key)
    # new_vals[0] = torch.zeros_like(true_val)

    # 可选：将除了true_index之外的其他key设为空，观察模型结果
    # new_keys[1:] = torch.zeros_like(new_keys[1:])
    # new_vals[1:] = torch.zeros_like(new_vals[1:])


    return (new_keys, new_vals)


def _resolve_trace_sample_id(row: dict, fallback: int):
    sample_id = row.get("_id", row.get("id", fallback))
    if isinstance(sample_id, (str, int, float)):
        return sample_id
    return str(sample_id)


def _extract_trace_kv_items(row: dict):
    dag = row.get("dag")
    if isinstance(dag, dict):
        kv_nodes = dag.get("kv_nodes") or []
        items = []
        for idx, kv in enumerate(kv_nodes):
            if isinstance(kv, dict):
                items.append(
                    {
                        "kv_id": int(idx),
                        "key": str(kv.get("key", "")),
                        "value": str(kv.get("value", "")),
                        "score": kv.get("score"),
                    }
                )
            else:
                items.append({"kv_id": int(idx), "key": str(kv), "value": "", "score": None})
        return items

    triple_lists = row.get("triple_lists")
    if isinstance(triple_lists, list):
        items = []
        for idx, tri in enumerate(triple_lists):
            if isinstance(tri, dict):
                key = tri.get("key_string")
                if key is None:
                    key = str(tri.get("entity", tri.get("head", "")))
                value = tri.get("value_string")
                if value is None:
                    value = tri.get("description", tri.get("tail", tri.get("value", "")))
                items.append(
                    {
                        "kv_id": int(idx),
                        "key": str(key or ""),
                        "value": str(value or ""),
                    }
                )
            else:
                items.append({"kv_id": int(idx), "key": str(tri), "value": ""})
        return items

    context = row.get("context")
    if isinstance(context, list):
        items = []
        kv_id = 0
        for ctx in context:
            if not isinstance(ctx, dict):
                continue
            for tri in ctx.get("triple_list", []) or []:
                if not isinstance(tri, dict):
                    continue
                items.append(
                    {
                        "kv_id": int(kv_id),
                        "key": str(tri.get("key_string", "")),
                        "value": str(tri.get("description", "")),
                    }
                )
                kv_id += 1
        return items

    triple_list = row.get("triple_list")
    if isinstance(triple_list, list):
        items = []
        kv_id = 0
        for tri in triple_list:
            if not isinstance(tri, dict):
                continue
            kv_lists = tri.get("kv_lists", []) or []
            if isinstance(kv_lists, list) and kv_lists:
                for kv in kv_lists:
                    if not isinstance(kv, dict):
                        continue
                    items.append(
                        {
                            "kv_id": int(kv_id),
                            "key": str(kv.get("key_string", "")),
                            "value": str(kv.get("value_string", "")),
                        }
                    )
                    kv_id += 1
                continue

            items.append(
                {
                    "kv_id": int(kv_id),
                    "key": str(tri.get("key_string", "")),
                    "value": str(tri.get("description", tri.get("value_string", ""))),
                }
            )
            kv_id += 1
        return items

    return []

def _perform_eval_batched(
    *,
    model,
    tokenizer,
    kb_retriever,
    kb_config,
    dataset,
    query_idx,
    kb_size,
    dataset_type,
    hop_num=None,            # None = single-hop, 2 = 2wiki
    use_kb_adj=False,
    filter_fn=None,          # dataset 级过滤（2wiki 用）
    remove_sorry=False,
    enable_retrieval=False,
    verbose=False,
    enable_silver: bool = True,
    output_first_samples: int = 0,
    enable_trace: bool = False,
    trace_dataset_name: str = "",
    dag_kb_size: int = 1,
):

    # ---------- optional filter (2wiki) ----------
    if filter_fn is not None:
        new_idx = []
        for idx in query_idx:
            if filter_fn(dataset[idx]):
                new_idx.append(idx)
        query_idx = new_idx

    query_size = len(query_idx)
    num_batches = (query_size + kb_size - 1) // kb_size
    all_model_outputs, all_answers = [], []
    all_source_indices = []

    TTFTs, TPOTs = [], []
    retrieval_time = 0.0
    start_time = time.time()

    if output_first_samples > 0:
        print(f"Outputting the first {output_first_samples} samples:")
        for i in range(min(output_first_samples, query_size)):
            idx = query_idx[i]
            print(f"{idx}: {output_dag_sample(dataset[idx])}")
            print("-----")
        print("----- End of samples -----")

    for batch_id in tqdm(range(num_batches)):
        start = batch_id * kb_size
        end = min((batch_id + 1) * kb_size, query_size)
        batch_idx = query_idx[start:end]
        batch = [dataset[i] for i in batch_idx]



        # ---------- KB construction ----------
        if hop_num is None:
            kb_adj = None
            if enable_retrieval:
                questions = [row["Q"] for row in batch]
                # 批量检索
                # 打印准确率
                # kb_keys, kb_vals = kb_retriever.get_kb_batch_by_hnsw(questions, kb_size, device=torch.device("cuda"), true_indices=batch_idx)
                # kb_keys, kb_vals = kb_retriever.get_kb_batch_by_hnsw(questions, topk=1, device=torch.device("cuda"))
                # top-1 加 随机KB
                kb_keys, kb_vals = kb_retriever.get_kb_batch_by_hnsw(questions, topk=1, device=RUNTIME_DEVICE, random_sample=kb_size-1)
            else:
                if dataset_type == "all_triples":
                    kb_keys, kb_vals = kb_retriever.get_all_triples_embeddings_batch(
                        batch_idx,
                        device=RUNTIME_DEVICE,
                    )
                else:
                    kb_keys, kb_vals = kb_retriever.get_key_embeddings(batch_idx)
        else:
            if enable_retrieval:
                rerank_policy=2
                print(f"[Retrieve] rerank_policy={rerank_policy}")
                questions = [row["Q"] for row in batch]
                kb_keys, kb_vals, kb_adj = kb_retriever.get_kb_adj_batch_by_hnsw(questions, ann_topk=10, rerank_topk=1, rerank_policy=rerank_policy, device=RUNTIME_DEVICE, random_sample=kb_size-1, hop_num=2, true_indices=batch_idx)
            else:
                if "autoschemakg" in dataset_type:
                    kb_keys, kb_vals, kb_adj = kb_retriever.get_kb_embedding_s(
                        batch_idx,
                        n_gold=1,
                        verbose=verbose,
                    )
                elif dataset_type=="at2qa_2wiki":                    
                    kb_keys, kb_vals, kb_adj=kb_retriever.get_embeddings_at2qa_from_precompute_batch(
                        sample_ids=batch_idx,
                        step=kb_config.current_step,
                        total_steps=kb_config.total_steps,
                        hop_num=2,
                        enable_silver=enable_silver,
                    )
                                        # always run in final stage
                    # kb_keys, kb_vals, kb_adj=kb_retriever.get_embeddings_at2qa_from_precompute_batch(
                    #     sample_ids=batch_idx,
                    #     step=1,
                    #     total_steps=1,
                    #     hop_num=2,
                    # )
                elif dataset_type=="dag":
                    kb_keys, kb_vals, kb_adj = [], [], []
                    for idx in batch_idx:
                        if dag_kb_size > 1:
                            candidate_ids = np.arange(len(dataset))
                            distractor_pool = candidate_ids[candidate_ids != idx]
                            num_distractors = min(dag_kb_size - 1, len(distractor_pool))
                            distractor_ids = (
                                np.random.choice(
                                    distractor_pool,
                                    size=num_distractors,
                                    replace=False,
                                ).astype(int).tolist()
                                if num_distractors > 0
                                else []
                            )
                            sample_ids = [int(idx), *distractor_ids]
                            k, v, a = kb_retriever.get_kb_embedding_s(
                                sample_ids,
                                device=RUNTIME_DEVICE,
                            )
                        else:
                            k, v, a = kb_retriever.get_kb_embedding(
                                idx,
                                device=RUNTIME_DEVICE,
                            )
                        kb_keys.append(k)
                        kb_vals.append(v)
                        kb_adj.append(a)
                else:
                    kb_keys, kb_vals, kb_adj = kb_retriever.get_embeddings_with_adj_2wiki(
                        batch_indices=batch_idx,
                        hop_num=2,
                    )

        # ---------- per-query ----------
        for i, row in enumerate(batch):
            if dataset_type=="at2qa_2wiki":
                # sr=StageRetriever()
                # stage=sr.get_stage(kb_config.current_step, kb_config.total_steps)
                # question_type = sr.get_question_type_stage(stage)
                question_type = get_question_type_sampled_T2(kb_config.current_step, kb_config.total_steps, 1, verbose=True if batch_id==0 and i==0 else False)[0]
                Q = row[question_type]
                A = row["A"]
            else:
                Q = row.get("Q", row.get("question", ""))
                A = row.get("A", row.get("answer", ""))

            #判断kb_keys是否是列表
            if isinstance(kb_keys, list):
                target_kb_keys = kb_keys[i]
                target_kb_vals = kb_vals[i]
                target_kb_adj = kb_adj[i] if hop_num is not None else None
            else:
                target_kb_keys = kb_keys
                target_kb_vals = kb_vals
                target_kb_adj = kb_adj if hop_num is not None else None

            if enable_trace:
                set_path_attn_trace_context(
                    sample_id=_resolve_trace_sample_id(row, batch_idx[i]),
                    dataset=trace_dataset_name or dataset_type,
                    dataset_type=dataset_type,
                    source_index=int(batch_idx[i]),
                    batch_id=int(batch_id),
                    batch_offset=int(i),
                    question=str(Q),
                    answer=str(A),
                    kv_items=_extract_trace_kv_items(row),
                    kb_scale_factor=(
                        None
                        if kb_config.kb_scale_factor is None
                        else float(kb_config.kb_scale_factor)
                    ),
                )

            output = answer_question_deterministic(
                tokenizer,
                model,
                Q,
                kb=(target_kb_keys, target_kb_vals),
                kb_config=kb_config,
                kb_adj=target_kb_adj,
            )

            if verbose:
                print('-------------------')
                print(f"stage: {int(kb_config.current_step / kb_config.total_steps * 5)}")
                print(f"Q: {Q}")
                print(f"A: {A}")
                print(f"Output: {output}")
                print('-------------------')

            if Q in output:
                output = output.split(Q)[1]

            prof = kblam_profile_get()
            kblam_profile_reset()

            prefill_s = prof["prefill_s"]
            decode_s = prof["decode_s"]
            decode_tokens = max(1, prof["decode_tokens"])

            TTFTs.append(prefill_s)
            TPOTs.append(decode_s / decode_tokens)

            model_out = format_output_for_synthetic(
                strip_generation_prefix(output, model)
            )
            gt = format_output_for_synthetic(A)

            if enable_trace:
                backfill_path_attn_trace_records(
                    model_output_raw=str(output),
                    model_output_final=str(model_out),
                    answer_final=str(gt),
                )

            if remove_sorry and "sorry" in model_out.lower():
                continue

            all_model_outputs.append(model_out)
            all_answers.append(gt)
            all_source_indices.append(batch_idx[i])

    end_time = time.time()
    if enable_retrieval:
        kb_retriever.print_metrics()
        retrieval_time = kb_retriever.get_avg_retrieval_time()
    print(f"QPS: {query_size / (end_time - start_time):.2f}")
    print(f"Average Latency: {(end_time - start_time) / query_size:.4f}")
    print(f"Avg TTFT: {np.mean(TTFTs)+retrieval_time:.4f}")
    print(f"Avg TPOT: {np.mean(TPOTs):.4f}")

    return all_model_outputs, all_answers, all_source_indices


def eval_main_process(
    dataset: list[dict],
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    encoder: KBEncoder,
    kb_config: KBLaMConfig,
    kb_retriever: KBRetriever | AutoSchemaKGKBRetriever | DAGKVKBRetriever,
    kb_scale_factor_range: list[float] | None = None,
    kb_scale_factor: float | None = None,
    dataset_type: str = "synthetic",
    seed: int = 0,
    kb_size: int = -1,
    query_size: int = -1,
    enable_retrieval: bool = False,
    enable_silver: bool = True,
    output_first_samples: int = 0,
    enable_trace: bool = False,
    trace_dataset_name: str = "",
    min_hop: int | None = None,
    dag_kb_size: int = 1,
):
    if query_size > len(dataset) or query_size == -1:
        query_size = len(dataset)
    if kb_size > query_size:
        kb_size = query_size
    if seed != 0:
        np.random.seed(seed)
        query_idx = np.random.randint(0, len(dataset), query_size)
    else:
        query_idx = np.arange(query_size)
    
    if min_hop is not None:
        new_query_idx=[]
        for qid in query_idx:
            item=dataset[qid]
            qd=item.get("question_decomposition", {})
            if qd is None:
                break
            if len(qd) < min_hop:
                continue
            new_query_idx.append(qid)
        if len(new_query_idx)>0:
            print(f"Get {len(new_query_idx)} samples from {len(query_idx)} samples with hop >= {min_hop}")
            query_idx=new_query_idx

    if kb_scale_factor_range is not None:
        scale_factor_list=[]
        start=kb_scale_factor_range[0]
        end=kb_scale_factor_range[1]
        while start<=end:
            scale_factor_list.append(start)
            start*=2
    else:
        scale_factor_list=[kb_scale_factor]
    print(f"---- kb_scale_factor_range: {scale_factor_list}")

    results_pair_list=[]
    for sf in scale_factor_list:
        kb_config.kb_scale_factor = sf
        batch_enable_retrieval = enable_retrieval
        if dataset_type == "all_triples":
            filter_fn=None
            hop_num=None
            use_kb_adj=False
            batch_enable_retrieval = False
        elif "2wiki" in dataset_type or "dag" in dataset_type:
            # 2wiki， hotpot_2hop, musique_2hop等两跳数据集
            # def _2hop_filter(row):
            #     ans = format_output_for_synthetic(row["A"])
            #     return ans == row["triple_lists"][1]["description"]
            # filter_fn=_2hop_filter
            filter_fn=None
            hop_num=2
            use_kb_adj=True
        else:
            # squad， Synthetic等单跳数据集
            filter_fn=None
            hop_num=None
            use_kb_adj=False

        model_outputs, answers, source_indices = _perform_eval_batched(
            model=model,
            tokenizer=tokenizer,
            kb_retriever=kb_retriever,
            kb_config=kb_config,
            dataset=dataset,
            query_idx=query_idx,
            kb_size=kb_size,
            dataset_type=dataset_type,
            hop_num=hop_num,
            use_kb_adj=use_kb_adj,
            filter_fn=filter_fn,
            enable_retrieval=batch_enable_retrieval,
            enable_silver=enable_silver,
            output_first_samples=output_first_samples,
            enable_trace=enable_trace,
            trace_dataset_name=trace_dataset_name,
            dag_kb_size=dag_kb_size,
        )
        
        results_pair_list.append((model_outputs, answers, source_indices))
    return results_pair_list, scale_factor_list



def eval_generate(
    args, 
    dataset, 
    tokenizer, 
    encoder, 
    model, 
    kb_config, 
    kb_retriever, 
    enable_silver: bool = True
):
    if args.enable_trace:
        clear_path_attn_trace()
        enable_path_attn_trace(
            True,
            store_raw=True,
            store_kb_normalized=True,
        )
        if not args.path_attn:
            print("[WARN] enable_trace=True but path_attn=False, trace may be empty.")
    else:
        enable_path_attn_trace(False)

    results_pair_list, scale_factor_list = eval_main_process(
        dataset,
        tokenizer,
        model,
        encoder,
        kb_config,
        kb_retriever,
        args.kb_scale_factor_range,
        args.kb_scale_factor,
        args.dataset_type,
        args.seed,
        args.kb_size,
        args.query_size,
        kb_retriever.is_hnsw_ready(),
        enable_silver=enable_silver,
        enable_trace=args.enable_trace,
        trace_dataset_name=args.dataset_type,
        min_hop=args.min_hop,
        dag_kb_size=args.dag_kb_size,
    )
    
    save_dir = None
    if args.save_dir is not None:
        save_dir = Path(args.save_dir) / args.exp_config_name
        save_dir.mkdir(exist_ok=True, parents=True)
    
    for idx, (results_pair, sf) in enumerate(zip(results_pair_list, scale_factor_list)):
        model_outputs, answers, source_indices = results_pair
        
        if args.full_eval:
            gen_results, score_results, faith01 = full_evaluation(
                model_outputs, answers, return_score=True
            )
        else:
            gen_results, score_results = simple_evaluation(model_outputs, answers)
            faith01 = None  # 统一变量作用域，避免后续引用报错
        
        # if torch.cuda.is_available():
        #     mem_cost = torch.cuda.max_memory_reserved("cuda") / (1024**3)  # 转换为GB
        #     score_results["mem_cost_gb"] = round(mem_cost, 2)
        #     torch.cuda.reset_peak_memory_stats("cuda")  # 重置峰值统计
        
        print(f"---- [{idx+1}/{len(results_pair_list)}] kb_scale_factor: {sf}, {score_results}")
        
        if save_dir is not None:
            try:
                json_path = save_dir / f"{args.exp_config_name}-{sf}.json"
                write_to_json(score_results, json_path)
                
                txt_path = save_dir / f"{args.exp_config_name}-{sf}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(gen_results)
                
                if args.full_eval and faith01 is not None:
                    wrong_idx = []
                    for i in range(len(faith01)):
                        if faith01[i] == 0:
                            wrong_idx.append(i)
                    if len(wrong_idx) > 0:
                        wrong_path = save_dir / f"{args.exp_config_name}-{sf}-wrong.txt"
                        with open(wrong_path, "w", encoding="utf-8") as f:
                            for idx_w in wrong_idx:
                                gt = answers[idx_w]
                                pred = model_outputs[idx_w]
                                sample = output_dag_sample(dataset[source_indices[idx_w]])
                                f.write(f"GT: {gt}\tPRED: {pred}\nSAMPLE: {sample}\n\n")
                
                print(f"Results saved to {json_path} and {txt_path}")
            
            except Exception as e:
                print(f"⚠️ Failed to save results for sf={sf}: {str(e)}")

    if args.enable_trace:
        if args.path_attn_trace_path is not None:
            trace_path = Path(args.path_attn_trace_path)
        elif save_dir is not None:
            trace_path = save_dir / f"{args.exp_config_name}-path_attn_trace.pt"
        else:
            trace_path = Path(f"{args.exp_config_name}-path_attn_trace.pt")

        trace_path.parent.mkdir(exist_ok=True, parents=True)
        dump_path_attn_trace(str(trace_path))
        enable_path_attn_trace(False)
        print(f"Path attention trace saved to {trace_path}")
def debug_measure_retrieval_accuracy(kb_retriever: KBRetriever, dataset: list[dict], topk: int = 1):
    # topk=[1, 10, 100, 1000]
    # eval_policy=[1, 2, 3]

    topk=[10]
    eval_policy=[2]
    for k in topk:
        print(f"==========K={k}==========")
        for policy in eval_policy:
            print(f"---------- Policy {policy} ----------")
            if policy == 1:
                kb_retriever.collect_recall_v1(topk=k)
            elif policy == 2:
                kb_retriever.collect_recall_v2(topk=k)
            elif policy == 3:
                kb_retriever.collect_recall_v3_1(topk=k)

parser = argparse.ArgumentParser(description="Evaluation script")

# Add arguments that will be shared across all subcommands
parent_parser = argparse.ArgumentParser(add_help=False)

parent_parser.add_argument(
    "--dataset_dir", type=str, help="Directory containing the dataset"
)
parent_parser.add_argument(
    "--encoder_dir", type=str, help="Directory containing the encoder model"
)
parent_parser.add_argument(
    "--encoder_spec",
    type=str,
    default="OAI",
    help="Specification for the encoder model",
)

parent_parser.add_argument(
    "--kb_layer_frequency",
    type=int,
    default=3,
    help="Frequency of knowledge base layers",
)
parent_parser.add_argument(
    "--kb_scale_factor",
    type=float,
    default=None,
    help="Scaling factor for knowledge base",
)

parent_parser.add_argument(
    "--kb_scale_factor_range",
    nargs=2,
    type=float,
    default=None,
    help="Range of scaling factor for knowledge base",
)

parent_parser.add_argument(
    "--kb_size", type=int, default=200, help="Size of the knowledge base"
)

parent_parser.add_argument(
    "--query_size",
    type=int,
    default=-1,
    help="Number of queries to generate per KB entry",
)

parent_parser.add_argument(
    "--llm_base_dir",
    type=str,
    help="llm to load, can be HF location or local directory",
)
parent_parser.add_argument(
    "--llm_type",
    type=str,
    default="phi3",
    help="Type of language model to use",
)
parent_parser.add_argument(
    "--model_dir", type=str, help="Directory containing the model"
)
parent_parser.add_argument("--save_dir", type=str, default=None, help="Directory to save outputs")
parent_parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parent_parser.add_argument(
    "--test_dataset", type=str, help="Source of test KB (assumes KV pair format)"
)
parent_parser.add_argument(
    "--precomputed_embed_keys_path", type=str, help="Path to precomputed key embeddings"
)
parent_parser.add_argument(
    "--precomputed_embed_values_path",
    type=str,
    help="Path to precomputed value embeddings",
)
parent_parser.add_argument(
    "--query_head_path", type=str, default="", help="Path to load KB head from"
)

parent_parser.add_argument(
    "--dataset_type",
    type=str,
    default="",
    help="Type of dataset to use",
)

parent_parser.add_argument(
    "--debug_flag",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable debug mode",
)

parent_parser.add_argument(
    "--format_short",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use short format",
)
parent_parser.add_argument(
    "--path_attn",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use path attention",
)
parent_parser.add_argument(
    "--enable_trace",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to collect and dump path attention traces",
)
parent_parser.add_argument(
    "--path_attn_trace_path",
    type=str,
    default=None,
    help="Optional output path for path attention trace .pt file",
)
parent_parser.add_argument(
    "--hnsw_index_path",
    type=str,
    default=None,
    help="Path to HNSW index, set to None to disable HNSW retrieval",
)
parent_parser.add_argument(
    "--base_embeder_path",
    type=str,
    default=None,
    help="Path to base embeder model",
)

# control stage
parent_parser.add_argument(
    "--step",
    type=int,
    default=1,
    help="current step",
)
parent_parser.add_argument(
    "--t_step",
    type=int,
    default=1,
    help="total steps",
)
parent_parser.add_argument(
    "--full_eval",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to perform full evaluation"
)
parent_parser.add_argument(
    "--min_hop",
    type=int,
    default=None,
    help="min hop",
)
parent_parser.add_argument("--path_attn_mix_ratio", type=float, default=0.8)
parent_parser.add_argument(
    "--dag_kb_size",
    type=int,
    default=1,
    help="For DAG inference, concatenate this many samples into one KB (1 = no distractors).",
)



# Create subparsers
subparsers = parser.add_subparsers(dest="command", required=True)

# Create the parser for the generation command
gen_parser = subparsers.add_parser(
    "generation", parents=[parent_parser], help="Evaluate generation"
)

gen_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="generation_results",
    help="Name of the experiment configuration",
)

gen_parser.add_argument(
    "--multi_entites",
    type=int,
    default=-1,
    help="Number of entities to process (-1 for unlimited)",
)
gen_parser.add_argument(
    "--remove_sorry",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Filter out "sorry" answers from the output',
)

debug_parser = subparsers.add_parser(
    "debug", parents=[parent_parser], help="Debug retrieval accuracy"
)

def main():
    args = parser.parse_args()

    dataset_path=os.path.join(args.dataset_dir, args.test_dataset)
    # 判断数据集是json还是jsonl格式
    if dataset_path.endswith(".jsonl"):
        dataset=[json.loads(line.strip()) for line in open(dataset_path)]
    elif dataset_path.endswith(".json"):
        dataset=json.load(open(dataset_path))
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")
    
    if "silver" in args.test_dataset:
        enable_silver=True
    else:
        enable_silver=False

    tokenizer, encoder, model, kb_config = _prepare_models(
        args.encoder_spec,
        args.encoder_dir,
        args.llm_type,
        args.llm_base_dir,
        args.model_dir,
        args.query_head_path,
        args.kb_layer_frequency,
        args.kb_scale_factor,
    )
    kb_config.format_short = args.format_short
    kb_config.path_attn = args.path_attn
    kb_config.current_step=args.step
    kb_config.total_steps=args.t_step
    kb_config.path_attn_mix_ratio=args.path_attn_mix_ratio

    if args.dataset_type == "autoschemakg_2wiki":
        kb_retriever = AutoSchemaKGKBRetriever(
            encoder,
            dataset,
            precomputed_embed_keys_path=args.precomputed_embed_keys_path,
            precomputed_embed_values_path=args.precomputed_embed_values_path,
        )
    elif args.dataset_type == "dag":
        kb_retriever = DAGKVKBRetriever(
            encoder,
            dataset,
            precomputed_embed_keys_path=args.precomputed_embed_keys_path,
            precomputed_embed_values_path=args.precomputed_embed_values_path,
            max_kv_per_sample=None,
            use_multihop_adj=True,
            max_hops=10,
            # hop_decay=0.5,
            hop_decay=1,
            dynamic_hops_by_longest_path=True,
        )
    else:
        kb_retriever = KBRetriever(
            encoder,
            dataset,
            precomputed_embed_keys_path=args.precomputed_embed_keys_path,
            precomputed_embed_values_path=args.precomputed_embed_values_path,
            hnsw_index_path=args.hnsw_index_path,
            base_embeder_path=args.base_embeder_path,
        )

    if args.command == "generation":
        eval_generate(
            args,
            dataset,
            tokenizer,
            encoder,
            model,
            kb_config,
            kb_retriever,
            enable_silver=enable_silver,
        )
    elif args.command == "debug":
        debug_measure_retrieval_accuracy(kb_retriever, dataset)
    else:
        raise ValueError(f"command {args.command} not recognised")


if __name__ == "__main__":
    main()
