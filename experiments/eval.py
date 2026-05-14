"""Script for evaluating KB models"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import nltk
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
# from kblam.utils.data_utils import aug_row, generate_multi_entity_qa
from kblam.utils.data_utils import generate_multi_entity_qa

from kblam.utils.eval_utils import (
    instruction_prompts,
    instruction_prompts_multi_entities,
    zero_shot_prompt,
    zero_shot_prompt_multi_entities,
    format_Q_llama,
    format_Q_phi3,
    model_prune_format_mapping,
    answer_question,
    answer_question_deterministic,
    softmax,
    format_output_for_synthetic,
    strip_generation_prefix,
)
from kblam.utils.train_utils import get_kb_embd
from kblam.kb_retriever import KBRetriever
from kblam.metrics_evaluator import full_evaluation
import time
from kblam.models.llama3_model import kblam_profile_get, kblam_profile_reset


debug_flag = True
logging.set_verbosity_warning()



def perform_eval_musique(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    kb_config: KBLaMConfig,
    eval_mode: str = "kb",
    kb_size: int = 1,
    seed: int = 1,
    query_size: int = 100,
):
    def _normalize_blocks(paragraphs, sample_id: int):
        """将( start_id, num_triples, para )按start_id排序并做重叠/越界校验"""
        blocks = []
        for p in paragraphs:
            s, n = int(p["start_id"]), int(p["num_triples"])
            if n <= 0:
                continue
            # 越界预检
            if s + n > len(kb_retriever.key_embds):
                raise IndexError(
                    f"[sample {sample_id}] start_id overflow: start={s}, num={n}, "
                    f"total={len(kb_retriever.key_embds)}"
                )
            blocks.append((s, n, p))

        if not blocks:
            return []

        # 排序
        blocks.sort(key=lambda x: x[0])

        # 重叠/单调性校验
        for j in range(1, len(blocks)):
            prev_s, prev_n, _ = blocks[j - 1]
            cur_s,  cur_n,  _ = blocks[j]
            prev_end = prev_s + prev_n
            assert prev_end <= cur_s, (
                f"[sample {sample_id}] KB blocks overlap or out-of-order: "
                f"({prev_s},{prev_n}) -> ({cur_s},{cur_n})"
            )
        return blocks
    # get samples to query
    query_size = min(query_size, len(kb_retriever.dataset))
    if seed < 0:
        sample_idx=np.arange(query_size)
    else:
        np.random.seed(seed)
        sample_idx=np.random.randint(0, len(kb_retriever.dataset), query_size)
    if debug_flag:
        query_size=1
        # sample_idx=[22]
        sample_idx=[24]
        
        
    
    test_samples=[kb_retriever.dataset[idx] for idx in sample_idx]
    
    # get kb embeddings
    start_ids = [[] for _ in range(query_size)]
    num_triples = [[] for _ in range(query_size)]
    paras=[[] for _ in range(query_size)]

    for i in range(query_size):
        sample = test_samples[i]
        paragraphs = sample["paragraphs"]

        if kb_size == -1:
            start_ids[i] = [paragraphs[0]["start_id"]]
            total_triples = sum(p["num_triples"] for p in paragraphs)
            num_triples[i] = [total_triples]
            
        elif kb_size == 1:
            # Use only ground-truth supporting paragraphs
            support_paras = []
            for qd in sample["question_decomposition"]:
                true_idx = int(qd["paragraph_support_idx"])
                support_paras.append(paragraphs[true_idx])

            blocks = _normalize_blocks(support_paras, sample_id=i)

            start_ids[i]   = [s for (s, n, p) in blocks]
            num_triples[i] = [n for (s, n, p) in blocks]
            paras[i]       = [p for (s, n, p) in blocks]

            assert all(start_ids[i][k] <= start_ids[i][k+1] for k in range(len(start_ids[i]) - 1)), \
                f"[sample {i}] start_ids not non-decreasing after normalization."
        else:
            raise ValueError(f"Unsupported kb_size: {kb_size}")   
    if debug_flag:
        print(f"sample {sample['id']} chosen triples {start_ids[i]} {num_triples[i]}")
        # 逐条打印三元组id加内容
        for para_in_sample in paras:
            t_id=0
            for para in para_in_sample:
                print(f"True para: {para['paragraph_text']}")
                triples=para["triples"]
                for triple in triples:
                    print(f"Triple {t_id} : {triple}")
                    t_id+=1
            
    kb_embeddings = kb_retriever.get_embeddings(start_ids, num_triples, query_size, is_inference=True)

    retrieval_time=0.0003

    kb_keys, kb_values = kb_embeddings
    
    full_outputs = []
    model_outputs = []
    answers = []
    start_time=time.time()
    TTFTs=[]
    TPOTs=[]
    for i in range(query_size):
        sample = test_samples[i]
        Q=sample["question"]
        answer=sample["answer"]
        kb_i=(kb_keys[i], kb_values[i])
        # zoro Value 测试
        # kb_i=(kb_keys[i], torch.zeros_like(kb_values[i]))
        
        kblam_profile_reset()
        if debug_flag:
            model_output=answer_question_deterministic(
                tokenizer,
                model,
                Q,
                kb=kb_i,
                kb_config=kb_config,
                save_attention_weights=True,
                attention_save_loc="./attn_weights/",
                attention_file_base_name=f"debug_kbscale{kb_config.kb_scale_factor}_sample-{sample['id']}",
                # save_attn_weights_policy="all-step-last-layer",
            )
            print(f"model_output: {model_output}")
            model_output=model_output.split(Q)[1]
        else:
            model_output=answer_question_deterministic(
                tokenizer,
                model,
                Q,
                kb=kb_i,
                kb_config=kb_config,
            )
            if Q in model_output:
                model_output=model_output.split(Q)[1]

        prof = kblam_profile_get()
        prefill_s = prof["prefill_s"]                # 模型 prefill
        decode_s = prof["decode_s"]
        decode_tokens = max(1, prof["decode_tokens"])
        tpot = decode_s / decode_tokens             # ms/token 可乘 1000
        ttft = retrieval_time + prefill_s

        TTFTs.append(ttft)
        TPOTs.append(tpot)

        model_output = strip_generation_prefix(model_output, model)
        full_outputs.append((model_output,answer))
        answers.append(answer)
        print(f"------------------sample {i}----------")
        print(f"Q: {Q}")
        print(f"PRED: {model_output}")
        print(f"GT: {answer}")
        print("---------------------------------------")

        # process the model_output
        model_outputs.append(model_output)
    
    end_time=time.time()
    print(f"query per second: {query_size/(end_time-start_time)}")

    print(f"Average TTFTs: {np.mean(TTFTs)}")
    print(f"Average TPOTs: {np.mean(TPOTs)}")

    if debug_flag:
        exit(0)

    return model_outputs, answers
    


def perform_eval(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    kb_config: KBLaMConfig,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    multi_entites: int = -1,
    remove_sorry: bool = False,
    query_size: int = 250,
):
    # np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)

    # if debug_flag:
    #     query_size=1

    print(f"kb_idx: {kb_idx}")
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    key_str = [row["key_string"] for row in test_kb]
    value_str = [row["description"] for row in test_kb]
    prompt_strs = ""
    for k, v in zip(key_str, value_str):
        prompt_strs += f"{k} is {v}; "

    kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

    model_outputs = []
    answers = []
    full_outputs = []
    # answer_question
    subset_size = min(
        query_size, len(test_kb)
    )  # Regardless of KB size, always test 250 questions, otherwise it will be too slow
    # subset_size = 50
    # for row in tqdm(test_kb[:subset_size]):
    # for i in range(subset_size):
    for i in tqdm(range(subset_size)):
        row=test_kb[i]
        idx=kb_idx[i]
        if multi_entites == -1:
            Q = row["Q"]
            answer = row["A"]
        else:
            kb_subset_idx = np.random.randint(0, len(test_kb), multi_entites)
            Q, A = generate_multi_entity_qa(
                [test_kb[i]["name"] for i in kb_subset_idx],
                [test_kb[i]["description_type"] for i in kb_subset_idx],
                [test_kb[i]["description"] for i in kb_subset_idx],
            )
            answer = A

        # k_emb, v_emb=(kb_embedding)
        # # 切割kb embed，只要其中第一个Token
        # # row_i = k_emb[i:i+1, :]
        # # v_i = v_emb[i:i+1, :]
        # row_i = k_emb[0:1, :]
        # v_i = v_emb[0:1, :]
        # kb_i=(row_i, v_i)
        kb_i=kb_embedding

        # 可选，移动正确 key 到开头，用于验证模型是否能正确定位答案
        kb_i = move_true_kb_to_front_inference(
            kb_i,
            i,
        )

        if eval_mode == "kb":
            model_output = answer_question_deterministic(
                tokenizer,
                model,
                Q,
                kb=kb_i,
                kb_config=kb_config,
                save_attention_weights=True,
                attention_save_loc="./attn_weights_kblam/",
                attention_file_base_name=f"kb-{idx}",
                # save_attn_weights_policy="all-step-last-layer",
            ).split(Q)[1]
            # 去除输出中的前两个换行符
            model_output = strip_generation_prefix(model_output, model)
            # print(f"raw model output: {model_output}")
        elif eval_mode == "icl":
            if multi_entites != -1:
                ins_prompt = instruction_prompts_multi_entities
            else:
                ins_prompt = instruction_prompts
            model_output = answer_question(
                tokenizer,
                model,
                ins_prompt + prompt_strs + Q,
                kb=None,
                kb_config=kb_config,
            )
            if Q in model_output:
                model_output=model_output.split(Q)[1]
        elif eval_mode == "zeroshot":
            if multi_entites != -1:
                ins_prompt = zero_shot_prompt_multi_entities
            else:
                ins_prompt = zero_shot_prompt
            model_output = answer_question(
                tokenizer, model, ins_prompt + Q, kb=None, kb_config=kb_config
            )
            if Q in model_output:
                model_output=model_output.split(Q)[1]
        # print(model_output)
        if remove_sorry:
            if "sorry" in model_output:
                continue
        full_outputs.append((model_output, answer))
        
        answers.append(row["description"])
        model_outputs.append(format_output_for_synthetic(model_output))
        # if multi_entites == -1:
        #     pattern = r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)'
        #     match = re.search(pattern, model_output)
        #     answers.append(row["description"])
        #     if match:
        #         model_output = match.group(1)
        # else:
        #     pattern = r"(?:is|are) (.*?)(?:\.|;)"
        #     matches = re.findall(pattern, model_output)
        #     model_output = "; ".join(matches)
        #     answers.append(";".join(re.findall(r"(?:is|are) (.*?);", answer)))
        # model_outputs.append(model_output)


    for pred, gt in zip(model_outputs, answers):
        print(f"PREDICTION: {pred}")
        print(f"GT: {gt}")
    print(f"KB size: {kb_size}, mode: {eval_mode}")

    return model_outputs, answers
    # results, results_dict=full_evaluation(model_outputs, answers)
    # return results, results_dict


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

def perform_eval_2wiki(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    kb_config: KBLaMConfig,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    multi_entites: int = -1,
    remove_sorry: bool = False,
    query_size: int = 250,
):

    if eval_mode != "kb":
        raise ValueError(f"eval_mode={eval_mode} is not supported for batched evaluation")

    if seed < 0:
        np.random.seed(None)
    else:
        np.random.seed(seed)

    # if query_size <= kb_size:
    #     raise ValueError(f"query_size={query_size} must be greater than kb_size={kb_size}")
    dataset = kb_retriever.dataset
    total_N = len(dataset)
    query_idx=np.random.randint(0, total_N, query_size)
    subset=[dataset[idx] for idx in query_idx]

    # 过滤掉三元组提取质量不够的样本
    new_subset = []
    new_query_idx=[]
    for row, idx in zip(subset, query_idx):
        ans=format_output_for_synthetic(row["A"])
        des_2=row["triple_lists"][1]["description"]
        if ans == des_2:
            new_subset.append(row)
            new_query_idx.append(idx)
    subset=new_subset
    query_idx=new_query_idx
    query_size=len(subset)    
    print(f"After filtering, query_size={query_size}")



    # 统一的结果容器
    all_model_outputs, all_answers = [], []

    # 计算批次数
    num_batches = (query_size + kb_size - 1) // kb_size

    print(f"[INFO] Eval mode={eval_mode}, KB size={kb_size}, Query size={query_size}, total_batches={num_batches}")
    
    start_time=time.time()
    TTFTs=[]
    TPOTs=[]
    retrieval_time=0.003
    
    for batch_id in tqdm(range(num_batches)):
        # ---- 当前批的 query 索引范围 ----
        start_idx = batch_id * kb_size
        end_idx = min((batch_id + 1) * kb_size, query_size)
        current_query_num = end_idx - start_idx
        
        key_embeddings, value_embeddings, kb_adj=kb_retriever.get_embeddings_with_adj_2wiki(
            batch_indices=query_idx[start_idx:end_idx],
            hop_num=2,
        )
        
        query_subset=subset[start_idx:end_idx]

        model_outputs, answers = [], []
        for i in range(current_query_num):
            row = query_subset[i]
            Q, A = row["Q"], row["A"]

            # 可选，移动正确 key 到开头，用于验证模型是否能正确定位答案
            # sub_kb_embedding = move_true_kb_to_front_inference(
            #     sub_kb_embedding,
            #     i,
            # )
            # if kb_adjs is None:
            #     kb_adj_i = None
            # elif isinstance(kb_adjs, list):
            #     kb_adj_i = kb_adjs[start_idx + i]
            # elif hasattr(kb_adjs, "dim") and kb_adjs.dim() == 3:
            #     kb_adj_i = kb_adjs[start_idx + i]
            # else:
            #     kb_adj_i = kb_adjs
                
            output_text = answer_question_deterministic(
                tokenizer,
                model,
                Q,
                kb=(key_embeddings, value_embeddings),
                kb_config=kb_config,
                kb_adj=kb_adj,
                # save_attention_weights=debug_flag,
                # attention_save_loc="./attn_weights_kblam/",
                # attention_file_base_name=f"kb-{idx}",
            )
            # print(f"[DEBUG] output_text={output_text}")
            if Q in output_text:
                output_text=output_text.split(Q)[1]
            # output_text=output_text.split(Q)[1]

            prof = kblam_profile_get()
            kblam_profile_reset()
            prefill_s = prof["prefill_s"]                # 模型 prefill
            decode_s = prof["decode_s"]
            decode_tokens = max(1, prof["decode_tokens"])
            tpot = decode_s / decode_tokens             # ms/token 可乘 1000
            ttft = retrieval_time + prefill_s

            TTFTs.append(ttft)
            TPOTs.append(tpot)

            model_output = strip_generation_prefix(output_text, model)
            if remove_sorry and "sorry" in model_output.lower():
                continue
            answers.append(format_output_for_synthetic(A))
            model_outputs.append(format_output_for_synthetic(model_output))

        all_model_outputs.extend(model_outputs)
        all_answers.extend(answers)

    end_time=time.time()
    print(f"============== 2wiki Performance ==================")
    print(f"query per second: {query_size/(end_time-start_time)}")

    print(f"Average TTFTs: {np.mean(TTFTs)}")
    print(f"Average TPOTs: {np.mean(TPOTs)}")

    return all_model_outputs, all_answers


def perform_eval_v2(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    kb_config: KBLaMConfig,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    multi_entites: int = -1,
    remove_sorry: bool = False,
    query_size: int = 250,
):
    if seed < 0:
        np.random.seed(None)
    else:
        np.random.seed(seed)

    if query_size <= kb_size:
        return perform_eval(
            model,
            tokenizer,
            kb_retriever,
            # encoder_model_spec,
            kb_config,
            eval_mode,
            kb_size,
            seed,
            multi_entites,
            remove_sorry,
            query_size,
        )    
    dataset = kb_retriever.dataset
    total_N = len(dataset)

    subset_idx=np.random.randint(0, total_N, query_size)
    subset=[dataset[idx] for idx in subset_idx]
    kb_embeddings=kb_retriever.get_key_embeddings(subset_idx)
    key_embeddings, value_embeddings=kb_embeddings

    # 统一的结果容器
    all_model_outputs, all_answers = [], []

    # 计算批次数
    num_batches = (query_size + kb_size - 1) // kb_size

    print(f"[INFO] Eval mode={eval_mode}, KB size={kb_size}, Query size={query_size}, total_batches={num_batches}")

    start_time=time.time()
    TTFTs=[]
    TPOTs=[]
    retrieval_time=0.003

    for batch_id in tqdm(range(num_batches)):
        # ---- 当前批的 query 索引范围 ----
        start_idx = batch_id * kb_size
        end_idx = min((batch_id + 1) * kb_size, query_size)
        current_query_num = end_idx - start_idx

        # ---- KB 子集：固定大小 ----
        sub_kb_embedding=(key_embeddings[start_idx:end_idx], value_embeddings[start_idx:end_idx])

        # ---- Query 子集 ----
        query_subset=subset[start_idx:end_idx]

        # ---- 处理本批次 ----
        model_outputs, answers = [], []
        for i in range(current_query_num):
            row = query_subset[i]
            idx = subset_idx[start_idx + i]
            if multi_entites == -1:
                Q, A = row["Q"], row["A"]
            else:
                kb_subset_idx = np.random.randint(0, total_N, multi_entites)
                Q, A = generate_multi_entity_qa(
                    [dataset[j]["name"] for j in kb_subset_idx],
                    [dataset[j]["description_type"] for j in kb_subset_idx],
                    [dataset[j]["description"] for j in kb_subset_idx],
                )
            if eval_mode != "kb":
                raise ValueError(f"eval_mode={eval_mode} is not supported for batched evaluation")

            # 可选，移动正确 key 到开头，用于验证模型是否能正确定位答案
            # sub_kb_embedding = move_true_kb_to_front_inference(
            #     sub_kb_embedding,
            #     i,
            # )
            output_text = answer_question_deterministic(
                tokenizer,
                model,
                Q,
                kb=sub_kb_embedding,
                kb_config=kb_config,
                # save_attention_weights=debug_flag,
                # attention_save_loc="./attn_weights_kblam/",
                # attention_file_base_name=f"kb-{idx}",
            )
            if Q in output_text:
                output_text = output_text.split(Q)[1]
                model_output = strip_generation_prefix(output_text, model)
            else:
                model_output=output_text
            # print(f"model split output: {model_output}")

            prof = kblam_profile_get()
            kblam_profile_reset()
            prefill_s = prof["prefill_s"]                # 模型 prefill
            decode_s = prof["decode_s"]
            decode_tokens = max(1, prof["decode_tokens"])
            # print(f"prefill_s: {prefill_s}, decode_s: {decode_s}, decode_tokens: {decode_tokens}")
            tpot = decode_s / decode_tokens             # ms/token 可乘 1000
            ttft = retrieval_time + prefill_s

            TTFTs.append(ttft)
            TPOTs.append(tpot)


            if remove_sorry and "sorry" in model_output.lower():
                continue

            # if multi_entites == -1:
            #     pattern = r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)'
            #     match = re.search(pattern, model_output)
            #     answers.append(row["description"])
            #     if match:
            #         model_output = match.group(1)
            # else:
            #     pattern = r"(?:is|are) (.*?)(?:\.|;)"
            #     matches = re.findall(pattern, model_output)
            #     model_output = "; ".join(matches)
            #     answers.append(";".join(re.findall(r"(?:is|are) (.*?);", answer)))
            answers.append(format_output_for_synthetic(A))
            model_outputs.append(format_output_for_synthetic(model_output))

        # ---- 聚合 ----
        all_model_outputs.extend(model_outputs)
        all_answers.extend(answers)


        # print(f"[INFO] Batch {batch_id+1}/{num_batches} done. Queries={current_query_num}")


    end_time=time.time()
    print(f"============== 2wiki Performance ==================")
    print(f"query per second: {query_size/(end_time-start_time)}")

    print(f"Average TTFTs: {np.mean(TTFTs)}")
    print(f"Average TPOTs: {np.mean(TPOTs)}")
    return all_model_outputs, all_answers

    # # ---- 统一计算评估指标 ----
    # results, results_dict = full_evaluation(all_model_outputs, all_answers)
    # print(f"[DONE] Evaluated {len(all_model_outputs)} queries with KB size={kb_size}")
    # print(results)
    # return results, results_dict


def perform_eval_refusal(
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    kb_retriever: KBRetriever,
    kb_config: Optional[KBLaMConfig] = None,
    eval_mode: str = "kb",
    kb_size: int = 250,
    seed: int = 1,
    outlier_ratio: float = 0.2,
    topk_size: int = -1,
    question_size: int = 100,
):
    instruction_prompts = (
        'Please answer questions based on the given text with format: "The {property} of {name} is {description}",'
        ' if relevant information cannot be found in the text, please respond "I am sorry I cannot find relevant information in the KB".'
    )
    zero_shot_prompt = """
    Please answer the question in a very compact manner with format: The {property} of {name} is {description}
    """

    np.random.seed(seed)
    kb_idx = np.random.randint(0, len(kb_retriever.dataset), kb_size)
    test_kb = [kb_retriever.dataset[idx] for idx in kb_idx]
    kb_embedding = ()
    key_str = [row["key_string"] for row in test_kb]
    value_str = [row["description"] for row in test_kb]
    prompt_strs = ""
    for k, v in zip(key_str, value_str):
        prompt_strs += f"{k} is {v}; "

    kb_embedding = kb_retriever.get_key_embeddings(kb_idx)

    model_outputs = []
    answers = []
    # answer_question
    outlier_idx = np.arange(len(kb_retriever.dataset))
    outlier_idx = outlier_idx[~np.isin(outlier_idx, kb_idx)]
    np.random.shuffle(outlier_idx)
    question_size = min(kb_size, question_size)
    outlier_idx = outlier_idx[: int(question_size * outlier_ratio)]
    test_kb = test_kb[: int(question_size * (1 - outlier_ratio))] + [
        kb_retriever.dataset[idx] for idx in outlier_idx
    ]
    change_point = int(question_size * (1 - outlier_ratio))
    for i, row in tqdm(enumerate(test_kb)):
        Q = row["Q"]
        if eval_mode == "kb":
            model_output = answer_question(
                tokenizer,
                model,
                Q,
                kb=kb_embedding,
                topk_size=topk_size,
                kb_config=kb_config,
            )
            if Q in model_output:
                model_output=model_output.split(Q)[1]

        elif eval_mode == "icl":
            model_output = answer_question(
                tokenizer,
                model,
                instruction_prompts + prompt_strs + Q,
                kb=None,
                kb_config=kb_config,
            )
            if Q in model_output:
                model_output=model_output.split(Q)[1]

        elif eval_mode == "zeroshot":
            model_output = answer_question(
                tokenizer,
                model,
                zero_shot_prompt + Q,
                kb=None,
                kb_config=kb_config,
            )
            if Q in model_output:
                model_output=model_output.split(Q)[1]
        model_outputs.append(model_output)
        if i < change_point:
            answers.append(row["description"])
        else:
            answers.append("Cannot find relevant information in the KB")
    true_label = [0] * change_point + [1] * int(question_size * outlier_ratio)
    prediction = [int("sorry" in model_output) for model_output in model_outputs]
    print(f"KB size: {kb_size}, mode: {eval_mode}, outlier ratio: {outlier_ratio}")
    results = ""
    for a, A in zip(model_outputs, answers):
        results += f"Model output: {a}\nTrue answer: {A}\n-------\n"
    return results, np.array([prediction, true_label])


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
    "--fancy_instruction",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use fancy instructions",
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
    default=100,
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
parent_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
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


# Create subparsers
subparsers = parser.add_subparsers(dest="command", required=True)

# Create the parser for the generation command
gen_parser = subparsers.add_parser(
    "generation", parents=[parent_parser], help="Evaluate generation"
)
gen_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
)
gen_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="generation_results",
    help="Name of the experiment configuration",
)
gen_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="Frequency of knowledge base token layers",
)
gen_parser.add_argument(
    "--multi_entites",
    type=int,
    default=-1,
    help="Number of entities to process (-1 for unlimited)",
)
gen_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use checkpoints trained without outliers",
)
gen_parser.add_argument(
    "--remove_sorry",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Filter out "sorry" answers from the output',
)
gen_parser.add_argument(
    "--topk_size", type=int, default=-1, help="Size of top-k selection (-1 for all)"
)


# Create the parser for the accuracy command
acc_parser = subparsers.add_parser(
    "accuracy", parents=[parent_parser], help="Evaluate accuracy"
)

acc_parser.add_argument(
    "--attn_save_dir", type=str, default="", help="Directory to save attention masks"
)
acc_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="accuracy_results",
    help="Name of the experiment configuration",
)
acc_parser.add_argument(
    "--fancy_question",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable fancy question format",
)
acc_parser.add_argument(
    "--log_save_dir", type=str, help="Directory to save accuracy results"
)
acc_parser.add_argument(
    "--test_batch_size", type=int, default=50, help="Batch size for testing"
)
acc_parser.add_argument(
    "--use_shift_match",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable shift matching",
)

# Create the parser for the accuracy eval
acc_results_parser = subparsers.add_parser(
    "acc_results", parents=[acc_parser], help="run accuracy eval", add_help=False
)


# Create the parser for the refusal command
ref_parser = subparsers.add_parser(
    "refusal", parents=[parent_parser], help="Evaluate refusal"
)
ref_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
)
ref_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="refusal_results",
    help="Name of the experiment configuration",
)
ref_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="Frequency of knowledge base token layers",
)
ref_parser.add_argument(
    "--multi_entites",
    type=int,
    default=-1,
    help="Number of entities to process (-1 for unlimited)",
)
ref_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use checkpoints trained without outliers",
)
ref_parser.add_argument(
    "--remove_sorry",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Filter out "sorry" answers from the output',
)
ref_parser.add_argument(
    "--topk_size", type=int, default=-1, help="Size of top-k selection (-1 for all)"
)

# Create the parser for the standard command
basic_parser = subparsers.add_parser(
    "standard", parents=[parent_parser], help="Evaluate basic performance"
)
basic_parser.add_argument(
    "--attn_summary_save_dir",
    type=str,
    default="",
    help="Directory to save attention masks",
)
basic_parser.add_argument(
    "--eval_mode",
    type=str,
    choices=["kb", "icl", "zeroshot"],
    default="kb",
    help="Evaluation mode: knowledge base, in-context learning, or zero-shot",
)
basic_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="basic_results",
    help="Name of the experiment configuration",
)
basic_parser.add_argument(
    "--exp_config_str", type=str, help="Experiment configuration string"
)
basic_parser.add_argument(
    "--kb_token_layer_frequency",
    type=int,
    default=None,
    help="Frequency of knowledge base token layers",
)
basic_parser.add_argument(
    "--no_outlier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use checkpoints trained without outliers",
)
basic_parser.add_argument(
    "--sample_size", default=5, type=int, help="Number of samples to process"
)
basic_parser.add_argument(
    "--subset_size", default=100, type=int, help="Size of the data subset to use"
)
basic_parser.add_argument(
    "--topk_size", type=int, default=-1, help="Size of top-k selection (-1 for all)"
)


def eval_main_process(
    dataset: list[dict],
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    encoder: KBEncoder,
    kb_config: KBLaMConfig,
    kb_retriever: KBRetriever,
    kb_scale_factor_range: list[float] | None = None,
    kb_scale_factor: float | None = None,
    dataset_type: str = "synthetic",
    seed: int = 42,
    kb_size: int = -1,
    query_size: int = -1,
    eval_mode: str = "kb",
):
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
        if dataset_type == "musique":
            model_outputs, answers = perform_eval_musique(
                model,
                tokenizer,
                kb_retriever,
                kb_config,
                eval_mode,
                seed=seed,
                kb_size=kb_size,
                query_size=query_size,
            )
        elif dataset_type=="2wiki":
           model_outputs, answers = perform_eval_2wiki(
                model,
                tokenizer,
                kb_retriever,
                kb_config,
                eval_mode,
                seed=seed,
                kb_size=kb_size,
                query_size=query_size,
            ) 
        else:
            model_outputs, answers = perform_eval_v2(
            # gen_results, score_results = perform_eval(
                model,
                tokenizer,
                kb_retriever,
                kb_config,
                eval_mode,
                seed=seed,
                kb_size=kb_size,
                query_size=query_size,
            )
        
        results_pair_list.append((model_outputs, answers))
    return results_pair_list, scale_factor_list



def eval_generate():
    """Evaluate generation using KB"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # adapter
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir


    eval_mode = args.eval_mode
    exp_config = args.exp_config_name
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_scale_factor_range = args.kb_scale_factor_range

    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    query_head_path = args.query_head_path

    # embeddings 
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path


    dataset_path=os.path.join(dataset_dir, test_dataset)
    # 判断数据集是json还是jsonl格式
    if dataset_path.endswith(".jsonl"):
        dataset=[json.loads(line.strip()) for line in open(dataset_path)]
    elif dataset_path.endswith(".json"):
        dataset=json.load(open(dataset_path))
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )
    kb_config.format_short = args.format_short
    kb_config.path_attn = args.path_attn

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    results_pair_list, scale_factor_list = eval_main_process(
        dataset,
        tokenizer,
        model,
        encoder,
        kb_config,
        kb_retriever,
        kb_scale_factor_range,
        kb_scale_factor,
        args.dataset_type,
        seed,
        kb_size,
        args.query_size,
    )

    for i in range(len(results_pair_list)):
        model_outputs, answers = results_pair_list[i]
        sf = scale_factor_list[i]
        
        gen_results, score_results = full_evaluation(model_outputs, answers)
        # mem_cost = torch.cuda.max_memory_reserved("cuda")
        # score_results["mem_cost"] = mem_cost
        # print(score_results)
        # print(gen_results)
        print(f"---- kb_scale_factor: {sf}, {score_results}")
        if args.save_dir is not None:
            (Path(args.save_dir) / exp_config).mkdir(exist_ok=True, parents=True)
            write_to_json(score_results, Path(args.save_dir) / f"{exp_config}-{sf}.json")
            text_file = open(os.path.join(args.save_dir, f"{exp_config}-{sf}.txt"), "w")
            text_file.write(gen_results)


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
    tokenizer = AutoTokenizer.from_pretrained(
        llm_base_dir, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = "^"

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
        from transformers import AutoModelForCausalLM
        from kblam.models.olmo3.kblam_olmo3_attention import replace_attention_with_kblam

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")

        model.set_attn_implementation("eager")
        replace_attention_with_kblam(model)
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
        encoder_name=encoder_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size
        * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
    )
    # print(f"encoder out_dim: {model.config.hidden_size
        # * (model.config.num_hidden_layers // kb_layer_frequency + 1)}")

    encoder.load_state_dict(torch.load(encoder_path))
    return tokenizer, encoder, model, kb_config


def eval_accuracy(
    tokenizer,
    kb_retriever,
    model,
    dataset,
    exp_config,
    fancy_question,
    kb_config,
    kb_size,
    llm_type,
    test_batch_size,
    save_dir,
    attn_save_dir,
):
    """Evaluate accuracy using KB"""

    if kb_size == len(dataset):
        dataset_subset_idx = range(len(dataset))
    elif kb_size > len(dataset):
        raise IndexError(
            f"The KB size {kb_size} is greater than the dataset size {len(dataset)}"
        )
    else:
        dataset_subset_idx = np.random.choice(len(dataset), kb_size, replace=False)

    dataset_subset = [dataset[i] for i in dataset_subset_idx]

    kb_embedding_real = kb_retriever.get_key_embeddings(dataset_subset_idx)

    format_func_map = {"llama3": format_Q_llama, "phi3": format_Q_phi3}

    if not fancy_question:
        input_strs_gen = (dataset_subset[i]["Q"] for i in range(test_batch_size))
    else:
        # input_strs_gen = (aug_row(dataset_subset[i]) for i in range(test_batch_size))
        print("ERROR: aug_row not implemented")
    input_strs = [format_func_map[llm_type](ex) for ex in input_strs_gen]

    tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
        "cuda"
    )
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb_embedding_real,
            max_new_tokens=60,
            tokenizer=tokenizer,
            output_attentions=True,
            save_attention_weights=True,
            kb_config=kb_config,
            attention_save_loc=attn_save_dir,
            attention_file_base_name=exp_config,
        )
        outputs = tokenizer.batch_decode(outputs.squeeze(), skip_special_tokens=False)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    with open(save_path / f"{exp_config}_acc.txt", "w+") as text_file:
        for output in outputs:
            output_string = output.strip("^")
            text_file.write(f"{str(output_string)}\n")

    accs = []
    with torch.autograd.no_grad():
        for idx in range(0, 32, kb_config.kb_layer_frequency):
            weight = np.load(os.path.join(attn_save_dir, f"{exp_config}_{idx}.npy"))
            weight = weight[..., :kb_size]
            label = np.arange(test_batch_size)
            weight = weight.reshape(test_batch_size, -1, kb_size)
            acc = (weight.sum(1).argmax(1) == label).mean()
            top_5_predictions = torch.topk(torch.from_numpy(weight.sum(1)), 5, dim=1)[1]
            top_5_acc = (top_5_predictions.numpy() == label[:, None]).any(1).mean()
            if idx == 15:
                print(f"ACC & TOP 5 ACC: {idx} {(acc, top_5_acc)}")
                print(f"min: {np.min(weight)}  max: {np.max(weight)}")
            accs.append(
                {
                    "idx": idx,
                    "acc": float(acc),
                    "top5acc": float(top_5_acc),
                }
            )

    np.save(
        save_path / f"{exp_config}_acc.npy",
        np.array([(a["acc"], a["top5acc"]) for a in accs]),
    )

    return accs


def eval_accuracy_cli():
    """Evaluate accuracy using KB"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_path = args.encoder_dir
    encoder_spec = args.encoder_spec
    exp_config = args.exp_config_name
    fancy_question = args.fancy_question
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = llm_type = args.llm_type
    model_path = args.model_dir
    test_batch_size = args.test_batch_size
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    query_head_path = args.query_head_path
    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )
    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    eval_accuracy(
        tokenizer,
        kb_retriever,
        model,
        dataset,
        exp_config,
        fancy_question,
        kb_config,
        kb_size,
        llm_type,
        test_batch_size,
        args.log_save_dir,
        args.attn_save_dir,
    )


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


def run_accuracy_evalution():
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_path = args.encoder_dir
    encoder_spec = args.encoder_spec
    exp_config = args.exp_config_name
    fancy_question = args.fancy_question
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    llm_base_dir = args.llm_base_dir
    llm_type = llm_type = args.llm_type
    model_path = args.model_dir
    test_dataset = args.test_dataset

    query_head_path = args.query_head_path
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))
    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    xs = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    accuracy_results = []
    for x in xs:
        print(f"kb_size {x}")

        accs = eval_accuracy(
            tokenizer,
            kb_retriever,
            model,
            dataset,
            exp_config,
            fancy_question,
            kb_config,
            x,
            llm_type,
            min(x, 200),
            args.log_save_dir,
            args.attn_save_dir,
        )
        shutil.rmtree(args.attn_save_dir)
        os.mkdir(args.attn_save_dir)
        accuracy_results.append({"kb_size": x, "accuracy_results": accs})
    write_to_json(
        accuracy_results, os.path.join(args.log_save_dir, "accuracy_results.json")
    )


def eval_refusal():
    """Evaluate refusal to answer questions for which the answer does not exist in the KB"""
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    eval_mode = args.eval_mode
    exp_config = args.exp_config_name
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path
    query_head_path = args.query_head_path

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    gen_results, refusal_results = perform_eval_refusal(
        model,
        tokenizer,
        kb_retriever,
        eval_mode=eval_mode,
        seed=seed,
        kb_size=kb_size,
        topk_size=args.topk_size,
        kb_config=kb_config,
    )

    np.save(os.path.join(args.save_dir, "OutLierTest" + exp_config), refusal_results)
    text_file = open(
        os.path.join(args.save_dir, "OutLierTest" + exp_config + ".txt"), "w"
    )
    text_file.write(gen_results)


def eval():
    """Evaluate the KB model"""
    args = parser.parse_args()
    attn_summary_save_dir = args.attn_summary_save_dir
    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    exp_config_str = args.exp_config_str
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    output_dir = args.save_dir
    sample_size = args.sample_size
    seed = args.seed
    subset_size = args.subset_size
    test_dataset = args.test_dataset
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path
    query_head_path = args.query_head_path
    sep_query_head = True
    actual_kb_token_layer_frequency = 3

    if kb_size == -1:
        kb_size = None

    # validation_part_start_idx = 120000 if 'gpt' in test_dataset else 0
    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    if sep_query_head:
        print("Having seperate query head for KB!")

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["ATTN_SAVE_DIR"] = output_dir
    os.environ["EVAL_MODE"] = "1"

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    for param in model.parameters():
        param.requires_grad = False

    # Set up the encoder
    encoder = KBEncoder(
        encoder_name=encoder_model_spec.upper(),
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // actual_kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=torch.device("cuda"),
    )
    encoder.load_state_dict(torch.load(encoder_path))

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )
    no_kb_predictions = []
    predictions = []
    answer = []

    for _ in range(sample_size):
        print("******")
        dataset_subset_idx = np.random.choice(len(dataset), subset_size, replace=False)
        dataset_subset = [dataset[i] for i in dataset_subset_idx]
        encoder.eval()
        with torch.autograd.no_grad():
            kb_embedding_real = kb_retriever.get_key_embeddings(dataset_subset_idx)
            kb_embedding_key, kb_embedding_val = kb_embedding_real
            kb_embedding_real = (kb_embedding_key, kb_embedding_val)

        format_func_map = {"llama3": format_Q_llama, "phi3": format_Q_phi3}

        input_strs = [
            format_func_map[llm_type](dataset_subset[i]["Q"])
            for i in range(subset_size)
        ]

        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
            "cuda"
        )
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        kb_embedding_real = (kb_embedding_real[0], kb_embedding_real[1])

        config_str = f"{exp_config_str}__kb_{subset_size}__seed_{seed}"
        with torch.autograd.no_grad():
            outputs_no_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=None,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=False,
                kb_config=kb_config,
            )

            outputs_true_kb = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=kb_embedding_real,
                max_new_tokens=40,
                tokenizer=tokenizer,
                output_attentions=True,
                save_attention_weights=True,
                attention_save_loc=output_dir,
                attention_file_base_name=config_str,
                kb_config=kb_config,
            )
        print("decoding")
        outputs_no_kb = tokenizer.batch_decode(outputs_no_kb, skip_special_tokens=False)

        outputs_true_kb = tokenizer.batch_decode(
            outputs_true_kb, skip_special_tokens=False
        )
        print("KB:")
        for i in range(subset_size):
            print(
                "{} : {}".format(
                    dataset_subset[i]["name"], dataset_subset[i]["description"]
                )
            )

        for m in model_prune_format_mapping:
            if isinstance(model, m):
                prune_str = model_prune_format_mapping[m]

        print("------------------")
        for i in range(subset_size):
            print("True KB", prune_str(outputs_true_kb[i]))
            print("True answer: ", dataset_subset[i]["A"])
            no_kb_predictions.append(
                prune_str(outputs_no_kb[i]).split(dataset_subset[i]["Q"])[1]
            )
            predictions.append(
                prune_str(outputs_true_kb[i]).split(dataset_subset[i]["Q"])[1]
            )
            answer.append(dataset_subset[i]["A"])
            print("--------------------")
        print("******")

    rogue_score = rouge.compute(predictions=predictions, references=answer)
    np.savez(
        os.path.join(attn_summary_save_dir, f"{config_str}_rouge.npy"), **rogue_score
    )

    rogue_score_no_kb = rouge.compute(predictions=no_kb_predictions, references=answer)
    np.savez(
        os.path.join(attn_summary_save_dir, f"{config_str}_rouge_no_kb.npy"),
        **rogue_score_no_kb,
    )

    # Start inspecting attention masks
    ranges = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 32)]

    save_dir = output_dir
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    accs, confidences = [], []
    for left, right in ranges:
        weights = []
        kb_size = subset_size
        for idx in range(32)[left:right]:
            if idx % 3 == 0:
                weight = np.load(os.path.join(save_dir, f"{config_str}_{idx}.npy"))
                weights.append(weight[..., :kb_size].reshape(kb_size, -1, kb_size))
        print(len(weights))
        weights = np.stack(weights)
        weights = weights.transpose(1, 0, 2, 3).reshape(kb_size, -1, kb_size)
        acc = (weights.sum(1).argmax(1) == np.arange(kb_size)).mean()
        top_5_predictions = torch.topk(torch.from_numpy(weights.sum(1)), 5, dim=1)[1]
        top_5_acc = (
            (top_5_predictions == torch.arange(kb_size)[:, None]).any(1).float().mean()
        )
        accs.append((acc, top_5_acc))
        confidence = softmax(weights.mean(1), -1).max()
        confidences.append(confidence)
    np.save(
        os.path.join(attn_summary_save_dir, f"{config_str}_acc.npy"), np.array(accs)
    )
    np.save(
        os.path.join(attn_summary_save_dir, f"{config_str}_conf.npy"),
        np.array(confidences),
    )


def main():
    args = parser.parse_args()
    global debug_flag
    if args.debug_flag:
        print("Debug mode enabled")
        debug_flag = True
    else:
        debug_flag = False

    print(args)
    if args.command == "generation":
        eval_generate()
    elif args.command == "accuracy":
        eval_accuracy_cli()
    elif args.command == "acc_results":
        run_accuracy_evalution()
    elif args.command == "refusal":
        eval_refusal()
    elif args.command == "standard":
        eval()
    else:
        raise ValueError(f"command {args.command} not recognised")


if __name__ == "__main__":
    main()
