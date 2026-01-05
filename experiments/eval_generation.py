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
from kblam.models.olmo3.kblam_olmo3_attention import load_kblam_olmo3_model, replace_attention_with_kblam
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
    model_prune_format_mapping,
    answer_question,
    answer_question_deterministic,
    softmax,
    format_output_for_synthetic,
)
from kblam.utils.train_utils import get_kb_embd
from kblam.kb_retriever import KBRetriever
from kblam.metrics_evaluator import full_evaluation
import time
from kblam.models.llama3_model import kblam_profile_get, kblam_profile_reset
logging.set_verbosity_warning()

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
        model = load_kblam_olmo3_model(
            base_model_dir=llm_base_dir,   # 原始 olmo3-7b
            checkpoint_dir=model_path,      # 这个 stage1_lr_..._step_5600 目录
            device="cuda",
        )
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
        out_dim=model.config.hidden_size
        * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
    )
    # print(f"encoder out_dim: {model.config.hidden_size
        # * (model.config.num_hidden_layers // kb_layer_frequency + 1)}")
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
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

def _perform_eval_batched(
    *,
    model,
    tokenizer,
    kb_retriever,
    kb_config,
    dataset,
    query_idx,
    kb_size,
    hop_num=None,            # None = single-hop, 2 = 2wiki
    use_kb_adj=False,
    filter_fn=None,          # dataset 级过滤（2wiki 用）
    remove_sorry=False,
    enable_retrieval=False,
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

    TTFTs, TPOTs = [], []
    retrieval_time = 0.003
    start_time = time.time()

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
                kb_keys, kb_vals = kb_retriever.get_kb_batch_by_hnsw(questions, topk=1, device=torch.device("cuda"), random_sample=kb_size-1)
            else:
                kb_keys, kb_vals = kb_retriever.get_key_embeddings(batch_idx)
        else:
            if enable_retrieval:
                rerank_policy=2
                print(f"[Retrieve] rerank_policy={rerank_policy}")
                questions = [row["Q"] for row in batch]
                kb_keys, kb_vals, kb_adj = kb_retriever.get_kb_adj_batch_by_hnsw(questions, ann_topk=10, rerank_topk=1, rerank_policy=rerank_policy, device=torch.device("cuda"), random_sample=kb_size-1, hop_num=2, true_indices=batch_idx)
            else:
                kb_keys, kb_vals, kb_adj = kb_retriever.get_embeddings_with_adj_2wiki(
                    batch_indices=batch_idx,
                    hop_num=2,
                )
            



        # ---------- per-query ----------
        for i, row in enumerate(batch):
            Q, A = row["Q"], row["A"]

            #判断kb_keys是否是列表
            if isinstance(kb_keys, list):
                target_kb_keys = kb_keys[i]
                target_kb_vals = kb_vals[i]
                target_kb_adj = kb_adj[i] if hop_num is not None else None
            else:
                target_kb_keys = kb_keys
                target_kb_vals = kb_vals
                target_kb_adj = kb_adj if hop_num is not None else None

            output = answer_question_deterministic(
                tokenizer,
                model,
                Q,
                kb=(target_kb_keys, target_kb_vals),
                kb_config=kb_config,
                kb_adj=target_kb_adj,
            )

            if Q in output:
                output = output.split(Q)[1]

            prof = kblam_profile_get()
            kblam_profile_reset()

            prefill_s = prof["prefill_s"]
            decode_s = prof["decode_s"]
            decode_tokens = max(1, prof["decode_tokens"])

            TTFTs.append(prefill_s)
            TPOTs.append(decode_s / decode_tokens)

            model_out = format_output_for_synthetic(output[2:])
            gt = format_output_for_synthetic(A)

            if remove_sorry and "sorry" in model_out.lower():
                continue

            all_model_outputs.append(model_out)
            all_answers.append(gt)

    end_time = time.time()
    if enable_retrieval:
        kb_retriever.print_metrics()
        retrieval_time = kb_retriever.get_avg_retrieval_time()
    print(f"QPS: {query_size / (end_time - start_time):.2f}")
    print(f"Avg TTFT: {np.mean(TTFTs)+retrieval_time:.4f}")
    print(f"Avg TPOT: {np.mean(TPOTs):.4f}")

    return all_model_outputs, all_answers


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
    enable_retrieval: bool = False,
):
    if query_size > len(dataset):
        query_size = len(dataset)
    if kb_size > query_size:
        kb_size = query_size

    np.random.seed(None if seed < 0 else seed)
    query_idx = np.random.randint(0, len(dataset), query_size)

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
        if dataset_type=="2wiki":
            # 2wiki， hotpot_2hop, musique_2hop等两跳数据集
            def _2hop_filter(row):
                ans = format_output_for_synthetic(row["A"])
                return ans == row["triple_lists"][1]["description"]
            filter_fn=_2hop_filter
            hop_num=2
            use_kb_adj=True
        else:
            # squad， Synthetic等单跳数据集
            filter_fn=None
            hop_num=None
            use_kb_adj=False

        model_outputs, answers = _perform_eval_batched(
            model=model,
            tokenizer=tokenizer,
            kb_retriever=kb_retriever,
            kb_config=kb_config,
            dataset=dataset,
            query_idx=query_idx,
            kb_size=kb_size,
            hop_num=hop_num,
            use_kb_adj=use_kb_adj,
            filter_fn=filter_fn,
            enable_retrieval=enable_retrieval,
        )
        
        results_pair_list.append((model_outputs, answers))
    return results_pair_list, scale_factor_list



def eval_generate(args, dataset, tokenizer, encoder, model, kb_config, kb_retriever):
    
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
            (Path(args.save_dir) / args.exp_config_name).mkdir(exist_ok=True, parents=True)
            write_to_json(score_results, Path(args.save_dir) / f"{args.exp_config_name}-{sf}.json")
            text_file = open(os.path.join(args.save_dir, f"{args.exp_config_name}-{sf}.txt"), "w")
            text_file.write(gen_results)

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
        )
    elif args.command == "debug":
        debug_measure_retrieval_accuracy(kb_retriever, dataset)
    else:
        raise ValueError(f"command {args.command} not recognised")


if __name__ == "__main__":
    main()
