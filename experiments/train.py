import argparse
import json
import logging
import os
import pathlib
import re
from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Optional, Union, Tuple
import glob
import traceback


import numpy as np
import torch
import transformers

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
import wandb

from accelerate import Accelerator
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.theme import Theme
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.olmo3.kblam_olmo3_attention import load_kblam_olmo3_model, replace_attention_with_kblam



from kblam.utils.data_utils import augment_row, generate_multi_entity_qa, get_i_dont_know_ans
from kblam.utils.train_utils import context_set_size_scheduler, get_kb_embd, setup_scheduler_and_optimizer
from kblam.kb_retriever import KBRetriever
# from eval import eval_main_process
from eval_generation import eval_main_process
from kblam.metrics_evaluator import simple_evaluation, full_evaluation
from kblam.utils.eval_utils import format_QA_llama, format_QA_phi3, format_QA_llama_short, format_QA_olmo3
import re
import shutil
import random
import gc
LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

debug_level=1


# setup logging
# Create a custom theme for Rich
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)

# Create a Rich console with the custom theme
console = Console(theme=custom_theme)

# Configure the root logger to WARNING
logging.basicConfig(
    level=logging.WARNING,  # Set the root logger to WARNING
    format=LOGFORMAT_RICH,
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--dataset_type",type=str,default="synthetic")
parser.add_argument("--N", type=int, default=120000, help="Size of training set, select the first N samples for training")
parser.add_argument("--B", type=int, default=10, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--sep_query_head", action=argparse.BooleanOptionalAction, help="Train a separate query head")
parser.add_argument("--use_oai_embd", action="store_true", help="Use OpenAI embedding")
parser.add_argument("--use_cached_embd", action="store_true", help="Choose to use pre-computed KV embeddings")
parser.add_argument("--total_steps", type=int, default=20000, help="Total steps")
parser.add_argument("--encoder_spec", type=str, default="OAI")
parser.add_argument("--key_embd_src", type=str, default="key", choices=["key", "answer", "questions", None], help="Source of key embedding")
parser.add_argument("--use_data_aug", action="store_true", help="Randomly pick templates for the question")
parser.add_argument("--use_lr_decay", action="store_true")
parser.add_argument("--train_data_path", type=str, default="synthetic_data")
parser.add_argument("--train_precomputed_embed_keys_path", type=str, default=None, help="The path of the precomputed embed keys for training")
parser.add_argument("--train_precomputed_embed_values_path", type=str, default=None, help="The path of the precomputed embed values for training")

parser.add_argument("--model_dir_to_resume", type=str, default=None, help="Checkpoint directory to resume training")
# parser.add_argument("--hf_model_spec", type=str, default="meta-llama/Llama-3.2-1B-Instruct", choices=["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Llama-3.2-1B-Instruct"])
parser.add_argument("--hf_model_spec", type=str, default="meta-llama/Llama-3.2-1B-Instruct")

parser.add_argument("--hf_token", type=str,default=None,help="Huggingface token")
parser.add_argument("--model_save_dir", type=str, default="output", help="Place to save the checkpoints")
parser.add_argument("--keep_old_checkpoints", action=argparse.BooleanOptionalAction, default=False, help="Remove old checkpoints")
parser.add_argument("--kb_size", type=int, default=None, help="The size of the KB set size")
parser.add_argument("--dynamic_kb_size", nargs=2, type=int, default=None, help="The size of the KB set size. Set a dynamic range for the kbsize specify min and max")
parser.add_argument("--duplicate_true_kb", action=argparse.BooleanOptionalAction, default=True, help="Duplicate true entity's KB token")
parser.add_argument("--length_invariance", action=argparse.BooleanOptionalAction, default=False, help="Scale the raw attention score")
parser.add_argument("--outlier_num", type=int, default=1, help="Introduce questions without correct KB entites")
parser.add_argument("--multi_entities", type=int, default=None, help="Introduce questions involving multiple entities")
parser.add_argument("--use_extended_qa", action="store_true", help="Introduce QA with extended open-ended parts")
parser.add_argument("--kb_token_layer_frequency", type=int, default=3, help="Introduce QA with extended open-ended parts")
parser.add_argument("--gradient_accm_step", type=int, default=20, help="Introduce QA with extended open-ended parts")
parser.add_argument("--verbose", action="store_true", help="Set logging to debug")
parser.add_argument("--log_to_file", action="store_true", help="Log to file as well as stdout")
parser.add_argument("--llm_type",type=str,default="llama3",choices=["llama3", "phi3", "olmo3"])
parser.add_argument("--max_seq_len",type=int,default=None)
parser.add_argument("--save_period", type=int, default=2000, help="Save every n steps")
parser.add_argument("--debug_level", type=int, default=0, help="Debug level")
parser.add_argument("--path_attn", action="store_true", default=False, help="Use path attention")


# fmt: on

# Test arguments
parser.add_argument("--test_data_path", type=str, default=None, help="The path of the test data")
parser.add_argument("--test_precomputed_embed_keys_path", type=str, default=None, help="The path of the precomputed embed keys for testing")
parser.add_argument("--test_precomputed_embed_values_path", type=str, default=None, help="The path of the precomputed embed values for testing")
parser.add_argument("--test_kb_size", type=int, default=None, help="The size of the KB set size for testing")
parser.add_argument("--test_query_size", type=int, default=None, help="The size of the query set size for testing")
parser.add_argument("--test_kb_scale_factor", type=float, default=None, help="The scale factor of the KB set size for testing")
parser.add_argument("--test_kb_scale_factor_range", nargs=2, type=float, default=None, help="The range of the scale factor of the KB set size for testing")
parser.add_argument("--eval_step", type=int, default=50, help="Evaluate every n steps")
parser.add_argument("--format_short", type=bool, default=False, help="Use short answer in prompt")



def create_custom_progress_bar(
    console: Console = None,  # type: ignore
    color: str = "cyan",
    show_time: bool = True,
    show_spinner: bool = True,
    spinner_style: str = "dots",
    disable=False,
) -> Progress:
    """
    Create a custom progress bar using Rich, optionally including loss reporting.

    :param description: Description of the task
    :param total: Total number of steps
    :param console: Rich Console object (if None, a new one will be created)
    :param color: Color of the progress bar
    :param show_time: Whether to show the time remaining
    :param show_spinner: Whether to show a spinner
    :param spinner_style: Style of the spinner (e.g., "dots", "dots12", "line", "arrow")
    :param show_loss: Whether to show loss information
    :return: A Rich Progress object and task ID
    """
    if console is None:
        console = Console()
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

    columns.extend(
        [
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
        ]
    )

    if show_time:
        columns.append(TimeRemainingColumn())

    progress = Progress(*columns, console=console, expand=True, disable=disable)
    return progress



# 构建训练标签：只计算模型对assistant回答部分的预测损失，而忽略对用户提问等其他内容的预测
def _create_labels_for_llama(input_ids: torch.Tensor, input_strs: List[str], tokenizer, attention_masks: torch.Tensor):
    # Not sure this is correct. This method simply masks the <|start_header_id|>user<|end_header_id|> then leaves the rest in the labels
    # Possibly what they want is to mask out the query. To do that swap the index from the tokenizer below from 1 to 2
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|start_header_id|>assistant<|end_header_id|>")["input_ids"][1]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 2)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels

def _create_labels_for_llama_enhanced(input_ids: torch.Tensor,
                                      input_strs: List[str],
                                      tokenizer,
                                      attention_masks: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:] = -100

    assistant_header = tokenizer("<|start_header_id|>assistant<|end_header_id|>",
                                 add_special_tokens=False).input_ids

    for b in range(input_ids.size(0)):
        seq = input_ids[b].tolist()
        # 1. 找到 assistant header 起始位置
        start_idx = None
        for i in range(len(seq) - len(assistant_header) + 1):
            if seq[i:i + len(assistant_header)] == assistant_header:
                start_idx = i
                break
        if start_idx is None:
            continue   # 没找到 header，整行保持 -100

        # 2. 解除 header + 答案的掩码（到末尾非 pad 为止）
        pad_id = tokenizer("<|eot_id|>", add_special_tokens=False).input_ids[0]
        # 最后一个非 pad token 位置
        last_real = (attention_masks[b] == 1).nonzero()[-1].item()

        # header → 答案
        labels[b, start_idx: last_real + 1] = input_ids[b, start_idx: last_real + 1]

        # 3. 紧跟其后的**第一个** pad 位置强制设为 eot_id
        if last_real + 1 < labels.size(1):
            labels[b, last_real + 1] = pad_id   # 仅此 1 个 token 非 -100

        # 4. 再往后所有 padding 仍保持 -100（模型无需关心）
    return labels

def _create_labels_for_phi3(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    # We just want to mask out the starting token.
    # The tokenized values are left padded so we want to know where our Q/A pairs start
    # Not 100% this is correct
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|user|>")["input_ids"][0]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 1)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels

def _create_labels_for_olmo3(
    input_ids: torch.Tensor,
    input_strs: list[str],
    tokenizer,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Only compute loss on assistant content for OLMo3.
    """
    labels = input_ids.clone()
    labels[:] = -100

    assistant_start = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids

    for b in range(input_ids.size(0)):
        seq = input_ids[b].tolist()

        # 1. 找到 assistant 起始位置
        start_idx = None
        for i in range(len(seq) - len(assistant_start)):
            if seq[i : i + len(assistant_start)] == assistant_start:
                start_idx = i + len(assistant_start)
                break

        if start_idx is None:
            continue  # 没有 assistant，整行忽略

        # 2. 从 assistant 内容开始算 loss
        labels[b, start_idx:] = input_ids[b, start_idx:]

        # 3. mask padding
        if tokenizer.pad_token_id is not None:
            labels[b, labels[b] == tokenizer.pad_token_id] = -100

        # 4. mask attention_mask == 0
        if attention_mask is not None:
            labels[b] = labels[b].masked_fill(attention_mask[b] == 0, -100)

    return labels

def create_labels_for_olmo3_adapter(
    input_ids,
    input_strs,        # get_batch 传的，但 OLMo3 不需要
    tokenizer,
    attention_masks,
):
    return _create_labels_for_olmo3(
        input_ids=input_ids,
        input_strs=input_strs,
        tokenizer=tokenizer,
        attention_mask=attention_masks,
    )

# 从数据集中构建训练批次：对应输入字符串的Token id，注意力掩码以及标签
def get_batch(
    qa_format_func: Callable[[str, str], str],
    label_func: Callable[[torch.Tensor, List, Callable, torch.Tensor], torch.Tensor],
    dataset: List[Dict],
    tokenizer,
    device: torch.device,
    B: int = 20,
    random_sample=True,
    use_data_aug=False,
    include_outlier=False,
    multi_entities=None,
    use_extended_qa=False,
    global_step: int = 0,
):
    labels = []
    if multi_entities is not None:
        assert not include_outlier

    # 数据采样
    if random_sample:
        if multi_entities is not None:
            batch_indices = np.random.choice(len(dataset), (B, multi_entities), replace=False)
        else:
            batch_indices = np.random.choice(len(dataset), B, replace=False)
    else:
        batch_indices = np.arange(B)
    # 构建问答对
    def get_question_and_answer(idx: int) -> tuple[str, str]:
        if use_extended_qa:
            Q, A = dataset[idx]["extended_Q"], dataset[idx]["extended_A"]

        elif multi_entities is not None:
            Q, A = generate_multi_entity_qa(
                [dataset[i]["name"] for i in idx],
                [dataset[i]["description_type"] for i in idx],
                [dataset[i]["description"] for i in idx],
            )
        else:
            Q = augment_row(dataset[idx]) if use_data_aug else dataset[idx]["Q"]
            A = get_i_dont_know_ans() if include_outlier else dataset[idx]["A"]
        return Q, A

    # 遍历采样索引，获取有效的问答对并应用格式化函数生成input string
    with torch.autograd.no_grad():
        input_strs = []
        real_batch_indices = []
        for idx in batch_indices:
            Q, A = get_question_and_answer(idx)
            
            # 消除A中的连续padding
            import re
            A = re.sub(r'(<\|eot_id\|>)+', '<|eot_id|>', A.strip())
            if A == '<|eot_id|>': A = 'N/A'   # 防止整段变空


            if Q is not None and A is not None:
                input_strs.append(qa_format_func(Q, A))
                real_batch_indices.append(idx)
            else:
                print("Q or Answer is none")
        batch_indices = real_batch_indices
        
        # 将input string转换为token IDs和注意力掩码 
        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(device)
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        # 生成训练标签
        labels = label_func(input_ids, input_strs, tokenizer, attention_masks)
    if include_outlier:
        # Generate a new set of indices, such that the KB does not contain the entity where the question comes from
        batch_indices = np.random.choice(len(dataset), B, replace=False)
    return input_ids, attention_masks, labels, batch_indices


# DATASET_SUPPORT
# 随机选择B篇文档，每篇文档随机抽取一个问答对，组成字符串输入，转换成TokenIDs和注意力掩码并返回
def get_batch_from_document(
    qa_format_func: Callable[[str, str], str],
    label_func: Callable[[torch.Tensor, List, Callable], torch.Tensor],
    dataset: List[Dict],
    tokenizer,
    device: torch.device,
    B: int = 20,
    random_sample=True,
    use_data_aug=False,
    include_outlier=False,
    multi_entities=None,
    use_extended_qa=False,
):
    labels = []
    if multi_entities is not None:
        assert not include_outlier

    # 数据采样
    if random_sample:
        if multi_entities is not None:
            batch_indices = np.random.choice(len(dataset), (B, multi_entities), replace=False)
        else:
            batch_indices = np.random.choice(len(dataset), B, replace=False)
    else:
        batch_indices = np.arange(B)
    
    # 构建问答对
    def get_question_and_answer(idx: Union[int, List[int]]) -> tuple[str | None, str | None]:
        if use_extended_qa:
            # # 扩展QA：基于triples生成更复杂的问答
            # if isinstance(idx, list):
            #     # 多个样本的情况
            #     samples = [dataset[i] for i in idx]
            #     all_triples = [triple for sample in samples for triple in sample["triples"]]
            #     if all_triples:
            #         triple = random.choice(all_triples)
            #         Q = f"Can you explain how {triple['name']} relates to {triple['description']}?"
            #         A = f"{triple['name']} {triple['description_type']} {triple['description']}."
            #         return Q, A
            #     return None, None
            # else:
            #     # 单个样本的情况
            #     sample = dataset[idx]
            #     if sample["triples"]:
            #         triple = random.choice(sample["triples"])
            #         Q = f"Can you explain the relationship between {triple['name']} and {triple['description']}?"
            #         A = f"{triple['name']} {triple['description_type']} {triple['description']}."
            #         return Q, A
            #     return None, None
            print("Extended QA is not supported yet.")
            return None, None
        elif multi_entities is not None:
            # # 优化多实体问题生成逻辑
            # samples = [dataset[i] for i in idx]
            # all_triples = []
            # for sample in samples:
            #     all_triples.extend(sample["triples"])
            
            # if not all_triples:
            #     return None, None
            
            # # 确保选择的三元组覆盖足够多的实体
            # selected_triples = []
            # selected_entities = set()
            
            # # 优先选择能形成关系链的三元组
            # for triple in all_triples:
            #     if len(selected_entities) >= multi_entities:
            #         break
            #     if triple["name"] not in selected_entities or triple["description"] not in selected_entities:
            #         selected_triples.append(triple)
            #         selected_entities.add(triple["name"])
            #         selected_entities.add(triple["description"])
            
            # # 如果实体不足，补充随机三元组
            # if len(selected_entities) < multi_entities and len(all_triples) > len(selected_triples):
            #     remaining = multi_entities - len(selected_entities)
            #     additional = random.sample([t for t in all_triples if t not in selected_triples], min(remaining, len(all_triples) - len(selected_triples)))
            #     selected_triples.extend(additional)
            
            # entities = [t["name"] for t in selected_triples]
            # desc_types = [t["description_type"] for t in selected_triples]
            # descriptions = [t["description"] for t in selected_triples]
            
            # return generate_multi_entity_qa(entities, desc_types, descriptions)
            print("Multi-entity QA is not supported yet.")
            return None, None
        else:
            if isinstance(idx, list):
                # 不应该发生，因为标准QA情况下idx是单个整数
                idx = idx[0]
                
            sample = dataset[idx]
            if not sample["QAs"]:
                return None, None
                
            # 随机选择一个问题-答案对
            qa = random.choice(sample["QAs"])
            Q = qa["question"]
            A = get_i_dont_know_ans() if include_outlier else qa["answer"]
            
            if use_data_aug:
                # 数据增强：基于triples生成不同的问题表述
                # if sample["triples"]:
                #     triple = random.choice(sample["triples"])
                #     if "what" in Q.lower():
                #         Q = f"Could you tell me about {triple['name']} and its relationship to {triple['description']}?"
                #     elif "when" in Q.lower():
                #         Q = f"At what point in time did {triple['name']} {triple['description_type']} {triple['description']}?"
                #     elif "where" in Q.lower():
                #         Q = f"In which location can we find {triple['name']} {triple['description_type']} {triple['description']}?"
                #     elif "who" in Q.lower():
                #         Q = f"Which person is associated with {triple['name']} {triple['description_type']} {triple['description']}?"
                
                # 暂时不开启数据增强功能
                pass
            return Q, A

    # 遍历采样索引，获取有效的问答对并应用格式化函数生成input string
    with torch.autograd.no_grad():
        input_strs = []
        real_batch_indices = []
        for idx in batch_indices:
            Q, A = get_question_and_answer(idx)
            if Q is not None and A is not None:
                input_strs.append(qa_format_func(Q, A))
                real_batch_indices.append(idx)
            else:
                print("Q or Answer is none")
        batch_indices = real_batch_indices
        # print(f"---- batch_indices: {batch_indices}, input_strs: {input_strs}")
        # 将input string转换为token IDs和注意力掩码 
        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(device)
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        # 生成训练标签
        labels = label_func(input_ids, input_strs, tokenizer)
    if include_outlier:
        # Generate a new set of indices, such that the KB does not contain the entity where the question comes from
        batch_indices = np.random.choice(len(dataset), B, replace=False)
    return input_ids, attention_masks, labels, batch_indices

def get_batch_musique(
    qa_format_func: Callable[[str, str], str],
    label_func: Callable[[torch.Tensor, List, Callable], torch.Tensor],
    dataset: List[Dict],
    tokenizer,
    device: torch.device,
    B: int = 20,
    random_sample=True,
    use_data_aug=False,
    include_outlier=False,
    multi_entities=None,
    use_extended_qa=False,
    global_step: int = 0,
):
    # 暂不开启随机采样
    # random_sample = False

    labels = []
    if random_sample:
        batch_indices = np.random.choice(len(dataset), B, replace=False)
    else:
        batch_indices = list(range(global_step * B, (global_step + 1) * B))
        # 确保索引不超出数据集范围
        batch_indices = [idx % len(dataset) for idx in batch_indices]


    def get_question_and_answer(idx) -> tuple[str | None, str | None]:
        sample = dataset[idx]
        Q=sample["question"]
        A=get_i_dont_know_ans() if include_outlier else sample["answer"]
        return Q, A

    with torch.autograd.no_grad():
        input_strs = []
        real_batch_indices = []
        for idx in batch_indices:
            Q, A = get_question_and_answer(idx)
            if Q is not None and A is not None:
                input_strs.append(qa_format_func(Q, A))
                real_batch_indices.append(idx)
            else:
                print("Q or Answer is none")
        batch_indices = real_batch_indices
        # print(f"---- real_batch_indices: {real_batch_indices}")
        # print(f"---- input_strs: {input_strs}")
        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(device)
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        labels = label_func(input_ids, input_strs, tokenizer)
    if include_outlier:
        # Generate a new set of indices, make sure each new index is different from the original index
        for i in range(len(batch_indices)):
            while True:
                new_idx = np.random.choice(len(dataset))
                if new_idx != batch_indices[i]:
                    batch_indices[i] = new_idx
                    break
    return input_ids, attention_masks, labels, batch_indices

def get_prefix_str(args):
    use_data_aug = args.use_data_aug
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size

    if dynamic_kb_size is not None:
        kb_size = "dynamic"  # Random size

    duplicate_true_kb = args.duplicate_true_kb
    length_invariance = args.length_invariance
    outlier_ratio = args.outlier_num
    use_outlier = outlier_ratio != -1
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    lr = args.lr

    prefix_string = f"stage1_lr_{lr}"
    if kb_token_layer_frequency is not None:
        prefix_string += f"KBTokenLayerFreq{kb_token_layer_frequency}"
    if use_extended_qa:
        prefix_string += "UseExtendedQA"
    if multi_entities is not None:
        prefix_string += f"MultiEntities{multi_entities}"
    if use_outlier:
        prefix_string += f"UseOutlier{outlier_ratio}"
    if length_invariance:
        prefix_string += "LengthInvariant"
    if not duplicate_true_kb:
        prefix_string += "NoDuplicate"
    if kb_size is not None:
        prefix_string += f"KBSize{kb_size}"
    if sep_query_head:
        prefix_string += "SepQueryHead"
    if use_data_aug:
        prefix_string += "UseDataAug"
    return prefix_string

# 加载缓存的KB embedding
def _load_cached_embeddings(encoder_model_spec: str, dataset_dir: str, key_embd_src: str):
    if encoder_model_spec == "OAI":
        encoder_model_spec_str = "oai"
    else:
        encoder_model_spec_str = encoder_model_spec
    key_embds = np.load(
        os.path.join(
            dataset_dir,
            f"train_datasets_{encoder_model_spec_str}_embd_{key_embd_src}.npy",
        )
    ).astype("float32")
    if key_embd_src == "answer":
        # If we are using the answer string as the key, we also use it as the value string
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"train_datasets_{encoder_model_spec_str}_embd_answer.npy",
            )
        ).astype("float32")
    else:
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"train_datasets_{encoder_model_spec_str}_embd_value.npy",
            )
        ).astype("float32")
    return key_embds, value_embds

def _load_cached_embeddings_v2(precomputed_embed_keys_path: str, precomputed_embed_values_path: str):
    key_embds = np.load(precomputed_embed_keys_path).astype("float32")
    value_embds = np.load(precomputed_embed_values_path).astype("float32")
    return key_embds, value_embds

def get_step_config(
    step: int,
    current_accum_step: int,
    total_accum_step: int,
    use_data_aug: bool,
    outlier_num: int,
    multi_entities: int | None,
    use_extended_qa: bool,
):
    """
    Our instruction tuning dataset is composed of different types of instructions.
    Strategies:
    Outlier QA takes the last `outlier_num` accum steps;
    Multiple entites QA (if included) takes 1/3 of the rest accum_steps;
    Extended QA (if included) takes 1/3 of the rest accum_steps;
    Standard QA takes the rest.
    """
    config = {
        "global_step": step*total_accum_step + current_accum_step,
    }
    config["use_data_aug"] = use_data_aug
    config["include_outlier"] = False
    config["multi_entities"] = None
    config["use_extended_qa"] = False
    include_outlier = current_accum_step >= total_accum_step - 1 - outlier_num
    # Decide to include outlier and has reached the time
    if include_outlier:
        config["include_outlier"] = True
        return config
    if current_accum_step % 3 == 0:
        # multi_entities could be None,
        # in which case we just use standard QA
        config["multi_entities"] = multi_entities
        return config
    if current_accum_step % 3 == 1:
        config["use_extended_qa"] = use_extended_qa
        return config
    return config


def _get_parameter_count(encoder):
    param_count = 0.0
    for p in encoder.parameters():
        if p.requires_grad:
            param_count += p.numel()
    return param_count


def _get_phi3_query_head_parameters(
    model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            # For phi3
            if "qkv_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight[: model.config.hidden_size, :])  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


def _get_llama3_query_head_parameters(
    model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:  # TODO: this is different for each model type
            # For llama3
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight)  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


class Trainer:
    def __init__(
        self,
        llm_model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
        kbretriever: KBRetriever,
        tokenizer: transformers.PreTrainedTokenizer,
        kb_token_layer_frequency: int,
        num_steps: int,
        lr: float,
        device: torch.device | None,
        use_lr_decay: bool,
        kb_size: int | List[int],
        llm_savename: str,
        output_dir: str,
        sep_query_head: bool = False,
        max_seq_len: int | None = None,
        dataset_format: str = "synthetic",
        test_dataset: list[dict] | None = None,
        precomputed_test_embed_keys_path: str | None = None,
        precomputed_test_embed_values_path: str | None = None,
        test_kb_size: int | None = None,
        test_query_size: int | None = None,
        test_kb_scale_factor: float | None = None,
        test_kb_scale_factor_range: tuple[float, float] | None = None,
        eval_step: int = 50,
        format_short: bool = False,
        keep_old_checkpoints: bool = False,
    ):
        self.accelerator = Accelerator()
        self.logger = logging.getLogger("training")
        self.tokenizer = tokenizer
        self.sep_query_head = sep_query_head
        self.kb_token_layer_frequency = kb_token_layer_frequency
        self.num_steps = num_steps
        self.lr = lr
        self.max_seq_len = max_seq_len

        self.model = llm_model
        self.model.gradient_checkpointing_enable()

        self.device = device if device is not None else self.accelerator.device
        self.kbretriever = kbretriever
        self.kb_size = kb_size
        self.use_lr_decay = use_lr_decay
        self.llm_savename = llm_savename
        self.output_path = pathlib.Path(output_dir)
        self.dataset_format = dataset_format
        self.keep_old_checkpoints = keep_old_checkpoints

        if isinstance(llm_model, KBLaMPhi3ForCausalLM):  # Phi3
            self._get_batch = partial(get_batch, format_QA_phi3, _create_labels_for_phi3)
            self._get_params = _get_phi3_query_head_parameters
        elif isinstance(llm_model, KblamLlamaForCausalLM):  # llama

            format_func = format_QA_llama if not format_short else format_QA_llama_short

            if dataset_format == "synthetic" or dataset_format == "2wiki" or dataset_format == "squad":
                self._get_batch = partial(get_batch, format_func, _create_labels_for_llama_enhanced)
            elif dataset_format == "multi_wiki_qa_train":
                self._get_batch = partial(get_batch_from_document, format_func, _create_labels_for_llama_enhanced)
            elif dataset_format == "musique":
                self._get_batch = partial(get_batch_musique, format_func, _create_labels_for_llama_enhanced)
            else:
                raise ValueError(f"{dataset_format} not recognised")
            self._get_params = _get_llama3_query_head_parameters
        elif getattr(llm_model.config, "model_type", None) == "olmo3":
            self._get_batch = partial(
                get_batch,
                lambda Q, A: format_QA_olmo3(Q, A, self.tokenizer),
                create_labels_for_olmo3_adapter,
            )
            self._get_params = lambda *args, **kwargs: []
        else:
            raise ValueError(f"{llm_model} not recognised")

        self.scheduler, self.optim = self.setup_scheduler_and_optim()

        self.model, self.optim, self._get_batch, self.kbretriever.encoder = self.accelerator.prepare(
            self.model, self.optim, self._get_batch, self.kbretriever.encoder
        )


        # ==== 测试集 ====
        self.test_dataset = test_dataset
        self.precomputed_test_embed_keys_path = precomputed_test_embed_keys_path
        self.precomputed_test_embed_values_path = precomputed_test_embed_values_path
        self.test_kb_size = test_kb_size
        self.test_query_size = test_query_size
        self.test_kb_scale_factor = test_kb_scale_factor
        self.test_kb_scale_factor_range = test_kb_scale_factor_range
        self.eval_step = eval_step

    def setup_scheduler_and_optim(self):
        if self.sep_query_head:
            self.logger.info("Query head being fine tuned!")
            llm_q_params = self._get_params(self.model, self.sep_query_head, self.kb_token_layer_frequency)
            scheduler, optim = setup_scheduler_and_optimizer(
                chain(self.kbretriever.encoder.parameters(), llm_q_params),
                self.lr,
                self.num_steps,
            )
            self.logger.info("Optimizer recreated")
        else:
            scheduler, optim = setup_scheduler_and_optimizer(
                self.kbretriever.encoder.parameters(), self.lr, self.num_steps
            )
            self.logger.info("Optimizer recreated")
        
        # ==== 验证 V-Adapter 是否生效 ====
        # # 1) 列出 encoder 里含有 value 映射的参数（名字可能叫 projector_v / value_proj 等）
        # for n, p in self.kbretriever.encoder.named_parameters():
        #     if "projector_v" in n or "value" in n:
        #         print("[V-ADAPTER PARAM]", n, p.shape, "requires_grad=", p.requires_grad)
        # # 打印所有参数
        # print("All Parameters:")
        # for n, p in self.kbretriever.encoder.named_parameters():
        #     print(n, p.shape, "requires_grad=", p.requires_grad)

        # # 2) 列出被 optimizer 管理的第一组参数名字，确认包含上面这些 name
        # pg0 = list(scheduler.optimizer.param_groups[0]["params"])
        # name_of = {id(p): n for n,p in self.kbretriever.encoder.named_parameters()}
        # hit = [name_of.get(id(p)) for p in pg0 if id(p) in name_of]
        # print("[IN OPTIMIZER GROUP0 V-params]:", hit)   
        return scheduler, optim


    def evaluate(
        self,
        seed,
        train_config: KBLaMConfig = None,
    ):
        test_kb_retriever = KBRetriever(
            self.kbretriever.encoder,
            self.test_dataset,
            precomputed_embed_keys_path=self.precomputed_test_embed_keys_path,
            precomputed_embed_values_path=self.precomputed_test_embed_values_path,
        )

        test_kb_config = KBLaMConfig(
            sep_query_head=True,
            kb_layer_frequency=self.kb_token_layer_frequency,
            path_attn=train_config.path_attn if train_config is not None else False,
        )
        
        results_pair_list, scale_factor_list = eval_main_process(
            self.test_dataset,
            self.tokenizer,
            self.model,
            self.kbretriever.encoder,
            test_kb_config,
            test_kb_retriever,
            kb_scale_factor_range=self.test_kb_scale_factor_range,
            kb_scale_factor=self.test_kb_scale_factor,
            dataset_type=self.dataset_format,
            seed=seed,
            kb_size=self.test_kb_size,
            query_size=self.test_query_size,
        )

        for (results_pair, scale_factor) in zip(results_pair_list, scale_factor_list):
            model_outputs, answers = results_pair
            simple_score_dict=simple_evaluation(model_outputs, answers)
            self.logger.info(f"------- Scale factor: {scale_factor}, Simple scores: {simple_score_dict}")
            # 输出前5个样本的结果
            for idx in range(5):
                self.logger.info(f"Model Output: {model_outputs[idx]}\nTrue Answer: {answers[idx]}")
            
            # 输出5个随机样本的结果
            random_idxs = np.random.choice(len(model_outputs), 5, replace=False)
            for idx in random_idxs:
                self.logger.info(f"Model Output: {model_outputs[idx]}\nTrue Answer: {answers[idx]}")



    def safe_evaluate_wrapper(self, seed: int = 1, delay_cleanup: bool = False, train_config: KBLaMConfig = None):
        self.logger.info("===== [SAFE EVALUATION START] =====")

        try:
            # Step 1️⃣ 同步与清理：确保所有GPU空闲
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

            # Step 2️⃣ 保存训练状态
            was_training_encoder = self.kbretriever.encoder.training
            model_grad_state = torch.is_grad_enabled()

            # Step 3️⃣ 禁用梯度与 checkpoint
            torch.set_grad_enabled(False)
            self.kbretriever.encoder.eval()
            torch.cuda.synchronize()

            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()

            # Step 4️⃣ 执行评估（复用已有 evaluate 逻辑）
            self.logger.info("Running evaluation under no-grad mode...")
            self.evaluate(seed=seed, train_config=train_config)

        except Exception as e:
            self.logger.error(f"[SAFE_EVAL ERROR] {type(e).__name__}: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            # Step 5️⃣ 恢复训练状态
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            torch.set_grad_enabled(model_grad_state)
            if was_training_encoder:
                self.kbretriever.encoder.train()

            # Step 6️⃣ 释放显存
            if not delay_cleanup:
                torch.cuda.empty_cache()
                gc.collect()
            

    def _debug_module_modes(self, tag: str = ""):
    # 汇总部分子模块状态（只打印前几个，避免刷屏）
        drop_modes, ln_modes = [], []
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Dropout) and len(drop_modes) < 5:
                drop_modes.append((name, m.training))
            if isinstance(m, torch.nn.LayerNorm) and len(ln_modes) < 5:
                ln_modes.append((name, m.training))
        self.logger.info(f"[ModeCheck{(':'+tag) if tag else ''}] "
                        f"model.training={self.model.training} "
                        f"encoder.training={self.kbretriever.encoder.training} "
                        f"dropout={drop_modes} "
                        f"layernorm={ln_modes}")

    def train(
        self,
        training_set: List[Dict],
        batch_size,
        grad_accum_steps: int,
        outlier_num: int,
        use_data_aug: bool = False,
        multi_entities: bool = False,
        use_extended_qa: bool = False,
        save_period: int = 2000,
        resumed_step: int = 0,
        kb_config: KBLaMConfig = None,
    ):
        
        # 初始化损失函数，每个GPU的梯度累积步数有有效批次大小
        train_losses = []
        start_step = resumed_step

        loss_fct = CrossEntropyLoss(reduction="none")

        # Calculate accumulation steps per GPU
        num_processes = self.accelerator.num_processes
        accum_steps_per_gpu = max(1, grad_accum_steps // num_processes)
        effective_batch_size = batch_size * grad_accum_steps

        if self.accelerator.is_main_process:
            self.logger.info(f"Training with {num_processes} GPUs")
            self.logger.info(f"Total accumulation steps: {grad_accum_steps}, Steps per GPU: {accum_steps_per_gpu}")
            self.logger.info(f"Batch size: {batch_size}")
            self.logger.info(f"Effective batch size: {effective_batch_size}")

        with create_custom_progress_bar(console=console, disable=not self.accelerator.is_main_process) as pbar:
            task = pbar.add_task("Training", total=self.num_steps, loss=100)
            # 训练循环
            for step in range(start_step, self.num_steps, 1):
                
                # 每次迭代开始时清空梯度
                self.optim.zero_grad()
                losses = []

                # 确定当前processor（GPU）需要处理的梯度累积步骤范围
                process_rank = self.accelerator.process_index
                start_accum_step = process_rank * accum_steps_per_gpu
                end_accum_step = min(start_accum_step + accum_steps_per_gpu, grad_accum_steps)

                # 梯度累积循环
                for a_step in range(start_accum_step, end_accum_step):
                    # 获取当前批次数据：输入IDs，注意力掩码，标签，批次索引
                    step_config = get_step_config(
                        step,
                        a_step,
                        grad_accum_steps,
                        use_data_aug,
                        outlier_num,
                        multi_entities,
                        use_extended_qa,
                    )
                    if debug_level > 0:
                        print(f"step_config: {step_config}")
                    input_ids, attention_masks, labels, batch_indices = self._get_batch(
                        training_set,
                        self.tokenizer,
                        self.device,
                        B=batch_size,
                        random_sample=True,
                        **step_config,
                    )
                    if a_step == 0 and step % 10 == 0:
                        self.logger.info(f"INPUT IDs SHAPE: {input_ids.shape}")

                    # 截断输入
                    if self.max_seq_len is not None:
                        input_ids = input_ids[:, : self.max_seq_len]
                        attention_masks = attention_masks[:, : self.max_seq_len]
                        labels = labels[:, : self.max_seq_len]
                        if a_step == 0 and step % 10 == 0:
                            self.logger.info(f"TRUNCATED INPUT IDs SHAPE: {input_ids.shape}")
                    
                    kb_adj=None
                    if self.dataset_format == "synthetic" or self.dataset_format == "squad":
                        kb_embedding = self.kbretriever.get_key_embeddings(
                            batch_indices, len(input_ids), step, self.kb_size
                        )
                    elif self.dataset_format == "2wiki":
                        key_embd, value_embd, kb_adj = self.kbretriever.get_embeddings_with_adj_2wiki(
                            batch_indices=batch_indices,
                            step=step,
                            kb_size=self.kb_size,
                            hop_num=2,
                        )
                        kb_embedding=(key_embd, value_embd)
                    else:
                        raise ValueError(f"Unknown data set format: {self.dataset_format}")
                    
                    
                    if debug_level > 1:
                        out = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_masks,
                            kb_kvs=kb_embedding,
                            output_attentions=True,
                            kb_config=kb_config,
                            save_attention_weights=True,
                            attention_save_loc="./attn_weights/",
                            attention_file_base_name=f"debug_train_kbscale{kb_config.kb_scale_factor}",
                        )
                    else:
                        out = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_masks,
                            kb_kvs=kb_embedding,
                            kb_adj=kb_adj,
                            output_attentions=True,
                            kb_config=kb_config,
                        )
                    logits = out["logits"]
                    # 打印部分结果
                    if (a_step == 0 and step % 10 == 0) or debug_level > 1:
                        batch_index = 0  # Which example in the batch to select

                        # max_logits = logits.argmax(axis=2)
                        # decoded_pred = self.tokenizer.decode(max_logits[batch_index, :-1])
                        # sel_labels = labels[batch_index, :]
                        # sel_labels = sel_labels[sel_labels >= 0]  # Remove padding token -100
                        # decoded_gt = self.tokenizer.decode(sel_labels)

                        # FIX
                        max_logits = logits.argmax(dim=2)
                        valid_pos = labels[batch_index] != -100
                        pred_ids = max_logits[batch_index][valid_pos]
                        decoded_pred = self.tokenizer.decode(
                            pred_ids,
                            skip_special_tokens=False,
                        )
                        sel_labels = labels[batch_index]
                        sel_labels = sel_labels[sel_labels != -100]

                        decoded_gt = self.tokenizer.decode(
                            sel_labels,
                            skip_special_tokens=False,
                        )
                        self.logger.info(f"KB SHAPE: {kb_embedding[0].shape}")
                        self.logger.info(f"GT: {decoded_gt}")
                        self.logger.info(f"PRED: {decoded_pred}")
                        wandb.log({"kbsize": kb_embedding[0].shape[1]})
                    # 计算交叉熵损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()
                    # FIX
                    valid_mask = shift_labels != -100
                    weights = valid_mask.sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()

                    # Flatten the tokens
                    model_config = (
                        self.model.config
                        if not isinstance(self.model, DistributedDataParallel)
                        else self.model.module.config
                    )
                    shift_logits = shift_logits.view(-1, model_config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    weights = weights.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = (
                        loss_fct(shift_logits, shift_labels) * weights.max() / weights
                    ).mean()  # Make sure each sample is equally weighted
                    # 执行反向传播并将损失值保存
                    self.accelerator.backward(loss)
                    losses.append(loss.item())

                # 达到累积梯度步数时更新模型参数
                # 如果启用了学习率衰减，则同时更新学习率
                self.optim.step()
                if self.use_lr_decay:
                    self.scheduler.step()

                # 收集所有GPU上的损失值并计算平均损失
                if losses:  # Only if this GPU processed any batches
                    local_loss = torch.tensor(np.mean(losses), device=self.device)
                else:
                    local_loss = torch.tensor(0.0, device=self.device)

                # Gather losses from all processes
                all_losses = self.accelerator.gather(local_loss)
                valid_losses = all_losses[all_losses > 0]  # Filter out zeros from GPUs that didn't process batches
                avg_loss = valid_losses.mean().item() if len(valid_losses) > 0 else 0.0

                # Only log from main process
                if self.accelerator.is_main_process:
                    self.logger.info(f"step: {step} , loss: {avg_loss}")
                    num_candidates=0
                    wandb.log({'train_loss': np.mean(losses)})
                    train_losses.append(avg_loss)
                    pbar.update(task, advance=1, loss=avg_loss)

                # 保存模型参数
                # 如果是最后一步
                if ((step == self.num_steps-1) or ((step % save_period) == 0 and (step != start_step))) and (debug_level < 2) :
                    try:
                        # Log memory usage before synchronization
                        self.logger.info(
                            f"Is main process: {self.accelerator.is_main_process}, GPU memory before save: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB"
                        )

                        # Try to free up memory
                        torch.cuda.empty_cache()

                        # Synchronize before saving
                        self.accelerator.wait_for_everyone()

                        if self.accelerator.is_main_process:
                            
                            self.logger.info("Saving checkpoint...")
                            self.logger.info("Making dirs...")
                            # Save model - using proper directory creation
                            model_ckpt_name = self.output_path / f"{self.llm_savename}_step_{step}"
                            model_ckpt_name.mkdir(parents=True, exist_ok=True)

                            # Also create encoder directory
                            encoder_dir = self.output_path / f"{self.llm_savename}_step_{step}_encoder"
                            encoder_dir.mkdir(parents=True, exist_ok=True)

                            self.logger.info("Saving model...")
                            # Unwrap and save model
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            unwrapped_model.save_pretrained(
                                model_ckpt_name,
                                is_main_process=self.accelerator.is_main_process,
                                save_function=self.accelerator.save,
                            )

                            self.logger.info("Saving encoder...")
                            # Save encoder and config from main process
                            encoder_ckpt_name = encoder_dir / "encoder.pt"
                            torch.save(self.kbretriever.encoder.state_dict(), encoder_ckpt_name)

                            self.logger.info("Saving config...")
                            # Explicitly save config as JSON
                            config_path = model_ckpt_name / "kb_config_explicit.json"
                            with open(config_path, 'w') as f:
                                f.write(kb_config.to_json_string())

                            # 删除旧的checkpoint
                            if not self.keep_old_checkpoints:
                                self.logger.info("Removing old checkpoints...")
                                # 获取所有checkpoint目录
                                for item in self.output_path.iterdir():
                                    if item.is_dir():
                                        # 匹配checkpoint目录名
                                        match = re.match(rf"{re.escape(self.llm_savename)}_step_(\d+)", item.name)
                                        if match:
                                            old_step = int(match.group(1))
                                            if old_step < step:
                                                # 删除旧的模型checkpoint
                                                try:
                                                    shutil.rmtree(item)
                                                    self.logger.info(f"Removed old model checkpoint: {item}")
                                                except FileNotFoundError:
                                                    # 目录可能已被其他进程删除
                                                    self.logger.info(f"Old model checkpoint already removed: {item}")
                                        
                                        # 匹配encoder目录名
                                        match_encoder = re.match(rf"{re.escape(self.llm_savename)}_step_(\d+)_encoder", item.name)
                                        if match_encoder:
                                            old_step = int(match_encoder.group(1))
                                            if old_step < step:
                                                # 删除旧的encoder checkpoint
                                                try:
                                                    shutil.rmtree(item)
                                                    self.logger.info(f"Removed old encoder checkpoint: {item}")
                                                except FileNotFoundError:
                                                    # 目录可能已被其他进程删除
                                                    self.logger.info(f"Old encoder checkpoint already removed: {item}")
                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint: {e}")
                        self.logger.error(f"Error details: {str(e)}")
                        raise e

                
                

                # 运行模型验证
                if ((step == self.num_steps-1) or ((step % self.eval_step) == 0 and (step != start_step))) and (self.test_dataset is not None) :                    
                    self.safe_evaluate_wrapper(seed=1, train_config=kb_config)

                
                
                

    
def main():

    os.environ["NCCL_TIMEOUT"] = "1200000"
    logger = logging.getLogger("training")

    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    global debug_level
    debug_level = args.debug_level

    print(vars(args))
    dataset_type = args.dataset_type
    seed = args.seed
    N = args.N
    B = args.B

    total_steps = args.total_steps
    encoder_spec = args.encoder_spec
    key_embd_src = args.key_embd_src
    use_data_aug = args.use_data_aug
    use_lr_decay = args.use_lr_decay
    use_cached_embd = args.use_cached_embd
    model_dir_to_resume = args.model_dir_to_resume
    model_save_dir = args.model_save_dir
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size
    max_seq_len = args.max_seq_len

    if kb_size is not None and dynamic_kb_size is not None:
        raise ValueError("Can't specify kb_size and dynamic_kb_size. Use only one")

    kb_size = kb_size if kb_size is not None else dynamic_kb_size

    gradient_accm_step = args.gradient_accm_step

    length_invariance = args.length_invariance
    outlier_num = args.outlier_num
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    llm_type = args.llm_type
    hf_model_spec = args.hf_model_spec
    hf_token = args.hf_token
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")

    print(f"[DEBUG]: hf_token: {hf_token}")

    # if seed is None:
    #     seed = time.time()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if Accelerator().is_main_process:
        wandb.init(
            # set the wandb project where this run will be logged
            project="kb-llm",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                'sep_query_head': sep_query_head,
                'kb_size': kb_size,
                'length_invariance': length_invariance,
                'dataset': dataset_type,
                'outlier_num': outlier_num,
                'multi_entities': multi_entities,
                'use_extended_qa': use_extended_qa,
                'kb_token_layer_frequency': kb_token_layer_frequency,
                'gradient_accm_step': gradient_accm_step,
                "encoder_spec": encoder_spec,
                "max_seq_len": max_seq_len,
            },
        )

    # Try to free up memory
    torch.cuda.empty_cache()

    if args.log_to_file:
        formatter = logging.Formatter(LOGFORMAT)
        f_handler = logging.FileHandler(model_save_dir / "log.txt")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    logger.info(f"Running on {device}")  # pyright: ignore[reportPossiblyUnboundVariable]

    logger.info("Started training")
    logger.info(f"Saving to  {model_save_dir}")
    if sep_query_head:
        os.environ["SEP_QUERY_HEAD"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    if length_invariance:
        os.environ["LENGTH_INVARIANCE"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    os.environ["SCALE_FACTOR"] = ""

    if use_cached_embd:
        key_embds, value_embds = _load_cached_embeddings_v2(args.train_precomputed_embed_keys_path, args.train_precomputed_embed_values_path)




    prefix_string = get_prefix_str(args)
    logger.info(f"Experiment prefix {get_prefix_str(args)}")

    # 判断数据集是json还是jsonl格式
    if args.train_data_path.endswith(".jsonl"):
        dataset=[json.loads(line.strip()) for line in open(args.train_data_path)]
    elif args.train_data_path.endswith(".json"):
        dataset=json.load(open(args.train_data_path))
    else:
        raise ValueError(f"Unknown dataset format: {args.train_data_path}")

#     if use_extended_qa:
#         dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}_augmented.json")))
#     else:
# # DATASET_SUPPORT
#         if dataset_name == "multi_wiki_qa_train":
#             dataset_path=os.path.join(dataset_dir, f"{dataset_name}.json")
#             with open(dataset_path, "r", encoding="utf-8") as f:
#                 dataset = [json.loads(line.strip()) for line in f]
#         elif "musique" in dataset_name:
#             # search for dataset file: end with "json", include "train"
#             dataset_path = glob.glob(os.path.join(dataset_dir, "*train*.json"))[0]
#             print(f"[INFO]: Using dataset file {dataset_path}")
#             with open(dataset_path, "r", encoding="utf-8") as f:
#                 dataset = json.load(f)
#         else:
#             dataset = json.load(open(os.path.join(dataset_dir, f"train_datasets.json")))

    N = min(N, len(dataset))
    training_set = dataset[:N]
    print(f"[INFO]: Loaded {N} samples for training")

    if args.test_data_path is not None:
        # 判断数据集是json还是jsonl格式
        if args.test_data_path.endswith(".jsonl"):
            test_dataset=[json.loads(line.strip()) for line in open(args.test_data_path)]
        elif args.test_data_path.endswith(".json"):
            test_dataset=json.load(open(args.test_data_path))
        else:
            raise ValueError(f"Unknown dataset format: {args.test_data_path}")
        print(f"[INFO]: Loaded {len(test_dataset)} samples for validation")
    else:
        test_dataset = None

    # Set up the LLM
    original_model_spec = hf_model_spec              # base olmo3-7b
    resume_ckpt_dir = model_dir_to_resume             # None or stage1_lr_..._step_xxxx

    if resume_ckpt_dir:
        resumed_step = int(resume_ckpt_dir.split("_")[-1])
        print(f"[INFO]: Resuming from {resume_ckpt_dir}, step: {resumed_step}")
    else:
        resumed_step = 0

    if model_dir_to_resume:
        print(f"[INFO]: Resuming from {model_dir_to_resume}, step: {resumed_step}")

    tokenizer = AutoTokenizer.from_pretrained(
        original_model_spec,
        trust_remote_code=True,
        token=hf_token if args.llm_type == "llama3" else None,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            resume_ckpt_dir if resume_ckpt_dir else original_model_spec,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    elif args.llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            resume_ckpt_dir if resume_ckpt_dir else original_model_spec,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    elif args.llm_type == "olmo3":
        model = load_kblam_olmo3_model(
            base_model_dir=original_model_spec,      # 永远是 base olmo3
            checkpoint_dir=resume_ckpt_dir,          # 只有 resume 时才非 None
            device="cuda",
        )

    else:
        raise ValueError(f"LLM type {args.llm_type} not recognised")


    print(
        f"[INFO]: Initialised model "
        f"{resume_ckpt_dir if resume_ckpt_dir else original_model_spec}"
    )

    logger.info(model.config)  # type: ignore

    model.eval()  # type: ignore
    # freeze model
    for _, param in model.named_parameters():  # type: ignore
        param.requires_grad = False

    # Set up the encoder
    encoder = KBEncoder(
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=device,
    )

    if model_dir_to_resume:
        encoder_path=os.path.join(model_dir_to_resume+"_encoder", "encoder.pt")
        if not os.path.exists(encoder_path):
            raise ValueError(f"[ERROR]: Encoder path {encoder_path} does not exist")
        encoder.load_state_dict(torch.load(encoder_path))

        kb_config_file=os.path.join(model_dir_to_resume, "kb_config_explicit.json")
        if not os.path.exists(kb_config_file):
            raise ValueError(f"[ERROR]: KB config file {kb_config_file} does not exist")
        kb_config = KBLaMConfig.from_pretrained(kb_config_file)
    else:
        kb_config = KBLaMConfig(
            sep_query_head=sep_query_head,
            kb_layer_frequency=kb_token_layer_frequency,
            path_attn=args.path_attn,
        )
    
    print(f"[INFO]: KBLaM config: {kb_config}")

    encoder.train()

    print(f"[INFO]: Initialised embedding encoder {encoder_spec}")

    kbretriever = KBRetriever(
        encoder,
        training_set,
        key_embds=key_embds,  # type: ignore
        value_embds=value_embds,  # type: ignore
    )

    logger.info("Model ready")

    print(f"[INFO]: Retriever ready")

    # Get the training started
    llm_ckpt_name = f"{prefix_string}KeyFrom{key_embd_src}_{encoder_spec}_{dataset_type}_{llm_type}"

    trainer = Trainer(
        model,  # type: ignore
        kbretriever,
        tokenizer,
        kb_token_layer_frequency,
        total_steps,
        args.lr,
        device,
        use_lr_decay,
        kb_size,  # type: ignore
        llm_ckpt_name,
        model_save_dir,
        sep_query_head=sep_query_head,
        max_seq_len=max_seq_len,
        dataset_format=dataset_type,
        keep_old_checkpoints=args.keep_old_checkpoints,
        test_dataset=test_dataset,
        precomputed_test_embed_keys_path=args.test_precomputed_embed_keys_path,
        precomputed_test_embed_values_path=args.test_precomputed_embed_values_path,
        test_kb_size=args.test_kb_size,
        test_query_size=args.test_query_size,
        test_kb_scale_factor=args.test_kb_scale_factor,
        test_kb_scale_factor_range=args.test_kb_scale_factor_range,
        eval_step=args.eval_step,
        format_short=args.format_short,
    )

    logger.info(f"Number of trainable parameters: {_get_parameter_count(encoder):,}")

    print("[INFO]: Training started")

    try: 
        trainer.train(
            training_set,
            B,
            gradient_accm_step,
            outlier_num,
            use_data_aug=use_data_aug,
            multi_entities=multi_entities,
            use_extended_qa=use_extended_qa,
            save_period=args.save_period,
            resumed_step=resumed_step,
            kb_config=kb_config,
        )
    except Exception as e:
        print("Training crashed:", e)
        raise
    finally:
        wandb.finish(exit_code=1)



if __name__ == "__main__":
    main()
