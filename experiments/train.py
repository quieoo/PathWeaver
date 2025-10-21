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

import numpy as np
import torch
import transformers
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
from kblam.utils.data_utils import augment_row, generate_multi_entity_qa, get_i_dont_know_ans
from kblam.utils.train_utils import context_set_size_scheduler, get_kb_embd, setup_scheduler_and_optimizer
import re
import shutil
import random
LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

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
parser.add_argument("--train_dataset",type=str,default="synthetic")
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
parser.add_argument("--dataset_dir", type=str, default="synthetic_data")
parser.add_argument("--model_dir_to_resume", type=str, default=None, help="Checkpoint directory to resume training")
# parser.add_argument("--hf_model_spec", type=str, default="meta-llama/Llama-3.2-1B-Instruct", choices=["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Llama-3.2-1B-Instruct"])
parser.add_argument("--hf_model_spec", type=str, default="meta-llama/Llama-3.2-1B-Instruct")

parser.add_argument("--hf_token", type=str,default=None,help="Huggingface token")
parser.add_argument("--model_save_dir", type=str, default="output", help="Place to save the checkpoints")
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
parser.add_argument("--llm_type",type=str,default="llama3",choices=["llama3", "phi3"])
parser.add_argument("--max_seq_len",type=int,default=None)
parser.add_argument("--save_period", type=int, default=2000, help="Save every n steps")
# fmt: on


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


def _format_QA_llama(Q: str, A: str):
    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )


def _format_QA_phi3(Q: str, A: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"


# 构建训练标签：只计算模型对assistant回答部分的预测损失，而忽略对用户提问等其他内容的预测
def _create_labels_for_llama(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
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

def _create_labels_for_llama_enhanced(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Create labels for Llama-3-style dialogue:
    - Mask everything before and including <|start_header_id|>assistant<|end_header_id|>
    - Only compute loss on the assistant's response.
    """
    labels = input_ids.clone()
    labels[:] = -100  # 默认全部 mask

    # Get the token IDs for the assistant header
    assistant_header = tokenizer("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False).input_ids
    # e.g., [128006, 78191, 128007]

    for b in range(input_ids.size(0)):
        seq = input_ids[b].tolist()
        # Find the starting index of the assistant header
        start_idx = None
        for i in range(len(seq) - len(assistant_header) + 1):
            if seq[i:i + len(assistant_header)] == assistant_header:
                start_idx = i
                break
        
        if start_idx is not None:
            # Assistant response starts AFTER the header
            response_start = start_idx + len(assistant_header)
            labels[b, response_start:] = input_ids[b, response_start:]
        # else: no assistant header found → keep all -100 (no loss)

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

# 从数据集中构建训练批次：对应输入字符串的Token id，注意力掩码以及标签
def get_batch(
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
    """
    dataset: List of dictionary, denoting the KB, used to extract QA pairs
    model: The LLM, used to provide the embedding
    kb_embedding: KB embedding (differentiable)
    B: Batchsize
    include_outlier : Create a batch of question without answer in the KB.
    multi_entities : Create a batch of question that involves more than one entities.
    """
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
        labels = label_func(input_ids, input_strs, tokenizer)
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
):
    labels = []
    batch_indices = np.random.choice(len(dataset), B, replace=False)

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
def _load_cached_embeddings(encoder_model_spec: str, dataset_dir: str, dataset_name: str, key_embd_src: str):
    if encoder_model_spec == "OAI":
        encoder_model_spec_str = "oai"
    else:
        encoder_model_spec_str = encoder_model_spec
    key_embds = np.load(
        os.path.join(
            dataset_dir,
            dataset_name,
            f"{encoder_model_spec_str}_embd_{key_embd_src}.npy",
        )
    ).astype("float32")
    if key_embd_src == "answer":
        # If we are using the answer string as the key, we also use it as the value string
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                dataset_name,
                f"{encoder_model_spec_str}_embd_answer.npy",
            )
        ).astype("float32")
    else:
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                dataset_name,
                f"{encoder_model_spec_str}_embd_value.npy",
            )
        ).astype("float32")
    return key_embds, value_embds


def get_step_config(
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
    config = {}
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


class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray],
        value_embds: Optional[np.ndarray],
    ):
        self.encoder = encoder
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.dataset = dataset

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False
    # 为当前训练批次生成KB embedding：与当前批次直接相关的KB和随机采样的上下文KB
    def get_key_embeddings(self, batch_indices, batch_size, step, kb_size):
        # 索引KB embedding并应用encoder
        if self._use_cached_embd():
            train_set_key, train_set_val = get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            train_set_key, train_set_val = get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)
        # train_set形状应该是(batch_size, embedding_dim)

        # 形状变为(batch_size, 1, embedding_dim)
        if len(train_set_key.shape) == 2:
            train_set_key = train_set_key.unsqueeze(0).transpose(0, 1)
            train_set_val = train_set_val.unsqueeze(0).transpose(0, 1)

        # 获得实时的知识库大小以及随机的知识库内容，获取其键值embedding
        context_set_size = context_set_size_scheduler(step, kb_size)
        context_set_index = np.random.choice(len(self.dataset), context_set_size, replace=False)  # type: ignore
        if self._use_cached_embd():
            context_set_key, context_set_val = get_kb_embd(
                self.encoder,
                context_set_index,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            context_set_key, context_set_val = get_kb_embd(self.encoder, context_set_index, kb_dict=self.dataset)
        # context_set形状应该是(context_set_size, embedding_dim)
        
        # 形状变为(batch_size, context_set_size, embedding_dim)
        context_set_key = context_set_key.unsqueeze(0).expand(batch_size, *context_set_key.shape)
        context_set_val = context_set_val.unsqueeze(0).expand(batch_size, *context_set_val.shape)
        # context_set_val = torch.randn_like(context_set_val)
        # Idea: Try torch.randn here context_set_tokens??

        true_kb_copy = 1
        kb_embedding = (
            torch.concat([*([train_set_key] * true_kb_copy), context_set_key], 1),
            torch.concat([*([train_set_val] * true_kb_copy), context_set_val], 1),
        )
        # 最后形状变为(batch_size, 1+context_set_size, embedding_dim)
        return kb_embedding

# DATASET_SUPPORT
    def get_key_embeddings_document(self, start_ids, num_triples, batch_size, step, kb_size):
        if not self._use_cached_embd():
            print("Current only supports cached KB embedding")
            return None
        if len(start_ids) != batch_size:
            print("Batch size mismatch")
            return None

        # 先收集三元组并执行编码
        key_embeddings = [[] for _ in range(batch_size)]
        value_embeddings = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            start_id = start_ids[i]
            for j in range(num_triples[i]):
                k_embed = self.encoder.encode_key(base_emb=self.key_embds[start_id + j])  # pyright: ignore[reportOptionalSubscript]
                v_embed = self.encoder.encode_val(base_emb=self.value_embds[start_id + j])  # pyright: ignore[reportOptionalSubscript]
                key_embeddings[i].append(k_embed)
                value_embeddings[i].append(v_embed)

        # 处理变长序列：首先将每个样本的嵌入堆叠成张量
        # 然后进行填充使所有样本具有相同的序列长度
        key_tensor_list = []
        value_tensor_list = []
        
        for i in range(batch_size):
            # 将每个样本的嵌入列表转换为张量
            key_tensor_list.append(torch.stack(key_embeddings[i]))
            value_tensor_list.append(torch.stack(value_embeddings[i]))
            
        # 获取最大序列长度
        max_seq_len = max([t.size(0) for t in key_tensor_list])
        
        # 对所有张量进行填充以匹配最大序列长度
        padded_key_tensors = []
        padded_value_tensors = []
        
        for i in range(batch_size):
            current_seq_len = key_tensor_list[i].size(0)
            if current_seq_len < max_seq_len:
                # 计算需要填充的数量
                padding_size = max_seq_len - current_seq_len
                # 创建填充张量（使用0填充）
                key_padding = torch.zeros(padding_size, key_tensor_list[i].size(1), 
                                          dtype=key_tensor_list[i].dtype, 
                                          device=key_tensor_list[i].device)
                value_padding = torch.zeros(padding_size, value_tensor_list[i].size(1), 
                                            dtype=value_tensor_list[i].dtype, 
                                            device=value_tensor_list[i].device)
                # 拼接原始张量和填充张量
                padded_key = torch.cat([key_tensor_list[i], key_padding], dim=0)
                padded_value = torch.cat([value_tensor_list[i], value_padding], dim=0)
            else:
                # 如果已经是最大长度，则不需要填充
                padded_key = key_tensor_list[i]
                padded_value = value_tensor_list[i]
                
            padded_key_tensors.append(padded_key)
            padded_value_tensors.append(padded_value)
        
        # 最后堆叠所有样本的张量
        key_embeddings = torch.stack(padded_key_tensors, dim=0)
        value_embeddings = torch.stack(padded_value_tensors, dim=0)

        # print(f"----shape of key embeddings: {key_embeddings.shape}")
        return (key_embeddings, value_embeddings)
    def get_embeddings(self, start_id_lists, num_triples_lists, batch_size):
        if not self._use_cached_embd():
            print("Currently only supports cached KB embedding")
            return None

        if len(start_id_lists) != batch_size or len(num_triples_lists) != batch_size:
            print("Batch size mismatch")
            return None

        # Step 1: Collect all indices needed across the batch
        all_indices = []
        seq_lengths = []  # number of triples per sample

        for i in range(batch_size):
            starts = start_id_lists[i] if isinstance(start_id_lists[i], list) else [start_id_lists[i]]
            nums = num_triples_lists[i] if isinstance(num_triples_lists[i], list) else [num_triples_lists[i]]

            sample_indices = []
            for start, num in zip(starts, nums):
                if num > 0:
                    # Ensure we don't go out of bounds (optional safety check)
                    if start + num > len(self.key_embds):
                        raise IndexError(f"Index out of range: start={start}, num={num}, total={len(self.key_embds)}")
                    sample_indices.extend(range(start, start + num))
            all_indices.extend(sample_indices)
            seq_lengths.append(len(sample_indices))

        total_triples = len(all_indices)
        if total_triples == 0:
            print("WARNING: No triples found in batch.")
            # Edge case: no triples in entire batch
            # Create dummy tensors with correct embedding dim
            dummy_key = self.encoder.encode_key(base_emb=np.zeros_like(self.key_embds[0:1]))  # (1, Dk)
            dummy_val = self.encoder.encode_val(base_emb=np.zeros_like(self.value_embds[0:1]))  # (1, Dv)
            dim_k = dummy_key.shape[1]
            dim_v = dummy_val.shape[1]
            max_len = 1
            padded_keys = torch.zeros(batch_size, max_len, dim_k, dtype=dummy_key.dtype, device=dummy_key.device)
            padded_vals = torch.zeros(batch_size, max_len, dim_v, dtype=dummy_val.dtype, device=dummy_val.device)
            return padded_keys, padded_vals

        # Step 2: Batch extract embeddings (as numpy)
        key_batch_np = self.key_embds[all_indices]      # (total_triples, Dk)
        val_batch_np = self.value_embds[all_indices]    # (total_triples, Dv)

        # Step 3: Batch encode via encoder (only 2 calls!)
        key_encoded = self.encoder.encode_key(base_emb=key_batch_np)    # (total_triples, Dk')
        val_encoded = self.encoder.encode_val(base_emb=val_batch_np)  # (total_triples, Dv')

        # Step 4: Split into per-sample sequences
        key_seq_list = []
        val_seq_list = []
        start = 0
        for length in seq_lengths:
            if length == 0:
                # Create empty tensor with correct feature dim
                k = torch.empty(0, key_encoded.shape[1], device=key_encoded.device, dtype=key_encoded.dtype)
                v = torch.empty(0, val_encoded.shape[1], device=val_encoded.device, dtype=val_encoded.dtype)
            else:
                k = key_encoded[start:start + length]
                v = val_encoded[start:start + length]
                start += length
            key_seq_list.append(k)
            val_seq_list.append(v)

        # Step 5: Post-padding (pad at the end of sequence)
        max_seq_len = max(t.size(0) for t in key_seq_list)
        padded_keys = []
        padded_vals = []

        for k, v in zip(key_seq_list, val_seq_list):
            cur_len = k.size(0)
            if cur_len < max_seq_len:
                pad_len = max_seq_len - cur_len
                k_pad = torch.zeros(pad_len, k.size(1), dtype=k.dtype, device=k.device)
                v_pad = torch.zeros(pad_len, v.size(1), dtype=v.dtype, device=v.device)
                k = torch.cat([k, k_pad], dim=0)
                v = torch.cat([v, v_pad], dim=0)
            padded_keys.append(k)
            padded_vals.append(v)

        return torch.stack(padded_keys, dim=0), torch.stack(padded_vals, dim=0)
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
        dataset_format: str = "default",
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

        if isinstance(llm_model, KBLaMPhi3ForCausalLM):  # Phi3
            self._get_batch = partial(get_batch, _format_QA_phi3, _create_labels_for_phi3)
            self._get_params = _get_phi3_query_head_parameters
        elif isinstance(llm_model, KblamLlamaForCausalLM):  # llama
# DATASET_SUPPORT
            if dataset_format == "default":            
                self._get_batch = partial(get_batch, _format_QA_llama, _create_labels_for_llama)
            elif dataset_format == "autoschemakg":
                self._get_batch = partial(get_batch_from_document, _format_QA_llama, _create_labels_for_llama)
            elif dataset_format == "musique":
                self._get_batch = partial(get_batch_musique, _format_QA_llama, _create_labels_for_llama)
            else:
                raise ValueError(f"{dataset_format} not recognised")
            self._get_params = _get_llama3_query_head_parameters
        else:
            raise ValueError(f"{llm_model} not recognised")

        self.scheduler, self.optim = self.setup_scheduler_and_optim()

        self.model, self.optim, self._get_batch, self.kbretriever.encoder = self.accelerator.prepare(
            self.model, self.optim, self._get_batch, self.kbretriever.encoder
        )

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
        return scheduler, optim

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
                        a_step,
                        grad_accum_steps,
                        use_data_aug,
                        outlier_num,
                        multi_entities,
                        use_extended_qa,
                    )
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
                    if self.dataset_format == "default":
                        # 可能是个拼写错误，实际应该是获得kb的键值嵌入，作为模型的kv Cache
                        # KB Token来自两个源头：当前批次索引所对应的KB Token（一个QA对有一个KB Token）；根据kb_size随机选择一些KB Token
                        # 使用当前的encoder对KB Token进行编码，获得键值嵌入
                        kb_embedding = self.kbretriever.get_key_embeddings(
                            batch_indices, len(input_ids), step, self.kb_size
                        )
                        # 前向传播（执行一次推理）
                    elif self.dataset_format == "autoschemakg":
                        # 根据batch_indices从training_set中获取"start_id"和"num_triples"域的值
                        start_ids=[]
                        num_triples=[]
                        for i in batch_indices:
                            start_ids.append(training_set[i]["start_id"])
                            num_triples.append(training_set[i]["num_triples"])

                        kb_embedding = self.kbretriever.get_key_embeddings_document(
                            start_ids, num_triples, len(input_ids), step, self.kb_size
                        )
                    elif self.dataset_format == "musique":
                        start_ids = [[] for _ in range(batch_size)]
                        num_triples = [[] for _ in range(batch_size)]

                        for i in range(batch_size):
                            sample = training_set[batch_indices[i]]
                            paragraphs = sample["paragraphs"]
                            assert len(paragraphs) > 0, f"Sample {batch_indices[i]} has no paragraphs"

                            if self.kb_size == -1:
                                # Use all triples from all paragraphs as one contiguous block
                                start_ids[i] = [paragraphs[0]["start_id"]]
                                total_triples = sum(p["num_triples"] for p in paragraphs)
                                num_triples[i] = [total_triples]
                                

                            elif self.kb_size == 1:
                                # Use only ground-truth supporting paragraphs
                                for qd in sample["question_decomposition"]:
                                    true_idx = qd["paragraph_support_idx"]
                                    para = paragraphs[true_idx]
                                    start_ids[i].append(para["start_id"])
                                    num_triples[i].append(para["num_triples"])

                            else:
                                raise ValueError(f"Unsupported kb_size: {self.kb_size}")
                        # print(f"using {num_triples} triples")
                        kb_embedding = self.kbretriever.get_embeddings(start_ids, num_triples, batch_size)
                    else:
                        raise ValueError(f"Unknown data set format: {self.dataset_format}")
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        kb_kvs=kb_embedding,
                        output_attentions=True,
                        kb_config=kb_config,
                    )
                    logits = out["logits"]
                    # 打印部分结果
                    if a_step == 0 and step % 10 == 0:
                        batch_index = 0  # Which example in the batch to select
                        max_logits = logits.argmax(axis=2)
                        decoded_pred = self.tokenizer.decode(max_logits[batch_index, :-1])
                        sel_labels = labels[batch_index, :]
                        sel_labels = sel_labels[sel_labels >= 0]  # Remove padding token -100
                        decoded_gt = self.tokenizer.decode(sel_labels)
                        self.logger.info(f"KB SHAPE: {kb_embedding[0].shape}")
                        self.logger.info(f"GT: {decoded_gt}")
                        self.logger.info(f"PRED: {decoded_pred}")
                        wandb.log({"kbsize": kb_embedding[0].shape[1]})
                    # 计算交叉熵损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()
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
                    self.logger.info(f"step: {step}, loss: {avg_loss}")
                    num_candidates=0
                    wandb.log({'train_loss': np.mean(losses)})
                    train_losses.append(avg_loss)
                    pbar.update(task, advance=1, loss=avg_loss)

                # 保存模型参数
                if (step % save_period) == 0 and (step != start_step):
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

    print(vars(args))
    dataset_name = args.train_dataset
    seed = args.seed
    N = args.N
    B = args.B

    total_steps = args.total_steps
    encoder_spec = args.encoder_spec
    key_embd_src = args.key_embd_src
    use_data_aug = args.use_data_aug
    use_lr_decay = args.use_lr_decay
    use_cached_embd = args.use_cached_embd
    dataset_dir = args.dataset_dir
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

    torch.manual_seed(seed)
    np.random.seed(seed)

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
                'dataset': dataset_name,
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
        # We load the pre-computed version stored on the disk rather
        # than computing them on the fly to make things faster
        logger.info(f"Using pre-computed {encoder_spec} embedding")
        key_embds, value_embds = _load_cached_embeddings(encoder_spec, dataset_dir, dataset_name, key_embd_src)

    prefix_string = get_prefix_str(args)
    logger.info(f"Experiment prefix {get_prefix_str(args)}")

    if use_extended_qa:
        dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}_augmented.json")))
    else:
# DATASET_SUPPORT
        if dataset_name == "multi_wiki_qa_train":
            dataset_path=os.path.join(dataset_dir, f"{dataset_name}.json")
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = [json.loads(line.strip()) for line in f]
        elif "musique" in dataset_name:
            # search for dataset file: end with "json", include "train"
            dataset_path = glob.glob(os.path.join(dataset_dir, dataset_name, "*train*.json"))[0]
            print(f"[INFO]: Using dataset file {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        else:
            dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}.json")))

    N = min(N, len(dataset))
    training_set = dataset[:N]
    print(f"[INFO]: Loaded {N} samples for training")

    # Set up the LLM
    llm_model_spec = model_dir_to_resume if model_dir_to_resume else hf_model_spec

    resumed_step = 0 if not model_dir_to_resume else int(model_dir_to_resume.split("_")[-1])

    if model_dir_to_resume:
        print(f"[INFO]: Resuming from {model_dir_to_resume}, step: {resumed_step}")
    if llm_model_spec is None:
        raise ValueError("Either supply model_dir_to_resume or hf_model_spec")

    if hf_token is None and args.llm_type == "llama3" and not os.path.exists(llm_model_spec):
        raise ValueError("Please supply HuggingFace token(hf_token) when loading model Llama weights from HuggingFace")

    # Tokenizer comes from the base model
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_spec,
        trust_remote_code=True,
        token=hf_token if hf_token is args.llm_type == "llama3" else None,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # token=hf_token,
        )
    elif args.llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        ValueError(f"LLM type {args.llm_type} not recognised")

    print(f"[INFO]: Initialised model {llm_model_spec}")
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
        )

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
    llm_ckpt_name = f"{prefix_string}KeyFrom{key_embd_src}_{encoder_spec}_{dataset_name}_{llm_type}"

    if dataset_name == "multi_wiki_qa_train":
        dataset_format = "autoschemakg"
    elif "musique" in dataset_name:
        dataset_format = "musique"
    else:
        dataset_format = "default"

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
        dataset_format=dataset_format,
    )

    logger.info(f"Number of trainable parameters: {_get_parameter_count(encoder):,}")

    print("[INFO]: Training started")

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

    # save model
    trainer.save_state()


if __name__ == "__main__":
    main()
