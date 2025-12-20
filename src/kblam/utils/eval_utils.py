from typing import Optional
import re

import numpy as np
import torch
import transformers

from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM

instruction_prompts = """
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
"""

instruction_prompts_multi_entities = """
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
"""

zero_shot_prompt = """
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
"""

zero_shot_prompt_multi_entities = """
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
"""


def _prune_for_llama(S: str) -> str:
    S = S.replace("<|eot_id|>", "")
    S = S.replace("<|start_header_id|>assistant<|end_header_id|>", "\n\n")
    S = S.replace("<|start_header_id|>user<|end_header_id|>", "")
    S = S.replace("<|end_of_text|>", "")
    return S


def _prune_for_phi3(S: str) -> str:
    S = S.replace("<|end|>", "")
    S = S.replace("<|assistant|>", "\n\n")
    S = S.replace("<|user|>", "")
    return S


def softmax(x: np.array, axis: int) -> np.array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )

def format_Q_llama_short(Q: str):
    # short answer 
    Q = f"{Q} Answer with the shortest span from the context, do not add extra words and do not repeat the question."
    return (
        "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>" + "<|start_header_id|>assistant<|end_header_id|>"
    )


def format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"

def format_QA_llama(Q: str, A: str):

    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )

def format_QA_llama_short(Q: str, A: str):
    # short answer 
    Q = f"{Q} Answer with the shortest span from the context, do not add extra words and do not repeat the question."

    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )

def format_QA_phi3(Q: str, A: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"


model_question_format_mapping = {
    KblamLlamaForCausalLM: format_Q_llama,
    KBLaMPhi3ForCausalLM: format_Q_phi3,
}
model_prune_format_mapping = {
    KblamLlamaForCausalLM: _prune_for_llama,
    KBLaMPhi3ForCausalLM: _prune_for_phi3,
}


def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    Q: str,
    kb=None,
    kb_config: Optional[KBLaMConfig] = None,
    attention_save_loc: Optional[str] = None,
    save_attention_weights: bool = False,
    attention_file_base_name: Optional[str] = None,
):
    for m in model_question_format_mapping:
        if isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)
    tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=150,
            tokenizer=tokenizer,
            output_attentions=True,
            kb_config=kb_config,
            pad_token_id=tokenizer.eos_token_id,
            save_attention_weights=save_attention_weights,
            attention_file_base_name=attention_file_base_name,
            attention_save_loc=attention_save_loc,
        ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)

    for m in model_prune_format_mapping:
        if isinstance(model, m):
            pruned_output = model_prune_format_mapping[m](outputs)
    return pruned_output

def format_output_for_synthetic(model_output: str) -> str:

    text = model_output

    # 找出所有作为「单词」出现的 is / are
    matches = list(re.finditer(r'\b(is|are)\b', text))

    if matches:
        # 如果有多个 is/are，就取最后一个；否则就取唯一的那个
        if len(matches) >= 2:
            cut_pos = matches[-1].end()
        else:
            cut_pos = matches[0].end()

        # 从选定的 is/are 之后截取
        model_output = text[cut_pos:].strip()
        # 可选：去掉前导标点和空格，和末尾的句号等
        model_output = model_output.lstrip(' ,.:;').rstrip(' .;')

    return model_output


def answer_question_deterministic(
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
    Q: str,
    kb=None,
    kb_config: Optional[KBLaMConfig] = None,
    kb_adj: Optional[torch.Tensor] = None,
    attention_save_loc: Optional[str] = None,
    save_attention_weights: bool = False,
    attention_file_base_name: Optional[str] = None,
    save_attn_weights_policy: str = "prefill-all-layer",
):
    for m in model_question_format_mapping:
        if kb_config.format_short:
                input_str = format_Q_llama_short(Q)
        elif isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)
            
    tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to("cuda")
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=150,
            tokenizer=tokenizer,
            output_attentions=True,
            kb_config=kb_config,
            kb_adj=kb_adj,
            pad_token_id=tokenizer.eos_token_id,
            save_attention_weights=save_attention_weights,
            attention_file_base_name=attention_file_base_name,
            attention_save_loc=attention_save_loc,
            save_attn_weights_policy=save_attn_weights_policy,
            do_sample=False,    # 确定性结果
            top_p=None,
        ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)

    for m in model_prune_format_mapping:
        if isinstance(model, m):
            pruned_output = model_prune_format_mapping[m](outputs)
    return pruned_output