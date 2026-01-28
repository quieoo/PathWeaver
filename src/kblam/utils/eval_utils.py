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

def _prune_for_olmo3(S: str) -> str:
    S = S.replace("<|im_end|>", "")
    S = S.replace("<|im_start|>assistant", "\n\n")
    S = S.replace("<|im_start|>user", "")
    S = S.replace("<|endoftext|>", "")
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


def format_QA_olmo3(Q: str, A: str, tokenizer):
    """
    Format a single-turn QA pair for OLMo3 using the official chat template.

    Returns a string that can be tokenized, or directly tokenized outputs.
    """
    messages = [{"role": "user", "content": Q}]
    if A is not None:
        messages.append({"role": "assistant", "content": A})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(A is None),
    )




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

    # 找出所有作为「单词」出现的 is / are / was / were
    matches = list(re.finditer(r'\b(is|are|was|were)\b', text))

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


# def answer_question_deterministic(
#     tokenizer: transformers.PreTrainedTokenizer,
#     model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
#     Q: str,
#     kb=None,
#     kb_config: Optional[KBLaMConfig] = None,
#     kb_adj: Optional[torch.Tensor] = None,
#     attention_save_loc: Optional[str] = None,
#     save_attention_weights: bool = False,
#     attention_file_base_name: Optional[str] = None,
#     save_attn_weights_policy: str = "prefill-all-layer",
# ):
#     for m in model_question_format_mapping:
#         if kb_config.format_short:
#                 input_str = format_Q_llama_short(Q)
#         elif isinstance(model, m):
#             input_str = model_question_format_mapping[m](Q)
            
#     tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to("cuda")
#     input_ids, attention_masks = (
#         tokenizer_output["input_ids"],
#         tokenizer_output["attention_mask"],
#     )

#     with torch.autograd.no_grad():
#         outputs = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_masks,
#             kb_kvs=kb,
#             max_new_tokens=150,
#             tokenizer=tokenizer,
#             output_attentions=True,
#             kb_config=kb_config,
#             kb_adj=kb_adj,
#             pad_token_id=tokenizer.eos_token_id,
#             save_attention_weights=save_attention_weights,
#             attention_file_base_name=attention_file_base_name,
#             attention_save_loc=attention_save_loc,
#             save_attn_weights_policy=save_attn_weights_policy,
#             do_sample=False,    # 确定性结果
#             top_p=None,
#         ).squeeze()
#     outputs = tokenizer.decode(outputs, skip_special_tokens=False)

#     for m in model_prune_format_mapping:
#         if isinstance(model, m):
#             pruned_output = model_prune_format_mapping[m](outputs)
#     return pruned_output

def build_4d_attention_mask(attention_mask_2d: torch.Tensor, dtype: torch.dtype):
    """
    训练阶段强制构造 4D mask，避免 OLMo3 eager 路径下 attention_mask=None 导致 KB injector 崩溃。
    输出 shape: (B, 1, Q, K) = (B, 1, T, T)
    值：允许=0；禁止=大负数
    """
    bsz, seqlen = attention_mask_2d.shape
    neg = torch.finfo(dtype).min
    # base (B, 1, 1, K)
    base = (1.0 - attention_mask_2d.float()) * neg
    base = base[:, None, None, :]  # (B,1,1,K)
    # expand to (B,1,Q,K)
    return base.expand(bsz, 1, seqlen, seqlen).to(dtype)

def extract_assistant_answer(text: str) -> str:
    """
    Extract the assistant's final answer from OLMo3-style raw output.
    """
    # 找到最后一个 assistant 段
    match = re.search(
        r"<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|$)",
        text,
        re.DOTALL,
    )

    if match:
        answer = match.group(1).strip()
        return answer

    # fallback：如果格式不完整，返回原文本
    return text.strip()

@torch.no_grad()
def olmo3_greedy_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kb_kvs=None,
    kb_adj=None,
    kb_config=None,
    max_new_tokens: int = 64,
):
    device = prompt_ids.device
    generated = prompt_ids.clone()

    for _ in range(max_new_tokens):
        out = model(
            input_ids=generated,
            attention_mask=attention_mask,
            kb_kvs=kb_kvs,
            kb_adj=kb_adj,
            kb_config=kb_config,
        )

        next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        tok = tokenizer.decode(
            next_token[0],
            skip_special_tokens=False
        )

        if (
            next_token.item() == tokenizer.eos_token_id
            or tok in ("<|im_end|>", "<|endoftext|>")
        ):
            break

        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )

    return generated


def answer_question_deterministic(
    tokenizer: transformers.PreTrainedTokenizer,
    model,
    Q: str,
    kb=None,
    kb_config: Optional[KBLaMConfig] = None,
    kb_adj: Optional[torch.Tensor] = None,
    attention_save_loc: Optional[str] = None,
    save_attention_weights: bool = False,
    attention_file_base_name: Optional[str] = None,
    save_attn_weights_policy: str = "prefill-all-layer",
):
    device = next(model.parameters()).device

    # ============================================================
    # 1. OLMo3 分支（推荐 & 正确做法）
    # ============================================================
    # print(f"model class: {model.__class__.__name__.lower()}, model type: {model.config.model_type}")
    if model.__class__.__name__.lower().startswith("olmo3forcausallm") or \
       model.config.model_type == "olmo3":

        prompt = format_QA_olmo3(Q, None, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generated = olmo3_greedy_generate(
            model,
            tokenizer,
            inputs.input_ids,
            inputs.attention_mask,
            kb_kvs=kb,
            kb_adj=kb_adj,
            kb_config=kb_config,
        )

        answer = tokenizer.decode(
            generated[0][inputs.input_ids.size(1):],
            skip_special_tokens=True,
        )

        return answer

    # ============================================================
    # 2. LLaMA / Phi3 分支（保持你原来的逻辑）
    # ============================================================
    for m in model_question_format_mapping:
        if kb_config is not None and getattr(kb_config, "format_short", False):
            input_str = format_Q_llama_short(Q)
        elif isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)

    tokenizer_output = tokenizer(
        input_str, return_tensors="pt", padding=True
    ).to(device)

    input_ids = tokenizer_output["input_ids"]
    attention_masks = tokenizer_output["attention_mask"]

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=150,
            tokenizer=tokenizer,
            output_attentions=save_attention_weights,
            kb_config=kb_config,
            kb_adj=kb_adj,
            pad_token_id=tokenizer.eos_token_id,
            save_attention_weights=save_attention_weights,
            attention_file_base_name=attention_file_base_name,
            attention_save_loc=attention_save_loc,
            save_attn_weights_policy=save_attn_weights_policy,
            do_sample=False,
            top_p=None,
        ).squeeze()

    outputs_text = tokenizer.decode(outputs, skip_special_tokens=False)

    for m in model_prune_format_mapping:
        if isinstance(model, m):
            outputs_text = model_prune_format_mapping[m](outputs_text)

    return outputs_text
