from typing import Optional
import re
import time

import numpy as np
import torch
import transformers
from transformers.generation.streamers import BaseStreamer

from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.llama3_model import kblam_profile_set
from kblam.kblam_attention.kblam_path import (
    is_path_attn_trace_enabled,
    update_path_attn_trace_context,
)

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

QWEN3_SHORT_ANSWER_PROMPT = (
    "Answer with a short span from the context. "
    "Do not explain or output reasoning."
)


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


def _prune_for_qwen3(S: str) -> str:
    S = S.replace("<|im_end|>", "")
    S = S.replace("<|im_start|>assistant", "\n\n")
    S = S.replace("<|im_start|>user", "")
    S = S.replace("<|im_start|>system", "")
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


def format_Q_qwen3(Q: str, tokenizer):
    user_content = f"{Q}\n\n{QWEN3_SHORT_ANSWER_PROMPT}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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


def format_QA_qwen3(Q: str, A: str, tokenizer):
    user_content = f"{Q}\n\n{QWEN3_SHORT_ANSWER_PROMPT}"
    messages = [{"role": "user", "content": user_content}]
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
    if getattr(model.config, "model_type", None) == "qwen3":
        input_str = format_Q_qwen3(Q, tokenizer)
        tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(
            input_ids=tokenizer_output["input_ids"],
            attention_mask=tokenizer_output["attention_mask"],
            kb_kvs=kb,
            kb_config=kb_config,
            max_new_tokens=64,
            tokenizer=tokenizer,
            output_attentions=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            top_p=None,
            use_cache=True,
        ).squeeze()
        answer = tokenizer.decode(
            outputs[tokenizer_output["input_ids"].size(1):],
            skip_special_tokens=True,
        )
        return answer.strip()

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
    text = str(model_output).strip()
    if not text:
        return ""

    # Normalize common chat/template artifacts before extracting the answer.
    for token in (
        "<|begin_of_text|>",
        "begin_of_text|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<|end|>",
        "<|assistant|>",
        "<|user|>",
        "<|im_end|>",
        "<|im_start|>assistant",
        "<|im_start|>user",
        "<|endoftext|>",
    ):
        text = text.replace(token, " ")

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    def _clean_candidate(candidate: str) -> str:
        candidate = re.sub(r"\s+", " ", candidate).strip()
        candidate = re.sub(
            r"^(model output|output|answer|assistant|response)\s*:\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = candidate.strip()
        candidate = re.sub(r"^([\"'`])(.+)\1$", r"\2", candidate)
        candidate = candidate.rstrip(" \t\n\r,;:.")
        return candidate

    def _looks_like_prompt_line(line: str) -> bool:
        lowered = line.lower().strip()
        if not lowered:
            return True
        return (
            lowered.endswith("?")
            or lowered.startswith("please answer")
            or lowered.startswith("answer with")
            or lowered.startswith("answer the question")
            or lowered.startswith("question:")
            or lowered.startswith("ground truth:")
        )

    lines = [_clean_candidate(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    # Prefer the last line that looks like an answer instead of a prompt echo.
    answer_like_lines = [line for line in lines if not _looks_like_prompt_line(line)]
    if answer_like_lines:
        text = answer_like_lines[-1]
    elif lines:
        text = lines[-1]
    else:
        text = _clean_candidate(text)

    # If prompt text and answer still share one line, keep the tail span.
    tail_parts = [
        part.strip() for part in re.split(r"\n{2,}|(?<=\?)\s+", text) if part.strip()
    ]
    if tail_parts:
        text = _clean_candidate(tail_parts[-1])

    # Only strip the synthetic template when it is explicitly present.
    template_matches = list(
        re.finditer(
            r"\bThe\s+.+?\s+of\s+.+?\s+(?:is|are|was|were)\s+(.+?)(?=(?:\s*;\s*The\s+.+?\s+of\s+.+?\s+(?:is|are|was|were)\s+)|$)",
            text,
            flags=re.IGNORECASE,
        )
    )
    if template_matches:
        text = template_matches[-1].group(1)

    return _clean_candidate(text)


def strip_generation_prefix(output: str, model) -> str:
    text = str(output)
    if getattr(model.config, "model_type", None) == "qwen3":
        return text.lstrip()

    if text.startswith("\n\n"):
        return text[2:]

    return text.lstrip()


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


class TimingStreamer(BaseStreamer):
    """Capture first-token and decode timing for synchronous HF generate()."""

    def __init__(self):
        self._skip_prompt = True
        self.first_token_time: Optional[float] = None
        self.generated_tokens = 0

    def put(self, value):
        if self._skip_prompt:
            self._skip_prompt = False
            return

        if isinstance(value, torch.Tensor):
            token_count = int(value.numel())
        elif isinstance(value, (list, tuple)):
            token_count = len(value)
        else:
            token_count = 1

        if token_count <= 0:
            return

        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.generated_tokens += token_count

    def end(self):
        return

@torch.no_grad()
def chat_template_greedy_generate(
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

    def _update_trace_prompt_metadata(prompt_text: str, input_ids: torch.Tensor) -> None:
        if not is_path_attn_trace_enabled():
            return
        try:
            ids_1d = input_ids[0].detach().to("cpu")
            token_ids = ids_1d.tolist()
            token_strs = tokenizer.convert_ids_to_tokens(token_ids)
            update_path_attn_trace_context(
                model_type=getattr(model.config, "model_type", ""),
                tokenizer_name_or_path=getattr(tokenizer, "name_or_path", ""),
                prompt_text=prompt_text,
                prompt_input_ids=token_ids,
                prompt_tokens=[str(tok) for tok in token_strs],
                prompt_len=int(len(token_ids)),
            )
        except Exception as exc:
            update_path_attn_trace_context(prompt_trace_error=str(exc))

    # ============================================================
    # 1. OLMo3 分支
    # ============================================================
    # print(f"model class: {model.__class__.__name__.lower()}, model type: {model.config.model_type}")
    if model.__class__.__name__.lower().startswith("olmo3forcausallm") or \
       model.config.model_type == "olmo3":

        prompt = format_QA_olmo3(Q, None, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        _update_trace_prompt_metadata(prompt, inputs.input_ids)

        generated = chat_template_greedy_generate(
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

    if getattr(model.config, "model_type", None) == "qwen3":
        prompt = format_QA_qwen3(Q, None, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        _update_trace_prompt_metadata(prompt, inputs.input_ids)
        timing_streamer = TimingStreamer()
        gen_start = time.perf_counter()

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            kb_kvs=kb,
            kb_adj=kb_adj,
            kb_config=kb_config,
            max_new_tokens=16,
            tokenizer=tokenizer,
            output_attentions=save_attention_weights,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            top_p=None,
            use_cache=True,
            streamer=timing_streamer,
        ).squeeze()
        gen_end = time.perf_counter()

        if timing_streamer.first_token_time is None:
            prefill_s = gen_end - gen_start
            decode_s = 0.0
            decode_tokens = 0
        else:
            prefill_s = timing_streamer.first_token_time - gen_start
            decode_s = max(0.0, gen_end - timing_streamer.first_token_time)
            decode_tokens = max(0, timing_streamer.generated_tokens - 1)
        kblam_profile_set(prefill_s, decode_s, decode_tokens)

        answer = tokenizer.decode(
            outputs[inputs.input_ids.size(1):],
            skip_special_tokens=True,
        )
        return answer.strip()

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
    _update_trace_prompt_metadata(input_str, input_ids)

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


def output_dag_sample(sample):
    # Older DAG samples stored per-context triple_list, while newer ones only keep
    # plain supporting passages in context and place triples at the sample level.
    filtered_context = []
    for context in sample.get("context", []):
        if not isinstance(context, dict):
            filtered_context.append(context)
            continue
        triple_list = context.get("triple_list")
        if triple_list is None or triple_list:
            filtered_context.append(context)
    new_sample = sample.copy()
    new_sample["context"] = filtered_context
    return new_sample
