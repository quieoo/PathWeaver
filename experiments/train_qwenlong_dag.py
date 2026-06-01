import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.theme import Theme
from safetensors.torch import save_file
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from kblam.dag_kv_retriever import DAGKVKBRetriever
from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.qwen3.kblam_qwen3_moe_attention import load_kblam_qwen3_moe_model
from kblam.utils.eval_utils import format_QA_qwen3
from kblam.utils.train_utils import (
    get_prefix_str,
    setup_scheduler_and_optimizer,
    setup_scheduler_and_optimizer_with_warmup,
)


LOGFORMAT_RICH = "%(message)s"
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)
console = Console(theme=custom_theme)


def create_custom_progress_bar(
    console: Console = None,  # type: ignore
    color: str = "cyan",
    show_time: bool = True,
    show_spinner: bool = True,
    spinner_style: str = "dots",
    disable: bool = False,
) -> Progress:
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


def _get_kb_encoder_out_dim(model_config, kb_layer_frequency: int) -> int:
    slots = model_config.num_hidden_layers // kb_layer_frequency + 1
    head_dim = getattr(model_config, "head_dim", None)
    num_heads = getattr(model_config, "num_attention_heads", None)

    if head_dim is not None and num_heads is not None:
        per_slot_dim = head_dim * num_heads
    else:
        per_slot_dim = model_config.hidden_size

    return per_slot_dim * slots


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
    raise ValueError(f"Unsupported file format: {path}")


def _create_labels_for_qwen3(
    input_ids: torch.Tensor,
    tokenizer,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)

    assistant_start = tokenizer(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    ).input_ids
    assistant_end = tokenizer(
        "<|im_end|>",
        add_special_tokens=False,
    ).input_ids
    qwen_pad_id = tokenizer.pad_token_id

    for b in range(input_ids.size(0)):
        seq = input_ids[b].tolist()
        start_idx = None
        end_idx = None

        for i in range(len(seq) - len(assistant_start) + 1):
            if seq[i : i + len(assistant_start)] == assistant_start:
                start_idx = i + len(assistant_start)
                break

        if start_idx is None:
            continue

        for i in range(start_idx, len(seq) - len(assistant_end) + 1):
            if seq[i : i + len(assistant_end)] == assistant_end:
                end_idx = i + len(assistant_end)
                break

        if end_idx is None:
            end_idx = int((attention_mask[b] == 1).nonzero()[-1].item()) + 1

        labels[b, start_idx:end_idx] = input_ids[b, start_idx:end_idx]

        if qwen_pad_id is not None:
            pad_positions = (input_ids[b] == qwen_pad_id) & (attention_mask[b] == 0)
            labels[b] = labels[b].masked_fill(pad_positions, -100)

        labels[b] = labels[b].masked_fill(attention_mask[b] == 0, -100)

    return labels


def build_qwen_batch(
    dataset: Sequence[Dict[str, Any]],
    tokenizer,
    device: torch.device,
    batch_size: int,
    random_sample: bool = True,
    max_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    if random_sample:
        batch_indices = np.random.choice(len(dataset), batch_size, replace=False)
    else:
        batch_indices = np.arange(batch_size)

    prompts: List[str] = []
    real_batch_indices: List[int] = []
    for idx in batch_indices:
        row = dataset[int(idx)]
        q = row.get("Q", row.get("question", ""))
        a = row.get("A", row.get("answer", ""))
        if q is None or a is None:
            continue
        prompts.append(format_QA_qwen3(str(q), str(a), tokenizer))
        real_batch_indices.append(int(idx))

    tokenized = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = _create_labels_for_qwen3(input_ids, tokenizer, attention_mask)

    if max_seq_len is not None:
        input_ids = input_ids[:, :max_seq_len]
        attention_mask = attention_mask[:, :max_seq_len]
        labels = labels[:, :max_seq_len]

    return input_ids, attention_mask, labels, np.asarray(real_batch_indices, dtype=np.int64)


def compute_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels != -100
    weights = valid_mask.sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    weights = weights.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)

    loss_fct = CrossEntropyLoss(reduction="none")
    loss = (loss_fct(shift_logits, shift_labels) * weights.max() / weights.clamp_min(1)).mean()
    return loss


def parse_resume_step(path: Optional[str]) -> int:
    if not path:
        return 0
    match = re.search(r"_step_(\d+)$", path.rstrip("/"))
    return int(match.group(1)) if match else 0


def get_q_proj_new_parameters(
    model,
    kb_layer_frequency: int,
) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if "q_proj_new" not in name:
            continue
        layer_match = re.search(r"layers\.(\d+)\.", name)
        if layer_match is None:
            continue
        layer_idx = int(layer_match.group(1))
        if layer_idx % kb_layer_frequency == 0:
            params.append(param)
    return params


def sync_module_gradients(module: torch.nn.Module, world_size: int) -> None:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return

    for param in module.parameters():
        grad = param.grad
        if grad is None:
            continue
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(world_size)


def log_trainable_parameter_summary(logger, model, encoder) -> None:
    q_proj_new_names = [name for name, p in model.named_parameters() if p.requires_grad]
    encoder_names = [name for name, p in encoder.named_parameters() if p.requires_grad]
    total_q_proj = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    total_encoder = sum(p.numel() for _, p in encoder.named_parameters() if p.requires_grad)

    logger.info(
        "Trainable parameter summary: q_proj_new=%s tensors (%s params), encoder=%s tensors (%s params)",
        len(q_proj_new_names),
        f"{total_q_proj:,}",
        len(encoder_names),
        f"{total_encoder:,}",
    )
    logger.info("First trainable model tensors: %s", q_proj_new_names[:6])
    logger.info("First trainable encoder tensors: %s", encoder_names[:6])


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def log_cuda_memory(
    logger,
    *,
    label: str,
    device: torch.device,
    accelerator: Accelerator,
    force: bool = False,
) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    if not (force or accelerator.is_main_process):
        return

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    logger.info(
        "[MEM][rank=%s] %s | alloc=%s reserved=%s peak_alloc=%s peak_reserved=%s free=%s total=%s",
        accelerator.process_index,
        label,
        _format_gib(allocated),
        _format_gib(reserved),
        _format_gib(max_allocated),
        _format_gib(max_reserved),
        _format_gib(free_mem),
        _format_gib(total_mem),
    )


def save_training_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    encoder,
    kb_config: KBLaMConfig,
    output_dir: Path,
    ckpt_name: str,
    step: int,
    optimizer=None,
    scheduler=None,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    model_dir = output_dir / f"{ckpt_name}_step_{step}"
    encoder_dir = output_dir / f"{ckpt_name}_step_{step}_encoder"
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    full_state = accelerator.get_state_dict(model)
    q_proj_state = {
        k: v.detach().cpu()
        for k, v in full_state.items()
        if "q_proj_new" in k
    }
    save_file(q_proj_state, str(model_dir / "model.safetensors"))

    encoder_state = {
        k: v.detach().cpu()
        for k, v in accelerator.unwrap_model(encoder).state_dict().items()
    }
    torch.save(encoder_state, encoder_dir / "encoder.pt")

    with open(model_dir / "kb_config_explicit.json", "w", encoding="utf-8") as f:
        f.write(kb_config.to_json_string())

    if optimizer is not None:
        accelerator.save(optimizer.state_dict(), model_dir / "optimizer.pt")
    if scheduler is not None:
        accelerator.save(scheduler.state_dict(), model_dir / "scheduler.pt")


def main():
    logging_handlers = [RichHandler(console=console, rich_tracebacks=True)]
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format=LOGFORMAT_RICH,
        datefmt="[%X]",
        handlers=logging_handlers,
    )
    logger = logging.getLogger("train_qwenlong_dag")

    parser = argparse.ArgumentParser(
        description="DAG-only multi-GPU trainer for KBLaM QwenLong/Qwen3-MoE models.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_type", type=str, default="dag")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sep_query_head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_oai_embd", action="store_true")
    parser.add_argument("--use_cached_embd", action="store_true")
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--keep_top_k_ckpt", type=int, default=1)
    parser.add_argument("--encoder_spec", type=str, default="OAI")
    parser.add_argument("--key_embd_src", type=str, default="key")
    parser.add_argument("--use_data_aug", action="store_true")
    parser.add_argument("--use_lr_decay", action="store_true")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--train_precomputed_embed_keys_path", type=str, default=None)
    parser.add_argument("--train_precomputed_embed_values_path", type=str, default=None)
    parser.add_argument("--model_dir_to_resume", type=str, default=None)
    parser.add_argument("--resume_steps", type=bool, default=False)
    parser.add_argument("--hf_model_spec", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default="output")
    parser.add_argument("--kb_size", type=int, default=None)
    parser.add_argument("--dynamic_kb_size", nargs=2, type=int, default=None)
    parser.add_argument("--duplicate_true_kb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length_invariance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outlier_num", type=int, default=0)
    parser.add_argument("--multi_entities", type=int, default=None)
    parser.add_argument("--use_extended_qa", action="store_true")
    parser.add_argument("--kb_token_layer_frequency", type=int, default=3)
    parser.add_argument("--kb_scale_factor", type=float, default=None)
    parser.add_argument("--gradient_accm_step", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--llm_type", type=str, default="qwen_moe")
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--save_period", type=int, default=2000)
    parser.add_argument("--debug_level", type=int, default=0)
    parser.add_argument("--path_attn", action="store_true", default=False)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--base_embeder_path", type=str, default=None)
    parser.add_argument("--disable_random_sample", action="store_true", default=False)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_data_paths", nargs="+", type=str, default=None)
    parser.add_argument("--test_precomputed_embed_keys_path", type=str, default=None)
    parser.add_argument("--test_precomputed_embed_values_path", type=str, default=None)
    parser.add_argument("--test_precomputed_embed_keys_paths", nargs="+", type=str, default=None)
    parser.add_argument("--test_precomputed_embed_values_paths", nargs="+", type=str, default=None)
    parser.add_argument("--test_kb_size", type=int, default=None)
    parser.add_argument("--test_query_size", type=int, default=None)
    parser.add_argument("--test_kb_scale_factor", type=float, default=None)
    parser.add_argument("--test_kb_scale_factor_range", nargs=2, type=float, default=None)
    parser.add_argument("--eval_step", type=int, default=50)
    parser.add_argument("--format_short", type=bool, default=False)
    parser.add_argument("--use_fsdp", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--memory_debug", action="store_true", default=False)
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    args = parser.parse_args()

    if args.dataset_type != "dag":
        raise ValueError("train_qwenlong_dag.py only supports --dataset_type dag")
    if args.llm_type not in {"qwen_moe", "qwen3_moe"}:
        raise ValueError("train_qwenlong_dag.py only supports --llm_type qwen_moe/qwen3_moe")
    if args.use_fsdp and not (
        args.use_cached_embd
        and args.train_precomputed_embed_keys_path is not None
        and args.train_precomputed_embed_values_path is not None
    ):
        raise ValueError(
            "FSDP mode requires --use_cached_embd with both precomputed train embedding paths."
        )

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accm_step)
    set_seed(args.seed + accelerator.process_index)
    distributed_type = str(getattr(accelerator.state, "distributed_type", "UNKNOWN"))

    if accelerator.is_main_process:
        logger.info(f"Accelerator device: {accelerator.device}")
        logger.info(f"Num processes: {accelerator.num_processes}")
        logger.info(f"Distributed type: {distributed_type}")
        if args.use_fsdp and "FSDP" not in distributed_type:
            logger.warning(
                "--use_fsdp was set, but Accelerator is not using FSDP. Launch with an FSDP accelerate config."
            )
        if args.use_fsdp and args.B != 1:
            logger.warning("FSDP path is tuned for --B 1 on 2xL40; current B=%s", args.B)
    if args.memory_debug:
        log_cuda_memory(
            logger,
            label="startup",
            device=accelerator.device,
            accelerator=accelerator,
            force=True,
        )

    training_set = read_json_or_jsonl(args.train_data_path)
    if args.N is not None:
        training_set = training_set[: args.N]

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_spec,
        trust_remote_code=True,
        token=args.hf_token,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"

    model = load_kblam_qwen3_moe_model(
        base_model_dir=args.hf_model_spec,
        checkpoint_dir=args.model_dir_to_resume,
        device=None,
        dtype=torch.bfloat16,
        device_map=None,
    )
    if args.memory_debug:
        log_cuda_memory(
            logger,
            label="after_model_load_before_prepare",
            device=accelerator.device,
            accelerator=accelerator,
            force=True,
        )
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and not args.use_fsdp:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    for _, param in model.named_parameters():
        param.requires_grad = False
    trainable_q_heads = get_q_proj_new_parameters(
        model,
        kb_layer_frequency=args.kb_token_layer_frequency,
    )
    if not trainable_q_heads:
        raise RuntimeError("No q_proj_new parameters found in injected Qwen3-MoE model")
    for param in trainable_q_heads:
        param.requires_grad = True

    encoder = KBEncoder(
        encoder_name=args.encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=_get_kb_encoder_out_dim(model.config, args.kb_token_layer_frequency),
        frozen_base_model=True,
        device=accelerator.device,
    )

    kb_config = KBLaMConfig(
        sep_query_head=args.sep_query_head,
        kb_layer_frequency=args.kb_token_layer_frequency,
        path_attn=args.path_attn,
        kb_scale_factor=args.kb_scale_factor,
        base_model_name_or_path=args.hf_model_spec,
        base_embeder_path=args.base_embeder_path,
    )

    if args.model_dir_to_resume:
        encoder_path = Path(args.model_dir_to_resume + "_encoder") / "encoder.pt"
        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        config_path = Path(args.model_dir_to_resume) / "kb_config_explicit.json"
        if config_path.exists():
            kb_config = KBLaMConfig.from_pretrained(str(config_path))
    if args.freeze_encoder:
        for _, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
    else:
        encoder.train()

    kbretriever = DAGKVKBRetriever(
        encoder=encoder,
        dataset=training_set,
        base_embeder_path=args.base_embeder_path,
        precomputed_embed_keys_path=args.train_precomputed_embed_keys_path,
        precomputed_embed_values_path=args.train_precomputed_embed_values_path,
        max_kv_per_sample=None,
        use_multihop_adj=True,
        max_hops=10,
        hop_decay=1,
        dynamic_hops_by_longest_path=True,
        device=str(accelerator.device),
    )

    trainable_params: List[torch.nn.Parameter] = []
    trainable_params.extend([p for p in model.parameters() if p.requires_grad])
    trainable_params.extend([p for p in encoder.parameters() if p.requires_grad])

    total_optimizer_steps = max(1, args.total_steps // max(1, args.gradient_accm_step))
    if args.warmup_ratio is not None:
        scheduler, optimizer = setup_scheduler_and_optimizer_with_warmup(
            trainable_params,
            lr=args.lr,
            total_optimizer_steps=total_optimizer_steps,
            warmup_ratio=args.warmup_ratio,
        )
    else:
        scheduler, optimizer = setup_scheduler_and_optimizer(
            trainable_params,
            lr=args.lr,
            max_iter=total_optimizer_steps,
        )

    model, optimizer = accelerator.prepare(model, optimizer)
    if args.memory_debug:
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        log_cuda_memory(
            logger,
            label="after_prepare",
            device=accelerator.device,
            accelerator=accelerator,
            force=True,
        )
    kbretriever.encoder = encoder
    prepared_trainable_params = [p for p in model.parameters() if p.requires_grad] + [
        p for p in encoder.parameters() if p.requires_grad
    ]

    output_dir = Path(args.model_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_string = get_prefix_str(args)
    ckpt_name = f"{prefix_string}KeyFrom{args.key_embd_src}_{args.encoder_spec}_{args.dataset_type}_{args.llm_type}"
    start_step = parse_resume_step(args.model_dir_to_resume) if args.resume_steps else 0

    if accelerator.is_main_process:
        logger.info(f"Training dataset size: {len(training_set)}")
        logger.info(f"Starting from step: {start_step}")
        logger.info(f"Checkpoint prefix: {ckpt_name}")
        if args.freeze_encoder:
            logger.info("Encoder is frozen; training q_proj_new only.")
        log_trainable_parameter_summary(logger, accelerator.unwrap_model(model), encoder)

    with create_custom_progress_bar(console=console, disable=not accelerator.is_main_process) as pbar:
        task = pbar.add_task("Training", total=args.total_steps, loss=0.0)
        for step in range(start_step, args.total_steps):
            kb_config.current_step = step
            kb_config.total_steps = args.total_steps
            if args.memory_debug and step == start_step:
                torch.cuda.reset_peak_memory_stats(accelerator.device)
                log_cuda_memory(
                    logger,
                    label=f"step_{step}_start",
                    device=accelerator.device,
                    accelerator=accelerator,
                    force=True,
                )

            input_ids, attention_mask, labels, batch_indices = build_qwen_batch(
                training_set,
                tokenizer,
                accelerator.device,
                batch_size=args.B,
                random_sample=not args.disable_random_sample,
                max_seq_len=args.max_seq_len,
            )
            if args.memory_debug and step == start_step:
                log_cuda_memory(
                    logger,
                    label=f"step_{step}_after_batch",
                    device=accelerator.device,
                    accelerator=accelerator,
                    force=True,
                )
            kb_keys, kb_vals, kb_adj = kbretriever.get_kb_embedding_s(
                batch_indices,
                device=accelerator.device,
            )
            if args.memory_debug and step == start_step:
                logger.info(
                    "[MEM][rank=%s] step_%s_kb_shapes | input=%s kb_keys=%s kb_vals=%s kb_adj_nnz=%s",
                    accelerator.process_index,
                    step,
                    tuple(input_ids.shape),
                    tuple(kb_keys.shape),
                    tuple(kb_vals.shape),
                    int(kb_adj._nnz()),
                )
                log_cuda_memory(
                    logger,
                    label=f"step_{step}_after_kb",
                    device=accelerator.device,
                    accelerator=accelerator,
                    force=True,
                )

            with accelerator.accumulate(model):
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        kb_kvs=(kb_keys, kb_vals),
                        kb_adj=kb_adj,
                        kb_config=kb_config,
                        use_cache=False,
                    )
                except torch.OutOfMemoryError:
                    log_cuda_memory(
                        logger,
                        label=f"step_{step}_oom_during_forward",
                        device=accelerator.device,
                        accelerator=accelerator,
                        force=True,
                    )
                    raise
                if args.memory_debug and step == start_step:
                    log_cuda_memory(
                        logger,
                        label=f"step_{step}_after_forward",
                        device=accelerator.device,
                        accelerator=accelerator,
                        force=True,
                    )
                logits = outputs.logits
                model_config = accelerator.unwrap_model(model).config
                loss = compute_weighted_ce_loss(
                    logits=logits,
                    labels=labels,
                    vocab_size=model_config.vocab_size,
                )
                if args.memory_debug and step == start_step:
                    logger.info(
                        "[MEM][rank=%s] step_%s_loss=%s",
                        accelerator.process_index,
                        step,
                        float(loss.detach().item()),
                    )
                    log_cuda_memory(
                        logger,
                        label=f"step_{step}_before_backward",
                        device=accelerator.device,
                        accelerator=accelerator,
                        force=True,
                    )

                try:
                    accelerator.backward(loss)
                except torch.OutOfMemoryError:
                    log_cuda_memory(
                        logger,
                        label=f"step_{step}_oom_during_backward",
                        device=accelerator.device,
                        accelerator=accelerator,
                        force=True,
                    )
                    raise
                if args.memory_debug and step == start_step:
                    log_cuda_memory(
                        logger,
                        label=f"step_{step}_after_backward",
                        device=accelerator.device,
                        accelerator=accelerator,
                        force=True,
                    )
                if accelerator.sync_gradients:
                    sync_module_gradients(encoder, accelerator.num_processes)
                if args.grad_clip is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prepared_trainable_params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                if args.memory_debug and step == start_step:
                    log_cuda_memory(
                        logger,
                        label=f"step_{step}_after_optimizer",
                        device=accelerator.device,
                        accelerator=accelerator,
                        force=True,
                    )

            loss_value = accelerator.gather(loss.detach()).mean().item()

            if accelerator.is_main_process:
                pbar.update(task, advance=1, loss=loss_value)

            if (
                accelerator.sync_gradients
                and (
                    step == args.total_steps - 1
                    or ((step + 1) % args.save_period == 0)
                )
            ):
                save_training_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    encoder=encoder,
                    kb_config=kb_config,
                    output_dir=output_dir,
                    ckpt_name=ckpt_name,
                    step=step + 1,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training finished.")


if __name__ == "__main__":
    main()
