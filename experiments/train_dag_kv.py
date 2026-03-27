import argparse
import gc
import heapq
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from kblam.dag_kv_retriever import DAGKVKBRetriever
from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.utils.train_utils import (
    setup_scheduler_and_optimizer,
    setup_scheduler_and_optimizer_with_warmup,
)
from kblam.utils.eval_utils import (
    format_QA_llama,
    format_QA_phi3,
)

try:
    from eval_generation_dag_kv import evaluate_generation as synced_evaluate_generation
except ImportError:
    from docs.experiments.eval_generation_dag_kv import evaluate_generation as synced_evaluate_generation
import tqdm



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


def get_qa(sample: Dict[str, Any]) -> Tuple[str, str]:
    q = sample.get("question", sample.get("Q", ""))
    a = sample.get("answer", sample.get("A", ""))
    return str(q), str(a)


def find_subsequence(seq: List[int], pattern: List[int]) -> int:
    if not pattern or len(pattern) > len(seq):
        return -1
    end = len(seq) - len(pattern) + 1
    for i in range(end):
        if seq[i : i + len(pattern)] == pattern:
            return i
    return -1


def create_labels_for_llama(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    marker = tokenizer(
        "<|start_header_id|>assistant<|end_header_id|>",
        add_special_tokens=False,
    ).input_ids
    for b in range(input_ids.size(0)):
        seq = input_ids[b].tolist()
        pos = find_subsequence(seq, marker)
        if pos < 0:
            continue
        start = pos + len(marker)
        valid_len = int(attention_mask[b].sum().item())
        labels[b, start:valid_len] = input_ids[b, start:valid_len]
    return labels


def create_labels_for_phi3(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    marker = tokenizer("<|assistant|>\n", add_special_tokens=False).input_ids
    if not marker:
        marker = tokenizer("<|assistant|>", add_special_tokens=False).input_ids
    for b in range(input_ids.size(0)):
        seq = input_ids[b].tolist()
        pos = find_subsequence(seq, marker)
        if pos < 0:
            continue
        start = pos + len(marker)
        valid_len = int(attention_mask[b].sum().item())
        labels[b, start:valid_len] = input_ids[b, start:valid_len]
    return labels


def build_train_batch(
    dataset: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    tokenizer,
    llm_type: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompts: List[str] = []
    for idx in indices:
        q, a = get_qa(dataset[idx])
        if llm_type == "llama3":
            prompts.append(format_QA_llama(q, a))
        elif llm_type == "phi3":
            prompts.append(format_QA_phi3(q, a))
        else:
            raise ValueError(f"Unsupported llm_type={llm_type}")

    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    if llm_type == "llama3":
        labels = create_labels_for_llama(input_ids, attention_mask, tokenizer)
    else:
        labels = create_labels_for_phi3(input_ids, attention_mask, tokenizer)
    return input_ids, attention_mask, labels


def prepare_model_and_tokenizer(
    *,
    llm_type: str,
    model_name_or_path: str,
    hf_token: Optional[str],
    device: torch.device,
    gradient_checkpointing: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        token=hf_token if llm_type == "llama3" else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda" if device.type == "cuda" else None,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
    elif llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda" if device.type == "cuda" else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported llm_type={llm_type}; choose llama3 or phi3")

    model.to(device)
    # Keep behavior aligned with legacy train.py
    # Some transformers/kblam combinations expect this helper to exist when
    # gradient checkpointing path is active.
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        # For checkpointing + partially frozen models, at least one checkpoint input
        # needs requires_grad=True, otherwise loss can become a leaf without grad_fn.
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        if hasattr(model, "model"):
            try:
                if hasattr(model.model, "gradient_checkpointing_enable"):
                    model.model.gradient_checkpointing_enable()
            except Exception:
                pass
            try:
                model.model.gradient_checkpointing = True
            except Exception:
                pass
    else:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        if hasattr(model, "model"):
            try:
                model.model.gradient_checkpointing = False
            except Exception:
                pass
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def compute_weighted_ce_loss_like_train_py(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fct: CrossEntropyLoss,
    vocab_size: int,
) -> torch.Tensor:
    """
    Match docs/experiments/train.py behavior:
      weighted token CE so each sample contributes similarly.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    valid_mask = shift_labels != -100
    weights = valid_mask.sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    weights = weights.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)

    loss = (loss_fct(shift_logits, shift_labels) * weights.max() / weights).mean()
    return loss


def get_llama3_query_head_parameters(
    model: KblamLlamaForCausalLM,
    kb_token_layer_frequency: int,
) -> List[torch.nn.Parameter]:
    llm_q_params: List[torch.nn.Parameter] = []
    old_weight: Optional[torch.Tensor] = None
    for name, param in model.named_parameters():
        if "q_proj.weight" in name:
            m = re.search(r"\d+", name)
            if m is not None:
                layer_id = int(m.group(0))
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
        if "q_proj_new.weight" in name:
            m = re.search(r"\d+", name)
            if m is None:
                continue
            layer_id = int(m.group(0))
            if layer_id % kb_token_layer_frequency != 0:
                continue
            if old_weight is not None and old_weight.shape == param.shape:
                with torch.no_grad():
                    param.copy_(old_weight)
            param.requires_grad = True
            llm_q_params.append(param)
    return llm_q_params


def get_phi3_query_head_parameters(
    model: KBLaMPhi3ForCausalLM,
    kb_token_layer_frequency: int,
) -> List[torch.nn.Parameter]:
    llm_q_params: List[torch.nn.Parameter] = []
    old_weight: Optional[torch.Tensor] = None
    for name, param in model.named_parameters():
        if "qkv_proj.weight" in name:
            m = re.search(r"\d+", name)
            if m is not None:
                layer_id = int(m.group(0))
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
        if "q_proj_new.weight" in name:
            m = re.search(r"\d+", name)
            if m is None:
                continue
            layer_id = int(m.group(0))
            if layer_id % kb_token_layer_frequency != 0:
                continue
            if old_weight is not None:
                with torch.no_grad():
                    param.copy_(old_weight[: model.config.hidden_size, :])
            param.requires_grad = True
            llm_q_params.append(param)
    return llm_q_params


def safe_evaluate_wrapper(
    *,
    dataset: Sequence[Dict[str, Any]],
    tokenizer,
    model,
    retriever: DAGKVKBRetriever,
    kb_config: KBLaMConfig,
    max_samples: Optional[int],
    seed: int,
) -> Optional[Dict[str, float]]:
    was_training_model = model.training
    was_training_encoder = retriever.encoder.training
    model_grad_state = torch.is_grad_enabled()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        torch.set_grad_enabled(False)
        model.eval()
        retriever.encoder.eval()

        result = synced_evaluate_generation(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            retriever=retriever,
            kb_config=kb_config,
            max_samples=max_samples,
            seed=seed,
            simple_eval=True,
        )
        scores = result.get("scores", {})
        normalized: Dict[str, float] = {}
        for k, v in scores.items():
            try:
                normalized[k] = float(v)
            except (TypeError, ValueError):
                continue
        return normalized
    except Exception as e:
        print(f"[safe-eval] ERROR: {type(e).__name__}: {e}")
        return None
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.set_grad_enabled(model_grad_state)

        if was_training_model:
            model.train()
        else:
            model.eval()
        if was_training_encoder:
            retriever.encoder.train()
        else:
            retriever.encoder.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@dataclass
class TrainState:
    step: int = 0
    best_eval_score: float = -1.0


def save_checkpoint(
    *,
    out_dir: Path,
    model,
    encoder: KBEncoder,
    kb_config: KBLaMConfig,
    state: TrainState,
    save_full_model: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_dir / "encoder.pt")
    qh_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "q_proj_new" in name
    }
    if qh_state:
        torch.save(qh_state, out_dir / "query_head.pt")
    (out_dir / "kb_config_explicit.json").write_text(kb_config.to_json_string(), encoding="utf-8")
    (out_dir / "train_state.json").write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    if save_full_model:
        model.save_pretrained(out_dir / "model")


def main() -> None:
    parser = argparse.ArgumentParser("Train KBLaM adapter on DAG_KV dataset")
    parser.add_argument("--train_data_path", required=True, type=str)
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output_dag_kv")

    parser.add_argument("--llm_type", type=str, default="llama3", choices=["llama3", "phi3"])
    parser.add_argument("--hf_model_spec", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--encoder_spec", type=str, default="OAI")
    parser.add_argument("--base_embeder_path", type=str, default="")
    parser.add_argument("--train_precomputed_embed_keys_path", type=str, default="")
    parser.add_argument("--train_precomputed_embed_values_path", type=str, default="")
    parser.add_argument("--val_precomputed_embed_keys_path", type=str, default="")
    parser.add_argument("--val_precomputed_embed_values_path", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_lr_decay", action="store_true", default=False)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=2000)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--keep_top_k_ckpt", type=int, default=1)
    parser.add_argument("--debug_print_every", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--kb_layer_frequency", type=int, default=3)
    parser.add_argument("--sep_query_head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path_attn", action="store_true", default=False)
    parser.add_argument("--path_attn_mix_ratio", type=float, default=1.0)

    parser.add_argument("--max_kv_per_sample", type=int, default=0)
    parser.add_argument("--use_multihop_adj", action="store_true", default=False)
    parser.add_argument("--max_hops", type=int, default=1)
    parser.add_argument("--hop_decay", type=float, default=0.5)
    parser.add_argument("--dynamic_hops_by_longest_path", action="store_true", default=False)

    parser.add_argument("--save_full_model", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = read_json_or_jsonl(args.train_data_path)
    val_data = read_json_or_jsonl(args.val_data_path) if args.val_data_path else train_data

    tokenizer, model = prepare_model_and_tokenizer(
        llm_type=args.llm_type,
        model_name_or_path=args.hf_model_spec,
        hf_token=args.hf_token or os.getenv("HF_TOKEN"),
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    encoder = KBEncoder(
        encoder_name=args.encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size * (model.config.num_hidden_layers // args.kb_layer_frequency + 1),
        frozen_base_model=True,
        device=device,
    )
    encoder.train()

    kb_config = KBLaMConfig(
        kb_layer_frequency=args.kb_layer_frequency,
        sep_query_head=args.sep_query_head,
        path_attn=args.path_attn,
    )
    kb_config.path_attn_mix_ratio = args.path_attn_mix_ratio

    train_retriever = DAGKVKBRetriever(
        encoder=encoder,
        dataset=train_data,
        base_embeder_path=args.base_embeder_path or None,
        precomputed_embed_keys_path=args.train_precomputed_embed_keys_path or None,
        precomputed_embed_values_path=args.train_precomputed_embed_values_path or None,
        max_kv_per_sample=args.max_kv_per_sample if args.max_kv_per_sample > 0 else None,
        use_multihop_adj=args.use_multihop_adj,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        dynamic_hops_by_longest_path=args.dynamic_hops_by_longest_path,
        device=str(device),
    )

    val_key_path = args.val_precomputed_embed_keys_path or args.train_precomputed_embed_keys_path
    val_val_path = args.val_precomputed_embed_values_path or args.train_precomputed_embed_values_path
    val_retriever = DAGKVKBRetriever(
        encoder=encoder,
        dataset=val_data,
        base_embeder_path=args.base_embeder_path or None,
        precomputed_embed_keys_path=val_key_path or None,
        precomputed_embed_values_path=val_val_path or None,
        max_kv_per_sample=args.max_kv_per_sample if args.max_kv_per_sample > 0 else None,
        use_multihop_adj=args.use_multihop_adj,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        dynamic_hops_by_longest_path=args.dynamic_hops_by_longest_path,
        device=str(device),
    )

    llm_q_params: List[torch.nn.Parameter] = []
    if args.sep_query_head:
        if args.llm_type == "llama3":
            llm_q_params = get_llama3_query_head_parameters(model, args.kb_layer_frequency)
        elif args.llm_type == "phi3":
            llm_q_params = get_phi3_query_head_parameters(model, args.kb_layer_frequency)
        print(f"[train] sep_query_head enabled, trainable query-head params={len(llm_q_params)}")
    trainable_params = list(chain(encoder.parameters(), llm_q_params)) if llm_q_params else list(encoder.parameters())
    if args.use_lr_decay:
        if args.warmup_ratio is not None:
            scheduler, optim = setup_scheduler_and_optimizer_with_warmup(
                trainable_params,
                args.lr,
                args.total_steps,
                args.warmup_ratio,
            )
        else:
            scheduler, optim = setup_scheduler_and_optimizer(
                trainable_params,
                args.lr,
                args.total_steps,
            )
    else:
        scheduler = None
        optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    state = TrainState()
    loss_fct = CrossEntropyLoss(reduction="none")
    topk_checkpoints: List[Tuple[float, int, Path]] = []  # min-heap: (score, step, dir)

    train_indices = np.arange(len(train_data))
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    for step in (range(args.total_steps)):
        state.step = step
        kb_config.current_step = step
        kb_config.total_steps = args.total_steps

        optim.zero_grad(set_to_none=True)
        micro_losses: List[float] = []
        debug_gt: Optional[str] = None
        debug_pred: Optional[str] = None
        debug_kb_shape: Optional[Tuple[int, ...]] = None
        for a_step in range(args.grad_accum_steps):
            batch_idx = np.random.choice(train_indices, size=min(args.batch_size, len(train_indices)), replace=False)
            input_ids, attention_mask, labels = build_train_batch(
                dataset=train_data,
                indices=batch_idx.tolist(),
                tokenizer=tokenizer,
                llm_type=args.llm_type,
                device=device,
            )
            if args.max_seq_len and args.max_seq_len > 0:
                input_ids = input_ids[:, : args.max_seq_len]
                attention_mask = attention_mask[:, : args.max_seq_len]
                labels = labels[:, : args.max_seq_len]

            kb_keys, kb_vals, kb_adj = train_retriever.get_kb_embedding_s(batch_idx.tolist(), device=device)

            if kb_keys.size(0) == 0:
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kb_kvs=(kb_keys, kb_vals),
                kb_adj=kb_adj,
                kb_config=kb_config,
                use_cache=False,
                output_attentions=False,
            )
            micro_loss = compute_weighted_ce_loss_like_train_py(
                logits=outputs.logits,
                labels=labels,
                loss_fct=loss_fct,
                vocab_size=model.config.vocab_size,
            )
            # print(f"[train] step={step}/{args.total_steps} a_step={a_step}/{args.grad_accum_steps} micro_loss={float(micro_loss.item()):.6f}")
            (micro_loss / args.grad_accum_steps).backward()
            micro_losses.append(float(micro_loss.item()))

            # Debug compare like legacy train.py
            if a_step==0 and args.debug_print_every > 0 and step % args.debug_print_every == 0 and debug_gt is None:
                with torch.no_grad():
                    logits = outputs.logits
                    batch_index = 0
                    # Causal LM: logits[t] predicts label[t+1], so shift before debug decode.
                    shift_logits = logits[batch_index, :-1, :]
                    shift_labels = labels[batch_index, 1:]
                    valid_pos = shift_labels != -100
                    pred_ids = shift_logits.argmax(dim=-1)[valid_pos]
                    gt_ids = shift_labels[valid_pos]
                    debug_pred = tokenizer.decode(pred_ids, skip_special_tokens=False)
                    debug_gt = tokenizer.decode(gt_ids, skip_special_tokens=False)
                    debug_kb_shape = tuple(kb_keys.shape)

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        optim.zero_grad(set_to_none=True)

        if step % 1 == 0:
            lr_now = optim.param_groups[0]["lr"]
            avg_micro_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
            print(
                f"[train] step={step}/{args.total_steps} "
                f"micro_loss(avg)={avg_micro_loss:.6f} "
                f"lr={lr_now:.6e}"
            )
        if args.debug_print_every > 0 and step % args.debug_print_every == 0:
            if debug_gt is not None and debug_pred is not None:
                print("-" * 80)
                print(f"[debug] kb_shape={debug_kb_shape}")
                print(f"[debug] GT: {debug_gt}")
                print(f"[debug] PRED: {debug_pred}")
                print("-" * 80)

        should_eval = (step + 1) % args.eval_every == 0 or step == args.total_steps - 1
        if should_eval:
            scores = safe_evaluate_wrapper(
                dataset=val_data,
                tokenizer=tokenizer,
                model=model,
                retriever=val_retriever,
                kb_config=kb_config,
                max_samples=args.eval_samples if args.eval_samples > 0 else None,
                seed=args.seed,
            )
            if scores is not None:
                metric = scores.get("f1", scores.get("exact_match", 0.0))
                print(f"[eval] step={step} scores={scores}")
                if metric > state.best_eval_score:
                    state.best_eval_score = metric
                should_save = (step + 1) % args.save_every == 0 or step == args.total_steps - 1
                if should_save and args.keep_top_k_ckpt > 0:
                    score = float(metric)
                    candidate_dir = out_dir / f"topk_step_{step + 1}_score_{score:.4f}"
                    should_keep = (
                        len(topk_checkpoints) < args.keep_top_k_ckpt
                        or score > topk_checkpoints[0][0]
                    )
                    if should_keep:
                        save_checkpoint(
                            out_dir=candidate_dir,
                            model=model,
                            encoder=encoder,
                            kb_config=kb_config,
                            state=state,
                            save_full_model=args.save_full_model,
                        )
                        heapq.heappush(topk_checkpoints, (score, step + 1, candidate_dir))
                        print(f"[top-k] added step={step + 1} score={score:.4f} path={candidate_dir}")
                        if len(topk_checkpoints) > args.keep_top_k_ckpt:
                            worst_score, worst_step, worst_dir = heapq.heappop(topk_checkpoints)
                            if worst_dir.exists():
                                shutil.rmtree(worst_dir, ignore_errors=True)
                            print(f"[top-k] removed step={worst_step} score={worst_score:.4f} path={worst_dir}")

    if topk_checkpoints:
        print(f"[top-k] final kept checkpoints (k={args.keep_top_k_ckpt}):")
        for score, ckpt_step, ckpt_dir in sorted(topk_checkpoints, key=lambda x: x[0], reverse=True):
            print(f"  step={ckpt_step}, score={score:.4f}, path={ckpt_dir}")
    print(f"[done] training completed. best_eval_score={state.best_eval_score:.6f}")


if __name__ == "__main__":
    main()
