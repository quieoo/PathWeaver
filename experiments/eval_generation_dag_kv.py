import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from kblam.dag_kv_retriever import DAGKVKBRetriever
from kblam.kb_encoder import KBEncoder
from kblam.metrics_evaluator import full_evaluation, simple_evaluation
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.utils.eval_utils import answer_question_deterministic, format_output_for_synthetic

import re

def _postprocess_generation(raw_output: str, question: str) -> str:
    """
    Keep behavior close to legacy eval_generation.py:
    - remove echoed question if present
    - strip chat/template artifacts
    - keep the final assistant segment
    """
    text = str(raw_output)

    if question and question in text:
        text = text.split(question)[-1]

    text = text.replace("<|begin_of_text|>", "")
    text = text.replace("<|end_of_text|>", "")
    text = text.replace("<|eot_id|>", " ")
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n:,-")

    if not text:
        return text

    # If decoding still contains multiple segments, keep the final answer-like part.
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if parts:
        text = parts[-1]
    return text



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


def prepare_model_and_tokenizer(
    llm_type: str,
    model_path: str,
    base_model_name_or_path: str,
    hf_token: Optional[str] = None,
    query_head_path: Optional[str] = None,
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        token=hf_token if llm_type == "llama3" else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    elif llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported llm_type={llm_type}; choose llama3 or phi3")

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if query_head_path:
        qh = torch.load(query_head_path, map_location=next(model.parameters()).device)
        missing, unexpected = model.load_state_dict(qh, strict=False)
        print(
            f"[eval] loaded query head from {query_head_path}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model


@torch.no_grad()
def evaluate_generation(
    *,
    dataset: Sequence[Dict[str, Any]],
    tokenizer,
    model,
    retriever: DAGKVKBRetriever,
    kb_config: KBLaMConfig,
    max_samples: Optional[int] = None,
    seed: int = 42,
    simple_eval: bool = False,
) -> Dict[str, Any]:
    n = len(dataset)
    if max_samples is None or max_samples <= 0 or max_samples > n:
        eval_indices = list(range(n))
    else:
        rng = random.Random(seed)
        eval_indices = list(range(n))
        rng.shuffle(eval_indices)
        eval_indices = eval_indices[:max_samples]

    preds: List[str] = []
    refs: List[str] = []

    device = next(model.parameters()).device
    for sid in tqdm(eval_indices, desc="Evaluating"):
        sample = dataset[sid]
        q, a = get_qa(sample)
        kb_keys, kb_vals, kb_adj = retriever.get_kb_embedding(sid, device=device)

        if kb_keys.shape[0] == 0:
            continue

        output = answer_question_deterministic(
            tokenizer=tokenizer,
            model=model,
            Q=q,
            kb=(kb_keys, kb_vals),
            kb_adj=kb_adj,
            kb_config=kb_config,
        )
        print(f"Q: {q}, A: {a}, Pred: {output}")
        # pred = format_output_for_synthetic(output)
        pred = format_output_for_synthetic(_postprocess_generation(output, q))
        ref = format_output_for_synthetic(a)
        preds.append(pred)
        refs.append(ref)

    if simple_eval:
        simple_score_dict = simple_evaluation(preds, refs)
        return {
            "num_samples": len(eval_indices),
            "scores": simple_score_dict,
        }
    else:
        gen_report, score_report = full_evaluation(preds, refs)
        return {
            "num_samples": len(eval_indices),
            "scores": score_report,
            "report_text": gen_report,
        }


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate DAG_KV generation")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--base_model_name_or_path", required=True, type=str)
    parser.add_argument("--encoder_path", required=True, type=str)
    parser.add_argument("--base_embeder_path", type=str, default="")
    parser.add_argument("--precomputed_embed_keys_path", type=str, default="")
    parser.add_argument("--precomputed_embed_values_path", type=str, default="")
    parser.add_argument("--encoder_spec", type=str, default="OAI")
    parser.add_argument("--llm_type", type=str, default="llama3", choices=["llama3", "phi3"])
    parser.add_argument("--kb_layer_frequency", type=int, default=3)
    parser.add_argument("--path_attn", action="store_true", default=False)
    parser.add_argument("--path_attn_mix_ratio", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_kv_per_sample", type=int, default=0)
    parser.add_argument("--use_multihop_adj", action="store_true", default=False)
    parser.add_argument("--max_hops", type=int, default=1)
    parser.add_argument("--hop_decay", type=float, default=0.5)
    parser.add_argument("--dynamic_hops_by_longest_path", action="store_true", default=False)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--query_head_path", type=str, default="")
    args = parser.parse_args()

    dataset = read_json_or_jsonl(args.data_path)
    tokenizer, model = prepare_model_and_tokenizer(
        llm_type=args.llm_type,
        model_path=args.model_path,
        base_model_name_or_path=args.base_model_name_or_path,
        hf_token=args.hf_token or None,
        query_head_path=args.query_head_path or None,
    )

    encoder = KBEncoder(
        encoder_name=args.encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size * (model.config.num_hidden_layers // args.kb_layer_frequency + 1),
        frozen_base_model=True,
        device=next(model.parameters()).device,
    )
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=next(model.parameters()).device))
    encoder.eval()

    kb_config = KBLaMConfig(
        kb_layer_frequency=args.kb_layer_frequency,
        path_attn=args.path_attn,
        sep_query_head=True,
    )
    kb_config.path_attn_mix_ratio = args.path_attn_mix_ratio

    retriever = DAGKVKBRetriever(
        encoder=encoder,
        dataset=dataset,
        base_embeder_path=args.base_embeder_path or None,
        precomputed_embed_keys_path=args.precomputed_embed_keys_path or None,
        precomputed_embed_values_path=args.precomputed_embed_values_path or None,
        max_kv_per_sample=args.max_kv_per_sample if args.max_kv_per_sample > 0 else None,
        use_multihop_adj=args.use_multihop_adj,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        dynamic_hops_by_longest_path=args.dynamic_hops_by_longest_path,
        device=str(next(model.parameters()).device),
    )

    result = evaluate_generation(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        retriever=retriever,
        kb_config=kb_config,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        seed=args.seed,
    )

    print(json.dumps(result["scores"], indent=2, ensure_ascii=False))
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scores": result["scores"], "num_samples": result["num_samples"]}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
