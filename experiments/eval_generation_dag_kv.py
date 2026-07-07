import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from kblam.dag_kv_retriever import DAGKVKBRetriever
from kblam.dag_store_retriever import (
    DAGExtractionConfig,
    TrainableDAGExtractor,
    entity_embedding_model_path,
)
from kblam.kb_encoder import KBEncoder
from kblam.online_dag_kv_retriever import OnlineDAGKBRetriever
from kblam.metrics_evaluator import full_evaluation
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.qwen3.kblam_qwen3_attention import load_kblam_qwen3_model, load_qwen3_query_head
from kblam.utils.eval_utils import answer_question_deterministic, format_output_for_synthetic
from kblam.models.llama3_model import kblam_profile_get, kblam_profile_reset

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
    if llm_type == "qwen3":
        tokenizer.padding_side = "left"
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
    elif llm_type == "qwen3":
        model = load_kblam_qwen3_model(
            base_model_dir=base_model_name_or_path,
            checkpoint_dir=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if query_head_path:
            load_qwen3_query_head(model, query_head_path)
    else:
        raise ValueError(f"Unsupported llm_type={llm_type}; choose llama3 or phi3")

    if llm_type != "qwen3":
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    if query_head_path and llm_type != "qwen3":
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


def get_encoder_out_dim(model_config, kb_layer_frequency: int) -> int:
    slots = model_config.num_hidden_layers // kb_layer_frequency + 1
    head_dim = getattr(model_config, "head_dim", None)
    num_heads = getattr(model_config, "num_attention_heads", None)
    per_slot_dim = head_dim * num_heads if head_dim is not None and num_heads is not None else model_config.hidden_size
    return per_slot_dim * slots


def prepare_online_retriever(args, encoder) -> OnlineDAGKBRetriever:
    if not args.online_dag_model_ckpt:
        raise ValueError("--online_dag_model_ckpt is required with --online_store_dir")
    from sentence_transformers import SentenceTransformer

    st_model = args.online_st_model or entity_embedding_model_path(args.online_store_dir)
    embedder = SentenceTransformer(st_model)
    config = DAGExtractionConfig(
        infer_batch_size=args.online_infer_batch_size,
        topic_top_k=args.online_topic_top_k,
        dde_hops=args.online_dde_hops,
        mention_bonus=args.online_mention_bonus,
        seed_edge_topk=args.online_seed_edge_topk,
        expansion_hops=args.online_expansion_hops,
        per_src_cap=args.online_per_src_cap,
        max_nodes=args.online_max_nodes,
        max_edges=args.online_max_edges,
        max_sinks=args.online_max_sinks,
        answer_aware=False,
        keep_score=True,
        reverse_sink_edge_topk=args.online_reverse_sink_edge_topk,
        reverse_sink_hops=args.online_reverse_sink_hops,
        reverse_sink_beam_width=args.online_reverse_sink_beam_width,
        selection_mode=args.online_selection_mode,
        terminal_reranker=args.online_terminal_reranker,
    )
    extractor = TrainableDAGExtractor(
        args.online_dag_script,
        args.online_dag_model_ckpt,
        embedder,
        config=config,
    )
    return OnlineDAGKBRetriever(
        encoder=encoder,
        store_dir=args.online_store_dir,
        entity_embedder=embedder,
        dag_extractor=extractor,
        entity_top_k=args.online_entity_top_k,
        subgraph_hops=args.online_subgraph_hops,
        search_backend=args.online_search_backend,
        seed_strategy=args.online_seed_strategy,
        mention_min_chars=args.online_mention_min_chars,
        store_version=args.online_store_version,
        entity_candidate_top_k=args.online_entity_candidate_top_k,
        use_multihop_adj=True,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        dynamic_hops_by_longest_path=args.dynamic_hops_by_longest_path,
        require_answer_blind=True,
    )


@torch.no_grad()
def evaluate_generation(
    *,
    dataset: Sequence[Dict[str, Any]],
    tokenizer,
    model,
    retriever: DAGKVKBRetriever | OnlineDAGKBRetriever,
    kb_config: KBLaMConfig,
    max_samples: Optional[int] = None,
    seed: int = 42,
    eval_batch_size: int = 10,
) -> Dict[str, Any]:
    n = len(dataset)
    if max_samples is None or max_samples <= 0 or max_samples > n:
        eval_indices = list(range(n))
    elif seed == 0:
        eval_indices = list(range(max_samples))
    else:
        rng = random.Random(seed)
        eval_indices = list(range(n))
        rng.shuffle(eval_indices)
        eval_indices = eval_indices[:max_samples]

    preds: List[str] = []
    refs: List[str] = []
    questions: List[str] = []
    source_indices: List[int] = []
    model_ttfts: List[float] = []
    tpots: List[float] = []
    retrieval_seconds = 0.0
    empty_kb_samples = 0

    device = next(model.parameters()).device
    eval_started = time.perf_counter()
    for start in tqdm(range(0, len(eval_indices), eval_batch_size), desc="Evaluating"):
        batch_indices = eval_indices[start : start + eval_batch_size]
        batch = [dataset[index] for index in batch_indices]
        batch_questions = [get_qa(sample)[0] for sample in batch]

        if getattr(retriever, "is_online_retriever", False):
            retrieval_started = time.perf_counter()
            online_results = retriever.get_kb_for_queries(batch_questions, device=device)
            retrieval_seconds += time.perf_counter() - retrieval_started
            kb_results = [
                (result.kb_keys, result.kb_values, result.kb_adj)
                for result in online_results
            ]
        else:
            kb_results = [retriever.get_kb_embedding(index, device=device) for index in batch_indices]

        for sid, sample, (kb_keys, kb_vals, kb_adj) in zip(batch_indices, batch, kb_results):
            q, a = get_qa(sample)
            if kb_keys.shape[0] == 0:
                kb = None
                kb_adj = None
                empty_kb_samples += 1
            else:
                kb = (kb_keys, kb_vals)

            kblam_profile_reset()
            generation_started = time.perf_counter()
            output = answer_question_deterministic(
                tokenizer=tokenizer,
                model=model,
                Q=q,
                kb=kb,
                kb_adj=kb_adj,
                kb_config=kb_config,
            )
            generation_elapsed = time.perf_counter() - generation_started
            profile = kblam_profile_get()
            prefill = float(profile.get("prefill_s", 0.0))
            decode = float(profile.get("decode_s", 0.0))
            decode_tokens = max(1, int(profile.get("decode_tokens", 0)))
            model_ttfts.append(prefill if prefill > 0 else generation_elapsed)
            tpots.append(decode / decode_tokens)

            pred = format_output_for_synthetic(_postprocess_generation(output, q))
            ref = format_output_for_synthetic(a)
            print(f"Q: {q}, A: {a}, Pred: {output}")
            preds.append(pred)
            refs.append(ref)
            questions.append(q)
            source_indices.append(sid)

    elapsed = time.perf_counter() - eval_started
    average_retrieval = retrieval_seconds / max(1, len(eval_indices))
    performance = {
        "queries": len(eval_indices),
        "generated_samples": len(preds),
        "empty_kb_samples": empty_kb_samples,
        "qps": len(eval_indices) / max(elapsed, 1e-12),
        "average_latency_seconds": elapsed / max(1, len(eval_indices)),
        "average_model_ttft_seconds": float(np.mean(model_ttfts)),
        "average_retrieval_seconds": average_retrieval,
        "average_end_to_end_ttft_seconds": float(np.mean(model_ttfts)) + average_retrieval,
        "average_tpot_seconds": float(np.mean(tpots)),
    }
    print(json.dumps({"performance": performance}, ensure_ascii=False, indent=2))
    if getattr(retriever, "is_online_retriever", False):
        retriever.print_metrics()

    report_text, scores = full_evaluation(preds, refs, questions=questions)
    return {
        "num_samples": len(eval_indices),
        "scores": scores,
        "report_text": report_text,
        "performance": performance,
        "online_retrieval": retriever.stats() if getattr(retriever, "is_online_retriever", False) else None,
        "predictions": preds,
        "references": refs,
        "questions": questions,
        "source_indices": source_indices,
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
    parser.add_argument("--llm_type", type=str, default="llama3", choices=["llama3", "phi3", "qwen3"])
    parser.add_argument("--kb_layer_frequency", type=int, default=3)
    parser.add_argument("--kb_scale_factor", type=float, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--t_step", type=int, default=1)
    parser.add_argument("--path_attn", action="store_true", default=False)
    parser.add_argument("--path_attn_mix_ratio", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--max_kv_per_sample", type=int, default=0)
    parser.add_argument("--use_multihop_adj", action="store_true", default=False)
    parser.add_argument("--max_hops", type=int, default=1)
    parser.add_argument("--hop_decay", type=float, default=0.5)
    parser.add_argument("--dynamic_hops_by_longest_path", action="store_true", default=False)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--query_head_path", type=str, default="")
    parser.add_argument("--online_store_dir", type=str, default="")
    parser.add_argument(
        "--online_dag_script",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v8_infer_only.py"
        ),
    )
    parser.add_argument("--online_dag_model_ckpt", type=str, default="")
    parser.add_argument("--online_st_model", type=str, default="")
    parser.add_argument("--online_entity_top_k", type=int, default=1)
    parser.add_argument("--online_entity_candidate_top_k", type=int, default=64)
    parser.add_argument("--online_subgraph_hops", type=int, default=2)
    parser.add_argument("--online_store_version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--online_search_backend", choices=["hnsw", "exact"], default="hnsw")
    parser.add_argument("--online_seed_strategy", choices=["vector", "hybrid"], default="vector")
    parser.add_argument("--online_mention_min_chars", type=int, default=8)
    parser.add_argument("--online_infer_batch_size", type=int, default=1024)
    parser.add_argument("--online_topic_top_k", type=int, default=8)
    parser.add_argument("--online_dde_hops", type=int, default=3)
    parser.add_argument("--online_mention_bonus", type=float, default=0.2)
    parser.add_argument("--online_seed_edge_topk", type=int, default=18)
    parser.add_argument("--online_expansion_hops", type=int, default=2)
    parser.add_argument("--online_per_src_cap", type=int, default=3)
    parser.add_argument("--online_max_nodes", type=int, default=30)
    parser.add_argument("--online_max_edges", type=int, default=40)
    parser.add_argument("--online_max_sinks", type=int, default=3)
    parser.add_argument("--online_reverse_sink_edge_topk", type=int, default=2)
    parser.add_argument("--online_reverse_sink_hops", type=int, default=4)
    parser.add_argument("--online_reverse_sink_beam_width", type=int, default=4)
    parser.add_argument("--online_selection_mode", choices=["legacy"], default="legacy")
    parser.add_argument("--online_terminal_reranker", choices=["joint", "heuristic"], default="joint")
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
        out_dim=get_encoder_out_dim(model.config, args.kb_layer_frequency),
        frozen_base_model=True,
        device=next(model.parameters()).device,
    )
    encoder.load_state_dict(
        torch.load(args.encoder_path, map_location=next(model.parameters()).device, weights_only=True)
    )
    encoder.eval()

    kb_config = KBLaMConfig(
        kb_layer_frequency=args.kb_layer_frequency,
        kb_scale_factor=args.kb_scale_factor,
        path_attn=args.path_attn,
        sep_query_head=True,
        current_step=args.step,
        total_steps=args.t_step,
    )
    kb_config.path_attn_mix_ratio = args.path_attn_mix_ratio

    if args.online_store_dir:
        retriever = prepare_online_retriever(args, encoder)
    else:
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
        eval_batch_size=args.eval_batch_size,
    )

    print(json.dumps(result["scores"], indent=2, ensure_ascii=False))
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in result.items() if key != "report_text"}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if getattr(retriever, "is_online_retriever", False):
        retriever.close()


if __name__ == "__main__":
    main()
