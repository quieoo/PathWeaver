#!/usr/bin/env python3
"""Knowledge-editable evaluation built on top of legacy eval_generation.py.

This script keeps the original inference pipeline intact and adds round-to-round
accuracy-retention metrics with a fixed round-0 baseline:

1. Run round-0 evaluation once.
2. Evaluate one or more target rounds.
3. For each target round, compare its global accuracy with round-0:
   - new-answer correct ratio (retained accuracy)
   - old-answer leakage ratio
   - other error ratio
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import eval_generation as legacy
from kblam.metrics_evaluator import normalize_text


def load_dataset(dataset_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unknown dataset format: {dataset_path}")


def filter_dataset_by_ids(dataset: list[dict[str, Any]], keep_ids: set[str]) -> list[dict[str, Any]]:
    filtered = []
    for idx, row in enumerate(dataset):
        sample_id = resolve_sample_id(row, idx)
        if sample_id in keep_ids:
            filtered.append(row)
    return filtered


def collect_query_indices_by_ids(dataset: list[dict[str, Any]], keep_ids: set[str] | None) -> list[int] | None:
    if keep_ids is None:
        return None
    out: list[int] = []
    for idx, row in enumerate(dataset):
        if resolve_sample_id(row, idx) in keep_ids:
            out.append(idx)
    return out


def build_retriever(args, dataset, encoder, key_path: str, value_path: str):
    if args.dataset_type == "autoschemakg_2wiki":
        return legacy.AutoSchemaKGKBRetriever(
            encoder,
            dataset,
            precomputed_embed_keys_path=key_path,
            precomputed_embed_values_path=value_path,
        )
    if args.dataset_type == "dag":
        return legacy.DAGKVKBRetriever(
            encoder,
            dataset,
            precomputed_embed_keys_path=key_path,
            precomputed_embed_values_path=value_path,
            max_kv_per_sample=None,
            use_multihop_adj=True,
            max_hops=10,
            hop_decay=1,
            dynamic_hops_by_longest_path=True,
        )
    return legacy.KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=key_path,
        precomputed_embed_values_path=value_path,
        hnsw_index_path=args.hnsw_index_path,
        base_embeder_path=args.base_embeder_path,
    )


def run_eval(
    args,
    dataset,
    tokenizer,
    encoder,
    model,
    kb_config,
    retriever,
    dag_kb_size: int,
    allowed_ids: set[str] | None = None,
):
    query_idx = collect_query_indices_by_ids(dataset, allowed_ids)
    if query_idx is not None and len(query_idx) == 0:
        raise ValueError("No samples left after applying allowed_ids filter.")

    if query_idx is None:
        if args.query_size > len(dataset) or args.query_size == -1:
            query_idx = list(range(len(dataset)))
        else:
            query_idx = list(range(args.query_size))

        if args.seed != 0:
            import numpy as np

            np.random.seed(args.seed)
            query_idx = np.random.randint(0, len(dataset), len(query_idx)).tolist()
    else:
        if args.seed != 0:
            import numpy as np

            np.random.seed(args.seed)
            np.random.shuffle(query_idx)
        if args.query_size != -1:
            query_idx = query_idx[: min(args.query_size, len(query_idx))]

    if args.min_hop is not None:
        filtered_query_idx = []
        for qid in query_idx:
            item = dataset[qid]
            qd = item.get("question_decomposition", {})
            if qd is None:
                break
            if len(qd) < args.min_hop:
                continue
            filtered_query_idx.append(qid)
        if filtered_query_idx:
            query_idx = filtered_query_idx

    if args.kb_scale_factor_range is not None:
        scale_factor_list = []
        start = args.kb_scale_factor_range[0]
        end = args.kb_scale_factor_range[1]
        while start <= end:
            scale_factor_list.append(start)
            start *= 2
    else:
        scale_factor_list = [args.kb_scale_factor]
    print(f"---- kb_scale_factor_range: {scale_factor_list}")

    results_pair_list = []
    for sf in scale_factor_list:
        kb_config.kb_scale_factor = sf
        batch_enable_retrieval = retriever.is_hnsw_ready()
        if args.dataset_type == "all_triples":
            filter_fn = None
            hop_num = None
            use_kb_adj = False
            batch_enable_retrieval = False
        elif "2wiki" in args.dataset_type or "dag" in args.dataset_type:
            filter_fn = None
            hop_num = 2
            use_kb_adj = True
        else:
            filter_fn = None
            hop_num = None
            use_kb_adj = False

        model_outputs, answers, source_indices = legacy._perform_eval_batched(
            model=model,
            tokenizer=tokenizer,
            kb_retriever=retriever,
            kb_config=kb_config,
            dataset=dataset,
            query_idx=query_idx,
            kb_size=args.kb_size,
            dataset_type=args.dataset_type,
            hop_num=hop_num,
            use_kb_adj=use_kb_adj,
            filter_fn=filter_fn,
            enable_retrieval=batch_enable_retrieval,
            enable_silver=("silver" in args.test_dataset),
            output_first_samples=0,
            enable_trace=False,
            trace_dataset_name=args.dataset_type,
            dag_kb_size=dag_kb_size,
        )
        results_pair_list.append((model_outputs, answers, source_indices))
    return results_pair_list, scale_factor_list


def resolve_sample_id(row: dict[str, Any], fallback: int) -> str:
    sample_id = row.get("_id", row.get("id", fallback))
    return str(sample_id)


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return normalize_text(legacy.format_output_for_synthetic(str(text)))


def compute_overall_accuracy(model_outputs: list[str], answers: list[str]) -> dict[str, Any]:
    total = len(model_outputs)
    correct = sum(
        1 for pred, gold in zip(model_outputs, answers) if normalize_answer(pred) == normalize_answer(gold)
    )
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def append_unique_answer(history_map: dict[str, list[str]], sample_id: str, answer: str | None) -> None:
    if answer is None:
        return
    answer = str(answer)
    if sample_id not in history_map:
        history_map[sample_id] = []
    if answer not in history_map[sample_id]:
        history_map[sample_id].append(answer)


def build_history_before_targets(
    round0_dataset: list[dict[str, Any]],
    target_datasets: list[list[dict[str, Any]]],
) -> list[dict[str, list[str]]]:
    history_map: dict[str, list[str]] = {}
    for idx, row in enumerate(round0_dataset):
        append_unique_answer(history_map, resolve_sample_id(row, idx), row.get("answer"))

    snapshots: list[dict[str, list[str]]] = []
    for dataset in target_datasets:
        snapshots.append({sid: list(answers) for sid, answers in history_map.items()})
        for idx, row in enumerate(dataset):
            append_unique_answer(history_map, resolve_sample_id(row, idx), row.get("answer"))
    return snapshots


def build_prediction_map(dataset: list[dict[str, Any]], results_pair) -> dict[str, dict[str, Any]]:
    outputs, answers, source_indices = results_pair
    prediction_map: dict[str, dict[str, Any]] = {}
    for pred, gold, src_idx in zip(outputs, answers, source_indices):
        row = dataset[src_idx]
        sample_id = resolve_sample_id(row, src_idx)
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)
        prediction_map[sample_id] = {
            "sample_id": sample_id,
            "source_index": int(src_idx),
            "prediction": pred,
            "prediction_normalized": pred_norm,
            "gold_answer": gold,
            "gold_answer_normalized": gold_norm,
            "is_correct": pred_norm == gold_norm,
            "row": row,
        }
    return prediction_map


def analyze_round_transition(
    baseline_dataset_name: str,
    baseline_dataset: list[dict[str, Any]],
    baseline_results_pair,
    target_dataset_name: str,
    target_dataset: list[dict[str, Any]],
    target_results_pair,
    history_before_target: dict[str, list[str]] | None = None,
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    baseline_overall = compute_overall_accuracy(baseline_results_pair[0], baseline_results_pair[1])
    target_overall = compute_overall_accuracy(target_results_pair[0], target_results_pair[1])

    prev_map = build_prediction_map(baseline_dataset, baseline_results_pair)
    target_map = build_prediction_map(target_dataset, target_results_pair)
    shared_ids = set(prev_map) & set(target_map)
    if allowed_ids is not None:
        shared_ids &= allowed_ids
    shared_ids = sorted(shared_ids)

    lost_details: list[dict[str, Any]] = []
    loss_counts = {
        "old_answer_leakage": 0,
        "other_error": 0,
    }
    baseline_correct_shared = 0

    for sample_id in shared_ids:
        prev_item = prev_map[sample_id]
        curr_item = target_map[sample_id]
        if not prev_item["is_correct"]:
            continue
        baseline_correct_shared += 1
        if curr_item["is_correct"]:
            continue

        row = curr_item["row"]
        edit_meta = row.get("edit_meta") or {}
        new_answer = normalize_answer(edit_meta.get("new_answer", row.get("answer", curr_item["gold_answer"])))
        historical_answers_raw = list((history_before_target or {}).get(sample_id, []))
        historical_answer_pairs = []
        for answer in historical_answers_raw:
            answer_norm = normalize_answer(answer)
            if answer_norm and answer_norm != new_answer:
                historical_answer_pairs.append((answer, answer_norm))
        historical_answer_norms = {answer_norm for _, answer_norm in historical_answer_pairs}
        pred_norm = curr_item["prediction_normalized"]

        if pred_norm in historical_answer_norms:
            category = "old_answer_leakage"
        else:
            category = "other_error"
        loss_counts[category] += 1

        matched_historical_answers = [
            answer for answer, answer_norm in historical_answer_pairs if answer_norm == pred_norm
        ]
        lost_details.append(
            {
                "sample_id": sample_id,
                "baseline_prediction": prev_item["prediction"],
                "baseline_gold_answer": prev_item["gold_answer"],
                "current_prediction": curr_item["prediction"],
                "current_gold_answer": curr_item["gold_answer"],
                "historical_answers_before_target": historical_answers_raw,
                "matched_historical_answers": matched_historical_answers,
                "category": category,
            }
        )

    prev_acc = baseline_overall["accuracy"]
    curr_acc = target_overall["accuracy"]
    if prev_acc <= 0:
        retention_ratio = 1.0 if curr_acc <= 0 else 0.0
        accuracy_drop_ratio = 0.0
    else:
        retention_ratio = 1.0 if curr_acc >= prev_acc else curr_acc / prev_acc
        accuracy_drop_ratio = 0.0 if curr_acc >= prev_acc else (prev_acc - curr_acc) / prev_acc

    lost_total = loss_counts["old_answer_leakage"] + loss_counts["other_error"]
    if lost_total > 0 and accuracy_drop_ratio > 0:
        old_answer_leakage_ratio = accuracy_drop_ratio * (loss_counts["old_answer_leakage"] / lost_total)
        other_error_ratio = accuracy_drop_ratio * (loss_counts["other_error"] / lost_total)
    else:
        old_answer_leakage_ratio = 0.0
        other_error_ratio = 0.0

    return {
        "baseline_dataset": baseline_dataset_name,
        "target_dataset": target_dataset_name,
        "baseline_overall": baseline_overall,
        "target_overall": target_overall,
        "shared_sample_total": len(shared_ids),
        "baseline_correct_shared_total": baseline_correct_shared,
        "lost_correct_total": lost_total,
        "loss_counts": loss_counts,
        "new_answer_correct_ratio": retention_ratio,
        "accuracy_drop_ratio": accuracy_drop_ratio,
        "old_answer_leakage_ratio": old_answer_leakage_ratio,
        "other_error_ratio": other_error_ratio,
        "lost_details": lost_details,
    }


def save_report(args, report: dict[str, Any], kb_scale_factor: float) -> None:
    if args.save_dir is None:
        return

    save_dir = Path(args.save_dir) / args.exp_config_name
    save_dir.mkdir(exist_ok=True, parents=True)

    sf_tag = str(kb_scale_factor).replace("/", "_")
    summary_path = save_dir / f"{args.exp_config_name}-editable-summary-{sf_tag}.json"
    detail_path = save_dir / f"{args.exp_config_name}-editable-details-{sf_tag}.json"
    summary_payload = {
        key: value
        for key, value in report.items()
        if key not in {"lost_details"}
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "lost_details": report.get("lost_details", []),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Editable results saved to {summary_path} and {detail_path}")


def _resolve_target_args(primary_value: Any, fallback_values: list[Any] | None, expected_len: int, arg_name: str):
    if fallback_values is None:
        if expected_len != 1:
            raise ValueError(f"{arg_name} must provide {expected_len} values.")
        return [primary_value]
    if len(fallback_values) != expected_len:
        raise ValueError(
            f"{arg_name} expects {expected_len} values, got {len(fallback_values)}."
        )
    return list(fallback_values)


parser = argparse.ArgumentParser(description="Knowledge-editable evaluation script")
subparsers = parser.add_subparsers(dest="command", required=True)
gen_parser = subparsers.add_parser(
    "generation",
    parents=[legacy.parent_parser],
    help="Evaluate editable knowledge retention and leakage",
)
gen_parser.add_argument(
    "--exp_config_name",
    type=str,
    default="generation_results",
    help="Name of the experiment configuration",
)
gen_parser.add_argument(
    "--round0_dataset_dir",
    type=str,
    default=None,
    help="Dataset dir for round-0 baseline. Defaults to --dataset_dir.",
)
gen_parser.add_argument(
    "--round0_test_dataset",
    type=str,
    default=None,
    help="Round-0 dataset file. Required when evaluating non-round0 datasets.",
)
gen_parser.add_argument(
    "--round0_precomputed_embed_keys_path",
    type=str,
    default=None,
    help="Round-0 key embedding path.",
)
gen_parser.add_argument(
    "--round0_precomputed_embed_values_path",
    type=str,
    default=None,
    help="Round-0 value embedding path.",
)
gen_parser.add_argument(
    "--round0_dag_kb_size",
    type=int,
    default=None,
    help="Optional DAG KB size for round-0 evaluation. Defaults to --dag_kb_size.",
)
gen_parser.add_argument(
    "--target_datasets",
    nargs="+",
    default=None,
    help="Optional list of target datasets to evaluate in one process.",
)
gen_parser.add_argument(
    "--target_precomputed_embed_keys_paths",
    nargs="+",
    default=None,
    help="Optional list of embedding key paths aligned with --target_datasets.",
)
gen_parser.add_argument(
    "--target_precomputed_embed_values_paths",
    nargs="+",
    default=None,
    help="Optional list of embedding value paths aligned with --target_datasets.",
)
gen_parser.add_argument(
    "--target_dag_kb_sizes",
    nargs="+",
    type=int,
    default=None,
    help="Optional list of DAG KB sizes aligned with --target_datasets.",
)
gen_parser.add_argument(
    "--target_exp_names",
    nargs="+",
    default=None,
    help="Optional list of experiment names aligned with --target_datasets.",
)
gen_parser.add_argument(
    "--use_id_intersection",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="If true, keep only sample IDs that appear in round-0 and all target datasets.",
)


def main():
    args = parser.parse_args()
    if args.command != "generation":
        raise ValueError(f"Unsupported command: {args.command}")

    round0_dataset_dir = args.round0_dataset_dir or args.dataset_dir
    round0_test_dataset = args.round0_test_dataset
    round0_key_path = args.round0_precomputed_embed_keys_path
    round0_value_path = args.round0_precomputed_embed_values_path
    round0_dag_kb_size = args.round0_dag_kb_size or args.dag_kb_size

    if round0_test_dataset is None:
        if args.test_dataset.startswith("round0"):
            round0_test_dataset = args.test_dataset
            round0_key_path = round0_key_path or args.precomputed_embed_keys_path
            round0_value_path = round0_value_path or args.precomputed_embed_values_path
        else:
            raise ValueError(
                "--round0_test_dataset is required when evaluating non-round0 datasets."
            )

    if round0_key_path is None or round0_value_path is None:
        if round0_test_dataset == args.test_dataset:
            round0_key_path = args.precomputed_embed_keys_path
            round0_value_path = args.precomputed_embed_values_path
        else:
            raise ValueError(
                "Round-0 embedding paths are required for non-round0 datasets."
            )

    target_dataset_names = list(args.target_datasets) if args.target_datasets else [args.test_dataset]
    target_key_paths = _resolve_target_args(
        args.precomputed_embed_keys_path,
        args.target_precomputed_embed_keys_paths,
        len(target_dataset_names),
        "--target_precomputed_embed_keys_paths",
    )
    target_value_paths = _resolve_target_args(
        args.precomputed_embed_values_path,
        args.target_precomputed_embed_values_paths,
        len(target_dataset_names),
        "--target_precomputed_embed_values_paths",
    )
    target_dag_kb_sizes = _resolve_target_args(
        args.dag_kb_size,
        args.target_dag_kb_sizes,
        len(target_dataset_names),
        "--target_dag_kb_sizes",
    )
    target_exp_names = _resolve_target_args(
        args.exp_config_name,
        args.target_exp_names,
        len(target_dataset_names),
        "--target_exp_names",
    )

    target_datasets = [load_dataset(args.dataset_dir, dataset_name) for dataset_name in target_dataset_names]
    round0_dataset = load_dataset(round0_dataset_dir, round0_test_dataset)

    intersection_info = None
    keep_ids: set[str] | None = None
    if args.use_id_intersection:
        id_sets = []
        round0_ids = {resolve_sample_id(row, idx) for idx, row in enumerate(round0_dataset)}
        id_sets.append(round0_ids)
        for dataset in target_datasets:
            dataset_ids = {resolve_sample_id(row, idx) for idx, row in enumerate(dataset)}
            id_sets.append(dataset_ids)
        keep_ids = set.intersection(*id_sets) if id_sets else set()
        intersection_info = {
            "enabled": True,
            "kept_sample_total": len(keep_ids),
            "round0_before": len(round0_dataset),
            "targets_before": {
                target_dataset_names[i]: len(target_datasets[i])
                for i in range(len(target_datasets))
            },
        }
        intersection_info["round0_after"] = len(collect_query_indices_by_ids(round0_dataset, keep_ids))
        intersection_info["targets_after"] = {
            target_dataset_names[i]: len(collect_query_indices_by_ids(target_datasets[i], keep_ids))
            for i in range(len(target_datasets))
        }

    tokenizer, encoder, model, kb_config = legacy._prepare_models(
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
    kb_config.current_step = args.step
    kb_config.total_steps = args.t_step
    kb_config.path_attn_mix_ratio = args.path_attn_mix_ratio

    round0_retriever = build_retriever(
        args,
        round0_dataset,
        encoder,
        round0_key_path,
        round0_value_path,
    )

    original_test_dataset = args.test_dataset
    args.test_dataset = round0_test_dataset
    round0_results_pair_list, round0_sf_list = run_eval(
        args,
        round0_dataset,
        tokenizer,
        encoder,
        model,
        kb_config,
        round0_retriever,
        round0_dag_kb_size,
        allowed_ids=keep_ids,
    )
    args.test_dataset = original_test_dataset
    history_before_targets = build_history_before_targets(round0_dataset, target_datasets)

    original_exp_name = args.exp_config_name
    for dataset_idx, target_dataset in enumerate(target_datasets):
        target_name = target_dataset_names[dataset_idx]
        args.test_dataset = target_name
        args.exp_config_name = target_exp_names[dataset_idx]
        target_retriever = build_retriever(
            args,
            target_dataset,
            encoder,
            target_key_paths[dataset_idx],
            target_value_paths[dataset_idx],
        )
        target_results_pair_list, target_sf_list = run_eval(
            args,
            target_dataset,
            tokenizer,
            encoder,
            model,
            kb_config,
            target_retriever,
            target_dag_kb_sizes[dataset_idx],
            allowed_ids=keep_ids,
        )

        if round0_sf_list != target_sf_list:
            raise ValueError(
                f"Mismatched kb_scale_factor lists: {round0_sf_list} vs {target_sf_list}"
            )

        for sf_idx, (sf, round0_pair, target_pair) in enumerate(
            zip(target_sf_list, round0_results_pair_list, target_results_pair_list),
            start=1,
        ):
            editable = analyze_round_transition(
                round0_test_dataset,
                round0_dataset,
                round0_pair,
                target_name,
                target_dataset,
                target_pair,
                history_before_target=history_before_targets[dataset_idx],
                allowed_ids=keep_ids,
            )

            report = {
                "kb_scale_factor": sf,
                "id_intersection": intersection_info,
                **editable,
            }

            intersection_line = (
                f"id intersection kept: {intersection_info['kept_sample_total']}\n"
                if intersection_info is not None
                else ""
            )
            print(
                f"---- target [{dataset_idx + 1}/{len(target_datasets)}]: {target_name} "
                f"scale [{sf_idx}/{len(target_sf_list)}] = {sf}\n"
                f"{intersection_line}"
                f"baseline dataset: {editable['baseline_dataset']}\n"
                f"baseline accuracy: {editable['baseline_overall']['correct']}/{editable['baseline_overall']['total']} = {editable['baseline_overall']['accuracy']:.4f}\n"
                f"target accuracy: {editable['target_overall']['correct']}/{editable['target_overall']['total']} = {editable['target_overall']['accuracy']:.4f}\n"
                f"shared samples: {editable['shared_sample_total']}\n"
                f"baseline-correct shared samples: {editable['baseline_correct_shared_total']}\n"
                f"lost correct samples: {editable['lost_correct_total']}\n"
                f"new-answer correct ratio: {editable['new_answer_correct_ratio']:.4f}\n"
                f"old-answer leakage ratio: {editable['old_answer_leakage_ratio']:.4f}\n"
                f"other error ratio: {editable['other_error_ratio']:.4f}\n"
                f"accuracy drop ratio: {editable['accuracy_drop_ratio']:.4f}"
            )

            save_report(args, report, sf)
    args.exp_config_name = original_exp_name


if __name__ == "__main__":
    main()
