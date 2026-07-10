#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/n0/PathWeaver}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/n0/uv_envs/kblam/bin/python}"
DATA_PATH="${DATA_PATH:-/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl}"
MODEL_DIR="${MODEL_DIR:-/mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800}"
BASE_MODEL="${BASE_MODEL:-/mnt/n0/models/qwen3-14B-Instruct}"
ENCODER_PATH="${ENCODER_PATH:-/mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_14B_aa_B2.1/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_15800_encoder/encoder.pt}"
STORE_DIR="${STORE_DIR:-experiments/stores/scale-sweep-20260706/training-tiers-v2/064000-with-train-store-v2-subject-only}"
DAG_SCRIPT="${DAG_SCRIPT:-docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v8_infer_only.py}"
DAG_CKPT="${DAG_CKPT:-/mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v2.1.pt}"
ST_MODEL="${ST_MODEL:-/mnt/n0/models/bge-en-v1.5/}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/online_dag_eval/cap_sweep_64k_subject_only}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CAP_LIST="${CAP_LIST:-none 64 48 32 24 16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
cd "${ROOT_DIR}"

for cap in ${CAP_LIST}; do
  output_json="${OUTPUT_DIR}/online_store_v2_64k_subject_only_cap_${cap}_top1_hop2_sink3_heuristic_${MAX_SAMPLES}.json"
  cmd=(
    "${PYTHON_BIN}" experiments/eval_generation_dag_kv.py
    --data_path "${DATA_PATH}"
    --model_path "${MODEL_DIR}"
    --base_model_name_or_path "${BASE_MODEL}"
    --encoder_path "${ENCODER_PATH}"
    --encoder_spec qwen-embedding-0.6B
    --llm_type qwen3
    --kb_layer_frequency 3
    --kb_scale_factor 4
    --path_attn
    --path_attn_mix_ratio 0.8
    --step 7999
    --t_step 8000
    --max_samples "${MAX_SAMPLES}"
    --seed 0
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --online_store_dir "${STORE_DIR}"
    --online_store_version v2
    --online_dag_script "${DAG_SCRIPT}"
    --online_dag_model_ckpt "${DAG_CKPT}"
    --online_st_model "${ST_MODEL}"
    --online_entity_top_k 1
    --online_entity_candidate_top_k 64
    --online_subgraph_hops 2
    --online_search_backend hnsw
    --online_seed_strategy hybrid
    --online_mention_min_chars 8
    --online_infer_batch_size 1024
    --online_topic_top_k 8
    --online_dde_hops 3
    --online_mention_bonus 0.2
    --online_seed_edge_topk 18
    --online_expansion_hops 2
    --online_per_src_cap 3
    --online_max_nodes 30
    --online_max_edges 40
    --online_max_sinks 3
    --online_reverse_sink_edge_topk 2
    --online_reverse_sink_hops 4
    --online_reverse_sink_beam_width 4
    --online_selection_mode legacy
    --online_terminal_reranker heuristic
    --use_multihop_adj
    --max_hops 10
    --hop_decay 1.0
    --dynamic_hops_by_longest_path
    --save_json "${output_json}"
  )
  if [[ "${cap}" != "none" ]]; then
    cmd+=(--online_max_incident_triples_per_node "${cap}")
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_parts=(${EXTRA_ARGS})
    cmd+=("${extra_parts[@]}")
  fi
  echo "[RUN] cap=${cap} -> ${output_json}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH=src "${cmd[@]}"
done
