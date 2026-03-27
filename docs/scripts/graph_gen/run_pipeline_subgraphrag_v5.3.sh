#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_SCRIPT="${ROOT_DIR}/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5.3.py"
EMBED_SCRIPT="${ROOT_DIR}/docs/scripts/embedding_v2.py"
EXPERIMENT_SCRIPT="${ROOT_DIR}/docs/experiments/train.py"

TRAIN_INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl"
DEV_INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl"

MODEL_CKPT="/mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/subgraphrag_mlp_v5.3.pt"
TRAIN_CACHE_PATH="/mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v5.3.pkl"

DEV_DAG_OUTPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3.jsonl"
TRAIN_DAG_OUTPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3.jsonl"

ST_MODEL="/home/sdu/zhu/models/bge-en-v1.5/"
EMBED_MODEL_NAME="qwen3-embedding-0.6B"
EMBED_LOCAL_MODEL_PATH="/home/sdu/zhu/models/qwen-embedding-0.6B/"
LLM_MODEL_PATH="/home/sdu/zhu/models/llama3_8B_instruct/"

RUN_NAME="${RUN_NAME:-pipeline_subgraphrag_v5.3_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/${RUN_NAME}}"
STEP_LOG_DIR="${LOG_DIR}/steps"
MASTER_LOG="${LOG_DIR}/pipeline.log"
SUMMARY_TSV="${LOG_DIR}/summary.tsv"

mkdir -p "${STEP_LOG_DIR}"

echo -e "step\tstatus\tstarted_at\tended_at\tlog_path" > "${SUMMARY_TSV}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing directory: ${path}" >&2
    exit 1
  fi
}

run_step() {
  local step_name="$1"
  shift

  local safe_name
  safe_name="$(echo "${step_name}" | tr ' /' '__')"
  local step_log="${STEP_LOG_DIR}/${safe_name}.log"
  local started_at ended_at status

  started_at="$(date '+%F %T')"
  status="success"

  {
    echo "===================================================================="
    echo "[STEP] ${step_name}"
    echo "[START] ${started_at}"
    printf "[CMD]"
    printf " %q" "$@"
    printf "\n"
    echo "===================================================================="
    "$@"
  } 2>&1 | tee "${step_log}" | tee -a "${MASTER_LOG}"
  local cmd_status=${PIPESTATUS[0]}

  ended_at="$(date '+%F %T')"
  if [[ ${cmd_status} -ne 0 ]]; then
    status="failed(${cmd_status})"
  fi
  echo -e "${step_name}\t${status}\t${started_at}\t${ended_at}\t${step_log}" >> "${SUMMARY_TSV}"

  if [[ ${cmd_status} -ne 0 ]]; then
    echo "[ERROR] Step failed: ${step_name}" | tee -a "${MASTER_LOG}" >&2
    exit "${cmd_status}"
  fi
}

require_file "${TRAIN_SCRIPT}"
require_file "${EMBED_SCRIPT}"
require_file "${EXPERIMENT_SCRIPT}"
require_file "${TRAIN_INPUT}"
require_file "${DEV_INPUT}"
require_dir "${ST_MODEL}"
require_dir "${EMBED_LOCAL_MODEL_PATH}"
require_dir "${LLM_MODEL_PATH}"

run_step "01_train_debug_verbose" \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --mode train \
  --input "${TRAIN_INPUT}" \
  --model_ckpt "${MODEL_CKPT}" \
  --st_model "${ST_MODEL}" \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 3 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path "${TRAIN_CACHE_PATH}" \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  --rebuild_train_cache \
  --verbose

run_step "02_train_full" \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --mode train \
  --input "${TRAIN_INPUT}" \
  --model_ckpt "${MODEL_CKPT}" \
  --st_model "${ST_MODEL}" \
  --batch_size 512 \
  --train_batch_size 512 \
  --topic_top_k 3 \
  --dde_hops 3 \
  --epochs 100 \
  --lr 3e-4 \
  --neg_pos_ratio 5 \
  --train_cache_path "${TRAIN_CACHE_PATH}" \
  --hidden_dim 768 \
  --patience 10 \
  --joint_training \
  --joint_lambda 0.4 \
  --end_alpha 0.60 \
  --end_beta 0.35 \
  --end_gamma 0.25 \
  --end_threshold 0.3 \
  --rebuild_train_cache

run_step "03_infer_dev" \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --mode infer \
  --input "${DEV_INPUT}" \
  --output "${DEV_DAG_OUTPUT}" \
  --model_ckpt "${MODEL_CKPT}" \
  --st_model "${ST_MODEL}" \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 1000 \
  --keep_score

run_step "04_infer_train" \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --mode infer \
  --input "${TRAIN_INPUT}" \
  --output "${TRAIN_DAG_OUTPUT}" \
  --model_ckpt "${MODEL_CKPT}" \
  --st_model "${ST_MODEL}" \
  --batch_size 512 \
  --infer_batch_size 512 \
  --topic_top_k 8 \
  --dde_hops 3 \
  --mention_bonus 0.2 \
  --seed_edge_topk 18 \
  --expansion_hops 2 \
  --per_src_cap 3 \
  --max_nodes 30 \
  --max_edges 40 \
  --max_sinks 3 \
  --limit 1000 \
  --answer_aware \
  --keep_score \
  --answerable_only

run_step "05_embed_dev_dag" \
  "${PYTHON_BIN}" "${EMBED_SCRIPT}" \
  --model_name "${EMBED_MODEL_NAME}" \
  --local_model_path "${EMBED_LOCAL_MODEL_PATH}" \
  --dataset_type dag \
  --dataset_path "${DEV_DAG_OUTPUT}" \
  --batch_size 1024

run_step "06_embed_train_dag" \
  "${PYTHON_BIN}" "${EMBED_SCRIPT}" \
  --model_name "${EMBED_MODEL_NAME}" \
  --local_model_path "${EMBED_LOCAL_MODEL_PATH}" \
  --dataset_type dag \
  --dataset_path "${TRAIN_DAG_OUTPUT}" \
  --batch_size 1024

run_step "07_train_kblam" \
  "${PYTHON_BIN}" "${EXPERIMENT_SCRIPT}" \
  --seed 1 \
  --B 5 \
  --lr 5e-4 \
  --use_lr_decay \
  --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head \
  --duplicate_true_kb \
  --dynamic_kb_size 10 50 \
  --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B \
  --key_embd_src key \
  --use_cached_embd \
  --train_data_path "${TRAIN_DAG_OUTPUT}" \
  --train_precomputed_embed_keys_path "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3_qwen-embedding-0.6B_embd_key.npy" \
  --train_precomputed_embed_values_path "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train_dag_v5.3_qwen-embedding-0.6B_embd_value.npy" \
  --test_data_path "${DEV_DAG_OUTPUT}" \
  --test_precomputed_embed_keys_path "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3_qwen-embedding-0.6B_embd_key.npy" \
  --test_precomputed_embed_values_path "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev_dag_v5.3_qwen-embedding-0.6B_embd_value.npy" \
  --hf_model_spec "${LLM_MODEL_PATH}" \
  --llm_type llama3 \
  --verbose \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 1 \
  --eval_step 100 \
  --total_steps 8000 \
  --N 9999999 \
  --model_save_dir "${ROOT_DIR}/experiments/train/dag_kv_hotpot_v5.3.1" \
  --save_period 100 \
  --keep_top_k_ckpt 10

echo
echo "Pipeline finished."
echo "Master log: ${MASTER_LOG}"
echo "Summary   : ${SUMMARY_TSV}"
