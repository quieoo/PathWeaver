#!/usr/bin/env bash
set -euo pipefail

# =========================================
# User-configurable paths
# =========================================
TRAIN_INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_train.jsonl"
DEV_INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl"
SCRIPT="DAG_KV_SubgraphRAG_trainable_v3.py"
PYTHON_BIN="python"
ST_MODEL="/home/sdu/zhu/models/bge-en-v1.5/"

OUTDIR="/mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/sweep_train_infer_v3"
CACHE_PATH="/mnt/n0/KBLAM/KBLaM/experiments/subgraph_mlp/training_data_cache_v2.pkl"

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/checkpoints"
mkdir -p "${OUTDIR}/train_logs"
mkdir -p "${OUTDIR}/infer_logs"
mkdir -p "${OUTDIR}/infer_outputs"

SUMMARY_TSV="${OUTDIR}/summary.tsv"

# =========================================
# Fixed training args
# =========================================
BATCH_SIZE=${BATCH_SIZE:-512}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
EPOCHS=${EPOCHS:-100}
DDE_HOPS=${DDE_HOPS:-3}
TRAIN_LIMIT=${TRAIN_LIMIT:-""}     # e.g. 1000 ; empty means full train set

# =========================================
# Fixed inference args
# =========================================
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-4096}
MAX_NODES=${MAX_NODES:-30}
MAX_EDGES=${MAX_EDGES:-40}
MAX_SINKS=${MAX_SINKS:-8}
DEV_LIMIT=${DEV_LIMIT:-100}        # keep your current quick setting; set empty for full dev

# =========================================
# Sweep dimensions: training
# =========================================
HIDDEN_DIM_LIST=(768)
LR_LIST=(5e-4)
NEG_POS_RATIO_LIST=(6)
HARD_NEG_RATIO_LIST=(0.1 0.15 0.2 0.25 0.3 0.35)
HARD_NEG_MIN=1

# =========================================
# Sweep dimensions: inference
# =========================================
SEED_EDGE_TOPK_LIST=(14 18)
PER_SRC_CAP_LIST=(2 3)
TOPIC_TOP_K_LIST=(4 6 8)

# =========================================
# Summary header
# =========================================

echo -e "train_id\tinfer_id\thidden_dim\tlr\tneg_pos_ratio\thard_neg_ratio\tseed_edge_topk\tper_src_cap\ttopic_top_k\tbest_dev_f1\tanswer_recall\tgraph_recall\tnone_sink_recall\tckpt\toutput_jsonl\ttrain_log\tinfer_log" > "${SUMMARY_TSV}"


train_run_id=0

for HIDDEN_DIM in "${HIDDEN_DIM_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for NEG_POS_RATIO in "${NEG_POS_RATIO_LIST[@]}"; do
      for HARD_NEG_RATIO in "${HARD_NEG_RATIO_LIST[@]}"; do
        train_run_id=$((train_run_id + 1))

        TRAIN_TAG=$(printf "train_%02d_hd%s_lr%s_neg%s_hardneg%s" \
          "${train_run_id}" "${HIDDEN_DIM}" "${LR}" "${NEG_POS_RATIO}" "${HARD_NEG_RATIO}")

        CKPT_PATH="${OUTDIR}/checkpoints/${TRAIN_TAG}.pt"
        TRAIN_LOG="${OUTDIR}/train_logs/${TRAIN_TAG}.log"

        echo "===================================================================="
        echo "[TRAIN ${train_run_id}] ${TRAIN_TAG}"
        echo "CKPT: ${CKPT_PATH}"
        echo "LOG : ${TRAIN_LOG}"
        echo "===================================================================="

        TRAIN_CMD=(
          "${PYTHON_BIN}" "${SCRIPT}"
          --mode train
          --input "${TRAIN_INPUT}"
          --model_ckpt "${CKPT_PATH}"
          --st_model "${ST_MODEL}"
          --batch_size "${BATCH_SIZE}"
          --train_batch_size "${TRAIN_BATCH_SIZE}"
          --topic_top_k 6
          --dde_hops "${DDE_HOPS}"
          --epochs "${EPOCHS}"
          --lr "${LR}"
          --neg_pos_ratio "${NEG_POS_RATIO}"
          --train_cache_path "${CACHE_PATH}"
          --hidden_dim "${HIDDEN_DIM}"
          --hard_neg_ratio "${HARD_NEG_RATIO}"
          --hard_neg_min "${HARD_NEG_MIN}"
          --rebuild_train_cache
        )

        if [[ -n "${TRAIN_LIMIT}" ]]; then
          TRAIN_CMD+=(--limit "${TRAIN_LIMIT}")
        fi

        {
          printf "COMMAND:"
          printf " %q" "${TRAIN_CMD[@]}"
          printf "\n\n"
          "${TRAIN_CMD[@]}"
        } 2>&1 | tee "${TRAIN_LOG}"

        BEST_DEV_F1=$(python - <<PY
import re, sys
text = open("${TRAIN_LOG}", "r", encoding="utf-8").read()
m = re.findall(r'"best_dev_f1"\s*:\s*([0-9.]+)', text)
print(m[-1] if m else "")
PY
)

        infer_run_id=0

        for SEED_EDGE_TOPK in "${SEED_EDGE_TOPK_LIST[@]}"; do
          for PER_SRC_CAP in "${PER_SRC_CAP_LIST[@]}"; do
            for TOPIC_TOP_K in "${TOPIC_TOP_K_LIST[@]}"; do
              infer_run_id=$((infer_run_id + 1))

              INFER_TAG=$(printf "%s__infer_%02d_topk%s_cap%s_topic%s" \
                "${TRAIN_TAG}" "${infer_run_id}" "${SEED_EDGE_TOPK}" "${PER_SRC_CAP}" "${TOPIC_TOP_K}")

              OUTPUT_JSONL="${OUTDIR}/infer_outputs/${INFER_TAG}.jsonl"
              INFER_LOG="${OUTDIR}/infer_logs/${INFER_TAG}.log"

              echo "--------------------------------------------------------------------"
              echo "[INFER ${train_run_id}.${infer_run_id}] ${INFER_TAG}"
              echo "OUT : ${OUTPUT_JSONL}"
              echo "LOG : ${INFER_LOG}"
              echo "--------------------------------------------------------------------"

              INFER_CMD=(
                "${PYTHON_BIN}" "${SCRIPT}"
                --mode infer
                --input "${DEV_INPUT}"
                --output "${OUTPUT_JSONL}"
                --model_ckpt "${CKPT_PATH}"
                --st_model "${ST_MODEL}"
                --batch_size "${BATCH_SIZE}"
                --infer_batch_size "${INFER_BATCH_SIZE}"
                --topic_top_k "${TOPIC_TOP_K}"
                --dde_hops "${DDE_HOPS}"
                --seed_edge_topk "${SEED_EDGE_TOPK}"
                --per_src_cap "${PER_SRC_CAP}"
                --max_nodes "${MAX_NODES}"
                --max_edges "${MAX_EDGES}"
                --max_sinks "${MAX_SINKS}"
                --keep_score
              )

              if [[ -n "${DEV_LIMIT}" ]]; then
                INFER_CMD+=(--limit "${DEV_LIMIT}")
              fi

              {
                printf "COMMAND:"
                printf " %q" "${INFER_CMD[@]}"
                printf "\n\n"
                "${INFER_CMD[@]}"
              } 2>&1 | tee "${INFER_LOG}"

              read -r ANSWER_RECALL GRAPH_RECALL NONE_SINK_RECALL < <(python - <<PY
import re
text = open("${INFER_LOG}", "r", encoding="utf-8").read()

def grab(name):
    m = re.findall(rf"{re.escape(name)}:\s*([0-9.]+)", text)
    return m[-1] if m else ""

a = grab("Answer recall")
g = grab("Graph  recall")
n = grab("None-sink recall")
print(a, g, n)
PY
)

              echo -e "${TRAIN_TAG}\t${INFER_TAG}\t${HIDDEN_DIM}\t${LR}\t${NEG_POS_RATIO}\t${SEED_EDGE_TOPK}\t${PER_SRC_CAP}\t${TOPIC_TOP_K}\t${BEST_DEV_F1}\t${ANSWER_RECALL}\t${GRAPH_RECALL}\t${NONE_SINK_RECALL}\t${CKPT_PATH}\t${OUTPUT_JSONL}\t${TRAIN_LOG}\t${INFER_LOG}" >> "${SUMMARY_TSV}"
            done
          done
        done
      done
    done
  done
done

echo
echo "All sweeps completed."
echo "Summary: ${SUMMARY_TSV}"