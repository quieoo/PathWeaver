#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-configurable inputs
# =========================
INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl"
OUTDIR="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/subgraphrag_answer_recall_sweep"
PYTHON_BIN="python"
SCRIPT="DAG_KV_SubgraphRAG.py"
ST_MODEL="/home/sdu/zhu/models/bge-en-v1.5/"
BATCH_SIZE=256
LIMIT=${LIMIT:-100}
SUPPORTING_ONLY=${SUPPORTING_ONLY:-1}   # 1=true, 0=false
KEEP_SCORE=${KEEP_SCORE:-1}             # 1=true, 0=false

mkdir -p "${OUTDIR}"

# =========================
# Fixed strong baseline
# =========================
MAX_NODES=${MAX_NODES:-30}
MAX_EDGES=${MAX_EDGES:-40}

TOPIC_TOP_K=${TOPIC_TOP_K:-6}
EXPANSION_HOPS=${EXPANSION_HOPS:-2}
MENTION_BONUS=${MENTION_BONUS:-0.20}

W_KEY=${W_KEY:-0.34}
W_REL=${W_REL:-0.18}
W_SRC=${W_SRC:-0.10}
W_LEX=${W_LEX:-0.12}

# =========================
# Answer-recall-oriented sweep
# 24 groups = 3 x 2 x 2 x 2
# Sweep dimensions:
#   A: sink pressure / destination bias
#   B: directional bias
#   C: candidate sparsity
# =========================

# A: more aggressive pushing answer-like nodes to sinks
MAX_SINKS_LIST=(8 6 4)
W_DST_LIST=(0.16 0.22 0.28)
W_VAL_LIST=(0.08 0.12 0.16)
TOPIC_DST_BONUS_LIST=(0.15 0.22 0.30)

# B: directional / structural prior
W_DIR_LIST=(0.12 0.18)
W_DDE_LIST=(0.22 0.30)
DDE_HOPS_LIST=(3 4)

# C: keep graph compact, reduce noisy leaves
SEED_EDGE_TOPK_LIST=(18 14)
PER_SRC_CAP_LIST=(3 2)

run_id=0

for idxA in 0 1 2; do
  MAX_SINKS="${MAX_SINKS_LIST[$idxA]}"
  W_DST="${W_DST_LIST[$idxA]}"
  W_VAL="${W_VAL_LIST[$idxA]}"
  TOPIC_DST_BONUS="${TOPIC_DST_BONUS_LIST[$idxA]}"

  for idxB in 0 1; do
    W_DIR="${W_DIR_LIST[$idxB]}"
    W_DDE="${W_DDE_LIST[$idxB]}"
    DDE_HOPS="${DDE_HOPS_LIST[$idxB]}"

    for idxC in 0 1; do
      SEED_EDGE_TOPK="${SEED_EDGE_TOPK_LIST[$idxC]}"
      PER_SRC_CAP="${PER_SRC_CAP_LIST[$idxC]}"

      run_id=$((run_id + 1))
      tag=$(printf "run_%02d_ms%s_dst%s_val%s_tdb%s_dir%s_dde%s_h%s_topk%s_cap%s" \
        "${run_id}" \
        "${MAX_SINKS}" "${W_DST}" "${W_VAL}" "${TOPIC_DST_BONUS}" \
        "${W_DIR}" "${W_DDE}" "${DDE_HOPS}" "${SEED_EDGE_TOPK}" "${PER_SRC_CAP}")

      output_jsonl="${OUTDIR}/${tag}.jsonl"
      log_file="${OUTDIR}/${tag}.log"

      echo "================================================================"
      echo "[${run_id}/24] ${tag}"
      echo "output = ${output_jsonl}"
      echo "log    = ${log_file}"
      echo "================================================================"

      cmd=(
        "${PYTHON_BIN}" "${SCRIPT}"
        --input "${INPUT}"
        --output "${output_jsonl}"
        --st_model "${ST_MODEL}"
        --batch_size "${BATCH_SIZE}"
        --topic_top_k "${TOPIC_TOP_K}"
        --dde_hops "${DDE_HOPS}"
        --mention_bonus "${MENTION_BONUS}"
        --seed_edge_topk "${SEED_EDGE_TOPK}"
        --expansion_hops "${EXPANSION_HOPS}"
        --per_src_cap "${PER_SRC_CAP}"
        --max_nodes "${MAX_NODES}"
        --max_edges "${MAX_EDGES}"
        --max_sinks "${MAX_SINKS}"
        --w_key "${W_KEY}"
        --w_rel "${W_REL}"
        --w_val "${W_VAL}"
        --w_src "${W_SRC}"
        --w_dst "${W_DST}"
        --w_dir "${W_DIR}"
        --w_dde "${W_DDE}"
        --w_lex "${W_LEX}"
        --topic_dst_bonus "${TOPIC_DST_BONUS}"
      )

      if [[ "${KEEP_SCORE}" == "1" ]]; then
        cmd+=(--keep_score)
      fi
      if [[ "${SUPPORTING_ONLY}" == "1" ]]; then
        cmd+=(--supporting_only)
      fi
      if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
      fi

      {
        printf "COMMAND:"
        printf " %q" "${cmd[@]}"
        printf "\n\n"
        "${cmd[@]}"
      } 2>&1 | tee "${log_file}"
    done
  done
done

echo
echo "All 24 runs completed."
echo "Results saved in: ${OUTDIR}"