#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-configurable inputs
# =========================
INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl"
OUTDIR="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hipporag_sweep_outputs"
PYTHON_BIN="python"
SCRIPT="DAG_KV_HippoRAG.py"
ST_MODEL="/home/sdu/zhu/models/bge-en-v1.5/"
BATCH_SIZE=256
LIMIT=${LIMIT:-100}
SUPPORTING_ONLY=${SUPPORTING_ONLY:-1}   # 1=true, 0=false
KEEP_SCORE=${KEEP_SCORE:-1}             # 1=true, 0=false

# Fixed graph budget (kept stable for fair comparison)
MAX_NODES=${MAX_NODES:-30}
MAX_EDGES=${MAX_EDGES:-40}
PPR_ITERS=${PPR_ITERS:-50}
MENTION_BONUS=${MENTION_BONUS:-0.25}
LEXICAL_WEIGHT=${LEXICAL_WEIGHT:-0.20}
EDGE_QUERY_WEIGHT=${EDGE_QUERY_WEIGHT:-0.10}
PRED_WEIGHT=${PRED_WEIGHT:-0.0}

mkdir -p "$OUTDIR/logs" "$OUTDIR/outputs"
SUMMARY="$OUTDIR/summary.csv"
echo "id,variant,seed_top_m,ppr_alpha,leaf_bonus,max_sinks,answer_recall,graph_recall,none_sink_recall,output_file,log_file" > "$SUMMARY"

run_one() {
  local id="$1"
  local variant="$2"
  local seed_top_m="$3"
  local ppr_alpha="$4"
  local leaf_bonus="$5"
  local max_sinks="$6"

  local out="$OUTDIR/outputs/${id}_${variant}_m${seed_top_m}_a${ppr_alpha}_leaf${leaf_bonus}_sink${max_sinks}.jsonl"
  local log="$OUTDIR/logs/${id}_${variant}_m${seed_top_m}_a${ppr_alpha}_leaf${leaf_bonus}_sink${max_sinks}.log"

  local cmd=(
    "$PYTHON_BIN" "$SCRIPT"
    --input "$INPUT"
    --output "$out"
    --variant "$variant"
    --st_model "$ST_MODEL"
    --batch_size "$BATCH_SIZE"
    --seed_top_m "$seed_top_m"
    --ppr_alpha "$ppr_alpha"
    --ppr_iters "$PPR_ITERS"
    --max_nodes "$MAX_NODES"
    --max_edges "$MAX_EDGES"
    --max_sinks "$max_sinks"
    --mention_bonus "$MENTION_BONUS"
    --lexical_weight "$LEXICAL_WEIGHT"
    --edge_query_weight "$EDGE_QUERY_WEIGHT"
    --leaf_bonus "$leaf_bonus"
    --pred_weight "$PRED_WEIGHT"
  )

  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [[ "$SUPPORTING_ONLY" == "1" ]]; then
    cmd+=(--supporting_only)
  fi
  if [[ "$KEEP_SCORE" == "1" ]]; then
    cmd+=(--keep_score)
  fi

  echo "[RUN] $id | $variant | seed_top_m=$seed_top_m ppr_alpha=$ppr_alpha leaf_bonus=$leaf_bonus max_sinks=$max_sinks"
  "${cmd[@]}" > "$log" 2>&1

  local answer graph none_sink
  answer=$(grep -E "Answer recall:" "$log" | tail -1 | awk '{print $3}')
  graph=$(grep -E "Graph  recall:" "$log" | tail -1 | awk '{print $3}')
  none_sink=$(grep -E "None-sink recall:" "$log" | tail -1 | awk '{print $3}')

  answer=${answer:-NA}
  graph=${graph:-NA}
  none_sink=${none_sink:-NA}

  echo "$id,$variant,$seed_top_m,$ppr_alpha,$leaf_bonus,$max_sinks,$answer,$graph,$none_sink,$out,$log" >> "$SUMMARY"
}

# ==========================================================
# 24-run sweep plan
# Goal: compare HippoRAG vs HippoRAG2 under the most valuable axes
#   1) retrieval coverage/depth proxy: seed_top_m, ppr_alpha
#   2) answer-to-sink pressure: leaf_bonus, max_sinks
#   3) interactions between propagation breadth and sink bias
# ==========================================================

# Group A (1-8): main effect of retrieval parameters under moderate sink bias
run_one 01 hipporag  4 0.10 0.08 3
run_one 02 hipporag  8 0.10 0.08 3
run_one 03 hipporag  8 0.15 0.08 3
run_one 04 hipporag 10 0.15 0.08 3
run_one 05 hipporag2 4 0.10 0.08 3
run_one 06 hipporag2 8 0.10 0.08 3
run_one 07 hipporag2 8 0.15 0.08 3
run_one 08 hipporag2 10 0.15 0.08 3

# Group B (9-16): sink/readout behavior around a strong mid-range retrieval setting
run_one 09 hipporag  8 0.15 0.04 2
run_one 10 hipporag  8 0.15 0.08 2
run_one 11 hipporag  8 0.15 0.12 2
run_one 12 hipporag  8 0.15 0.08 4
run_one 13 hipporag2 8 0.15 0.04 2
run_one 14 hipporag2 8 0.15 0.08 2
run_one 15 hipporag2 8 0.15 0.12 2
run_one 16 hipporag2 8 0.15 0.08 4

# Group C (17-24): interaction of stronger propagation with stronger sink pressure
run_one 17 hipporag  6 0.20 0.08 3
run_one 18 hipporag  8 0.20 0.12 3
run_one 19 hipporag 10 0.20 0.12 2
run_one 20 hipporag  6 0.15 0.12 2
run_one 21 hipporag2 6 0.20 0.08 3
run_one 22 hipporag2 8 0.20 0.12 3
run_one 23 hipporag2 10 0.20 0.12 2
run_one 24 hipporag2 6 0.15 0.12 2

echo

echo "[DONE] Sweep finished. Summary saved to: $SUMMARY"
echo "Top rows by answer recall:"
python - <<PY
import csv
from pathlib import Path
p = Path(r"$SUMMARY")
rows = list(csv.DictReader(p.open()))
rows = [r for r in rows if r['answer_recall'] not in ('', 'NA')]
rows.sort(key=lambda r: (float(r['answer_recall']), float(r['none_sink_recall']), float(r['graph_recall'])), reverse=True)
for r in rows[:8]:
    print(f"{r['id']}: {r['variant']} m={r['seed_top_m']} a={r['ppr_alpha']} leaf={r['leaf_bonus']} sink={r['max_sinks']} | ans={r['answer_recall']} graph={r['graph_recall']} none_sink={r['none_sink_recall']}")
PY
