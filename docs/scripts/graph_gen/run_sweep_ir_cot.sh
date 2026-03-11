#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-configurable paths
# =========================
INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl"
OUTDIR="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/ir_cot_sweep_outputs"
PY="python"
SCRIPT="DAG_KV_IRCoT.py"
ST_MODEL="/home/sdu/zhu/models/bge-en-v1.5/"
BATCH_SIZE=256
# EXTRA_ARGS="---supporting_only --keep_score"

mkdir -p "$OUTDIR"
CSV="$OUTDIR/summary.csv"

echo "run_id,seed_top_m,max_steps,edges_per_step,leaf_bonus,max_sinks,beam_width,max_nodes,max_edges,answer_recall,graph_recall,none_sink_recall,output_file" > "$CSV"

# 27 carefully-chosen runs:
# Group A (1-9): retrieval depth/width tradeoff
# Group B (10-18): sink shaping / answer-at-leaf bias
# Group C (19-27): interaction cases near likely sweet spots
readarray -t CONFIGS <<'CFG'
01 4 2 3 0.04 2 2 24 32
02 4 2 4 0.04 2 2 24 36
03 4 3 4 0.04 2 2 28 40
04 6 2 3 0.04 2 2 24 32
05 6 2 4 0.04 2 2 28 40
06 6 3 4 0.04 2 2 30 40
07 8 2 4 0.04 2 2 30 40
08 8 3 4 0.04 2 2 32 44
09 8 3 5 0.04 2 2 34 48
10 6 3 4 0.00 2 2 30 40
11 6 3 4 0.08 2 2 30 40
12 6 3 4 0.12 2 2 30 40
13 6 3 4 0.08 3 2 30 40
14 6 3 4 0.12 3 2 30 40
15 6 3 4 0.08 4 2 30 40
16 8 3 4 0.08 2 2 32 44
17 8 3 4 0.12 3 2 32 44
18 4 3 4 0.12 2 2 28 40
19 6 4 4 0.08 2 2 32 44
20 6 4 5 0.08 2 2 34 48
21 8 4 4 0.08 2 2 34 48
22 8 4 5 0.08 3 2 36 52
23 6 3 5 0.12 2 2 32 44
24 8 3 5 0.12 2 2 34 48
25 6 4 4 0.12 2 3 32 44
26 8 4 4 0.12 3 3 36 52
27 6 3 4 0.12 2 3 30 40
CFG

for row in "${CONFIGS[@]}"; do
  read -r RUN_ID SEED STEPS EPS LEAF SINKS BEAM MAXN MAXE <<< "$row"

  OUTFILE="$OUTDIR/run_${RUN_ID}.jsonl"
  LOGFILE="$OUTDIR/logs/run_${RUN_ID}.log"

  echo "============================================================"
  echo "[RUN ${RUN_ID}] seed_top_m=${SEED} max_steps=${STEPS} edges_per_step=${EPS} leaf_bonus=${LEAF} max_sinks=${SINKS} beam_width=${BEAM} max_nodes=${MAXN} max_edges=${MAXE}"
  echo "============================================================"

  $PY "$SCRIPT" \
    --input "$INPUT" \
    --output "$OUTFILE" \
    --st_model "$ST_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --seed_top_m "$SEED" \
    --max_steps "$STEPS" \
    --edges_per_step "$EPS" \
    --leaf_bonus "$LEAF" \
    --max_sinks "$SINKS" \
    --beam_width "$BEAM" \
    --max_nodes "$MAXN" \
    --max_edges "$MAXE" \
    --limit 100 2>&1 | tee "$LOGFILE"

  ANSWER=$(grep -E "Answer recall:" "$LOGFILE" | tail -1 | awk '{print $3}')
  GRAPH=$(grep -E "Graph  recall:" "$LOGFILE" | tail -1 | awk '{print $3}')
  NONE_SINK=$(grep -E "None-sink recall:" "$LOGFILE" | tail -1 | awk '{print $3}')

  ANSWER=${ANSWER:-NA}
  GRAPH=${GRAPH:-NA}
  NONE_SINK=${NONE_SINK:-NA}

  echo "${RUN_ID},${SEED},${STEPS},${EPS},${LEAF},${SINKS},${BEAM},${MAXN},${MAXE},${ANSWER},${GRAPH},${NONE_SINK},${OUTFILE}" >> "$CSV"
done

echo
printf "[DONE] Summary saved to: %s\n" "$CSV"
printf "Top 10 by answer recall:\n"
{ head -n 1 "$CSV"; tail -n +2 "$CSV" | sort -t, -k10,10gr | head -n 10; } | column -s, -t
