#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-configurable inputs
# =========================
INPUT="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl"
OUTDIR="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/uniqorn_sweep_outputs"
PYTHON_BIN="python"
SCRIPT="DAG_KV_UNIQORN.py"
ST_MODEL="/home/sdu/zhu/models/bge-en-v1.5/"
BATCH_SIZE=256
LIMIT=${LIMIT:-100}
SUPPORTING_ONLY=${SUPPORTING_ONLY:-1}   # 1=true, 0=false
KEEP_SCORE=${KEEP_SCORE:-1}             # 1=true, 0=false

# =========================
# Fixed params
# =========================
PRED_WEIGHT=${PRED_WEIGHT:-0.25}
TITLE_WEIGHT=${TITLE_WEIGHT:-0.05}
ENTITY_SIM_TH=${ENTITY_SIM_TH:-0.32}
REL_SIM_TH=${REL_SIM_TH:-0.30}
MIN_TOKEN_OVERLAP=${MIN_TOKEN_OVERLAP:-1}
MAX_ROOTS=${MAX_ROOTS:-24}

mkdir -p "$OUTDIR/logs" "$OUTDIR/outputs"
SUMMARY="$OUTDIR/summary.csv"
echo "id,max_anchor_groups,max_nodes_per_group,per_group_top_t,top_k_gst,max_nodes,max_edges,answer_recall,graph_recall,none_sink_recall,output_file,log_file" > "$SUMMARY"

run_one() {
  local id="$1"
  local max_anchor_groups="$2"
  local max_nodes_per_group="$3"
  local per_group_top_t="$4"
  local top_k_gst="$5"
  local max_nodes="$6"
  local max_edges="$7"

  local out="$OUTDIR/outputs/${id}_g${max_anchor_groups}_ng${max_nodes_per_group}_t${per_group_top_t}_gst${top_k_gst}_n${max_nodes}_e${max_edges}.jsonl"
  local log="$OUTDIR/logs/${id}_g${max_anchor_groups}_ng${max_nodes_per_group}_t${per_group_top_t}_gst${top_k_gst}_n${max_nodes}_e${max_edges}.log"

  local cmd=(
    "$PYTHON_BIN" "$SCRIPT"
    --input "$INPUT"
    --output "$out"
    --st_model "$ST_MODEL"
    --batch_size "$BATCH_SIZE"
    --pred_weight "$PRED_WEIGHT"
    --title_weight "$TITLE_WEIGHT"
    --max_anchor_groups "$max_anchor_groups"
    --max_nodes_per_group "$max_nodes_per_group"
    --entity_sim_th "$ENTITY_SIM_TH"
    --rel_sim_th "$REL_SIM_TH"
    --min_token_overlap "$MIN_TOKEN_OVERLAP"
    --top_k_gst "$top_k_gst"
    --max_roots "$MAX_ROOTS"
    --per_group_top_t "$per_group_top_t"
    --max_nodes "$max_nodes"
    --max_edges "$max_edges"
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

  echo "[RUN] $id | groups=$max_anchor_groups nodes/group=$max_nodes_per_group top_t=$per_group_top_t gst=$top_k_gst max_nodes=$max_nodes max_edges=$max_edges"
  "${cmd[@]}" > "$log" 2>&1

  local answer graph none_sink
  answer=$(grep -E "Answer recall:" "$log" | tail -1 | awk '{print $3}')
  graph=$(grep -E "Graph  recall:" "$log" | tail -1 | awk '{print $3}')
  none_sink=$(grep -E "None-sink recall:" "$log" | tail -1 | awk '{print $3}')

  answer=${answer:-NA}
  graph=${graph:-NA}
  none_sink=${none_sink:-NA}

  echo "$id,$max_anchor_groups,$max_nodes_per_group,$per_group_top_t,$top_k_gst,$max_nodes,$max_edges,$answer,$graph,$none_sink,$out,$log" >> "$SUMMARY"
}

# ==========================================================
# 96-run sweep plan
# Goal:
#   1) anchor coverage:         max_anchor_groups
#   2) candidate breadth:       max_nodes_per_group
#   3) terminal selection:      per_group_top_t
#   4) evidence diversity:      top_k_gst
#   5) final DAG size budget:   max_nodes, max_edges
#
# Total:
#   3 x 2 x 2 x 2 x 2 x 2 = 96
# ==========================================================

# Group A: max_anchor_groups = 4
run_one 001 4 6 3 3 24 32
run_one 002 4 6 3 3 24 40
run_one 003 4 6 3 3 30 32
run_one 004 4 6 3 3 30 40
run_one 005 4 6 3 5 24 32
run_one 006 4 6 3 5 24 40
run_one 007 4 6 3 5 30 32
run_one 008 4 6 3 5 30 40
run_one 009 4 6 4 3 24 32
run_one 010 4 6 4 3 24 40
run_one 011 4 6 4 3 30 32
run_one 012 4 6 4 3 30 40
run_one 013 4 6 4 5 24 32
run_one 014 4 6 4 5 24 40
run_one 015 4 6 4 5 30 32
run_one 016 4 6 4 5 30 40
run_one 017 4 8 3 3 24 32
run_one 018 4 8 3 3 24 40
run_one 019 4 8 3 3 30 32
run_one 020 4 8 3 3 30 40
run_one 021 4 8 3 5 24 32
run_one 022 4 8 3 5 24 40
run_one 023 4 8 3 5 30 32
run_one 024 4 8 3 5 30 40
run_one 025 4 8 4 3 24 32
run_one 026 4 8 4 3 24 40
run_one 027 4 8 4 3 30 32
run_one 028 4 8 4 3 30 40
run_one 029 4 8 4 5 24 32
run_one 030 4 8 4 5 24 40
run_one 031 4 8 4 5 30 32
run_one 032 4 8 4 5 30 40

# Group B: max_anchor_groups = 5
run_one 033 5 6 3 3 24 32
run_one 034 5 6 3 3 24 40
run_one 035 5 6 3 3 30 32
run_one 036 5 6 3 3 30 40
run_one 037 5 6 3 5 24 32
run_one 038 5 6 3 5 24 40
run_one 039 5 6 3 5 30 32
run_one 040 5 6 3 5 30 40
run_one 041 5 6 4 3 24 32
run_one 042 5 6 4 3 24 40
run_one 043 5 6 4 3 30 32
run_one 044 5 6 4 3 30 40
run_one 045 5 6 4 5 24 32
run_one 046 5 6 4 5 24 40
run_one 047 5 6 4 5 30 32
run_one 048 5 6 4 5 30 40
run_one 049 5 8 3 3 24 32
run_one 050 5 8 3 3 24 40
run_one 051 5 8 3 3 30 32
run_one 052 5 8 3 3 30 40
run_one 053 5 8 3 5 24 32
run_one 054 5 8 3 5 24 40
run_one 055 5 8 3 5 30 32
run_one 056 5 8 3 5 30 40
run_one 057 5 8 4 3 24 32
run_one 058 5 8 4 3 24 40
run_one 059 5 8 4 3 30 32
run_one 060 5 8 4 3 30 40
run_one 061 5 8 4 5 24 32
run_one 062 5 8 4 5 24 40
run_one 063 5 8 4 5 30 32
run_one 064 5 8 4 5 30 40

# Group C: max_anchor_groups = 6
run_one 065 6 6 3 3 24 32
run_one 066 6 6 3 3 24 40
run_one 067 6 6 3 3 30 32
run_one 068 6 6 3 3 30 40
run_one 069 6 6 3 5 24 32
run_one 070 6 6 3 5 24 40
run_one 071 6 6 3 5 30 32
run_one 072 6 6 3 5 30 40
run_one 073 6 6 4 3 24 32
run_one 074 6 6 4 3 24 40
run_one 075 6 6 4 3 30 32
run_one 076 6 6 4 3 30 40
run_one 077 6 6 4 5 24 32
run_one 078 6 6 4 5 24 40
run_one 079 6 6 4 5 30 32
run_one 080 6 6 4 5 30 40
run_one 081 6 8 3 3 24 32
run_one 082 6 8 3 3 24 40
run_one 083 6 8 3 3 30 32
run_one 084 6 8 3 3 30 40
run_one 085 6 8 3 5 24 32
run_one 086 6 8 3 5 24 40
run_one 087 6 8 3 5 30 32
run_one 088 6 8 3 5 30 40
run_one 089 6 8 4 3 24 32
run_one 090 6 8 4 3 24 40
run_one 091 6 8 4 3 30 32
run_one 092 6 8 4 3 30 40
run_one 093 6 8 4 5 24 32
run_one 094 6 8 4 5 24 40
run_one 095 6 8 4 5 30 32
run_one 096 6 8 4 5 30 40

echo
echo "[DONE] Sweep finished. Summary saved to: $SUMMARY"
echo "Top rows by answer recall:"
python - <<PY
import csv
from pathlib import Path

p = Path(r"$SUMMARY")
rows = list(csv.DictReader(p.open()))
rows = [r for r in rows if r['answer_recall'] not in ('', 'NA')]
rows.sort(
    key=lambda r: (
        float(r['answer_recall']),
        float(r['none_sink_recall']),
        float(r['graph_recall'])
    ),
    reverse=True
)
for r in rows[:10]:
    print(
        f"{r['id']}: "
        f"groups={r['max_anchor_groups']} "
        f"nodes/group={r['max_nodes_per_group']} "
        f"top_t={r['per_group_top_t']} "
        f"gst={r['top_k_gst']} "
        f"n={r['max_nodes']} "
        f"e={r['max_edges']} | "
        f"ans={r['answer_recall']} "
        f"graph={r['graph_recall']} "
        f"none_sink={r['none_sink_recall']}"
    )
PY