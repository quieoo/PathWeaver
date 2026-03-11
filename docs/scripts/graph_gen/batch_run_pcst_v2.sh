#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=python
SCRIPT=DAG_KV_GRetriever_PCST_v2.py

INPUT=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/hotpot_dev.jsonl
OUTPUT_DIR=/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv

ST_MODEL=/home/sdu/zhu/models/bge-en-v1.5/
BATCH_SIZE=256

mkdir -p "${OUTPUT_DIR}"

# best_topk_node_prize=8
# best_topk_edge_prize=8
# best_pcst_cost_e=0.3
# for max_nodes in 20 30 40; do
#     for max_edges in 30 40 50; do
max_nodes=40
max_edges=50

for topk_node_prize in 4 6 8; do
  for topk_edge_prize in 8 12 16; do
    for pcst_cost_e in 0.3 0.5 0.7; do
        output_file="${OUTPUT_DIR}/out_topn${topk_node_prize}_tope${topk_edge_prize}_c${pcst_cost_e}_n${max_nodes}_e${max_edges}.json"

        echo "============================================================"
        echo "Running: node=${topk_node_prize}, edge=${topk_edge_prize}, cost=${pcst_cost_e} max_nodes=${max_nodes}, max_edges=${max_edges}"
        echo "============================================================"

        ${PYTHON_BIN} ${SCRIPT} \
        --input "${INPUT}" \
        --output "${output_file}" \
        --st_model "${ST_MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --topk_node_prize "${topk_node_prize}" \
        --topk_edge_prize "${topk_edge_prize}" \
        --pcst_cost_e "${pcst_cost_e}" \
        --max_nodes "${max_nodes}" \
        --max_edges "${max_edges}" \
        --leaf_bonus 0.35 \
        --anchor_penalty 0.25 \
        --limit 100 
        done
    done
done