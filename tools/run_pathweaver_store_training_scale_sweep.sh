#!/usr/bin/env bash
set -euo pipefail

cd /mnt/n0/PathWeaver

PYTHON_BIN="${PYTHON_BIN:-/mnt/n0/uv_envs/kblam/bin/python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/stores/scale-sweep-20260706/training-tiers-v2}"
TRAIN_DATA="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/training_set/merged_multi_hop_train_tripled_v5_qwen2.5-72B_4bit.jsonl"

env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
  tools/build_pathweaver_store_scale_sweep.py \
  --base-store experiments/stores/scale-sweep-20260706/tiers/000837-2wiki-hotpot-musique \
  --output-root "${OUTPUT_ROOT}" \
  --base-label 000837-2wiki-hotpot-musique \
  --append-tier "016000-with-train::train-v5::${TRAIN_DATA}::0::15163" \
  --append-tier "032000-with-train::train-v5::${TRAIN_DATA}::15163::16000" \
  --append-tier "064000-with-train::train-v5::${TRAIN_DATA}::31163::32000" \
  --hnsw-embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --kv-embedding-model /mnt/n0/models/qwen-embedding-0.6B/ \
  --hnsw-embedding-batch-size 4096 \
  --kv-embedding-batch-size 4096 \
  --kv-encoding-profile qwen3-embedding-v2 \
  --ingest-commit-interval 100
