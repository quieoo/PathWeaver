#!/bin/bash

############################
# Common settings
############################

DATASET_DIR=../datasets/synthetic_embd
TEST_DATASET=test_synthetic_augmented.json
SAVE_DIR=./results

ENCODER_SPEC=all-MiniLM-L6-v2
LLM_TYPE=llama3
LLM_BASE_DIR=/home/sdu/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B

KB_LAYER_FREQUENCY=3
EVAL_MODE=kb

PRECOMPUTED_KEY_EMBD=${DATASET_DIR}/test_synthetic_all-MiniLM-L6-v2_embd_key.npy
PRECOMPUTED_VALUE_EMBD=${DATASET_DIR}/test_synthetic_all-MiniLM-L6-v2_embd_value.npy

############################
# Model & encoder paths
############################

ENCODER_DIR=/mnt/n0/KBLaM/old_train/train/synthetic/stage1_lr_0.0005KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_18000_encoder/encoder.pt

MODEL_DIR=/mnt/n0/KBLaM/old_train/train/synthetic/stage1_lr_0.0005KBTokenLayerFreq3UseExtendedQAMultiEntities2UseOutlier2KBSizedynamicSepQueryHeadUseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_18000

############################
# Evaluate HNSW (no parallel)
############################

echo "===== Evaluating HNSW (without parallel) ====="

python eval_hnsw_without_parallel.py generation \
  --dataset_dir ${DATASET_DIR} \
  --encoder_dir ${ENCODER_DIR} \
  --encoder_spec ${ENCODER_SPEC} \
  --llm_base_dir ${LLM_BASE_DIR} \
  --llm_type ${LLM_TYPE} \
  --model_dir ${MODEL_DIR} \
  --save_dir ${SAVE_DIR} \
  --test_dataset ${TEST_DATASET} \
  --precomputed_embed_keys_path ${PRECOMPUTED_KEY_EMBD} \
  --precomputed_embed_values_path ${PRECOMPUTED_VALUE_EMBD} \
  --eval_mode ${EVAL_MODE} \
  --kb_layer_frequency ${KB_LAYER_FREQUENCY}

echo "===== HNSW evaluation finished ====="
