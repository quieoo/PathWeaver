# DAG-KV Qwen3-4B inference on Ascend NPU

Run these commands inside the already running Docker container. Select an idle
physical NPU before starting; the process exposes it as logical `npu:0`.

```bash
export ASCEND_RT_VISIBLE_DEVICES=7
export DAG_KV_DEVICE=npu
export PYTHONPATH=/workspace/dag_kv_ascend/PathWeaver/src:${PYTHONPATH:-}
```

All commands below evaluate the first 100 examples. Run them one at a time.

## PopQA

```bash
python /workspace/dag_kv_ascend/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /workspace/dag_kv_ascend/datasets/test_set \
  --test_dataset popqa.jsonl \
  --model_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800 \
  --encoder_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /workspace/dag_kv_ascend/models/qwen3-4B-Instruct \
  --llm_type qwen3 --dataset_type dag \
  --precomputed_embed_keys_path /workspace/dag_kv_ascend/datasets/test_set/popqa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /workspace/dag_kv_ascend/datasets/test_set/popqa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 --dag_kb_size 1 --query_size 100 --path_attn \
  --step 6800 --t_step 8000 --kb_scale_factor 4 --no-full_eval
```

## SQuAD v4

```bash
python /workspace/dag_kv_ascend/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /workspace/dag_kv_ascend/datasets/test_set \
  --test_dataset squad_v4.jsonl \
  --model_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800 \
  --encoder_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /workspace/dag_kv_ascend/models/qwen3-4B-Instruct \
  --llm_type qwen3 --dataset_type dag \
  --precomputed_embed_keys_path /workspace/dag_kv_ascend/datasets/test_set/squad_v4_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /workspace/dag_kv_ascend/datasets/test_set/squad_v4_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 --dag_kb_size 1 --query_size 100 --path_attn \
  --step 6800 --t_step 8000 --kb_scale_factor 4 --no-full_eval
```

## 2WikiMultiHopQA

```bash
python /workspace/dag_kv_ascend/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /workspace/dag_kv_ascend/datasets/test_set \
  --test_dataset 2wiki_dev_2hop_tripled_v5-qwen3-4B-Instruct_dag_aa.jsonl \
  --model_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800 \
  --encoder_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /workspace/dag_kv_ascend/models/qwen3-4B-Instruct \
  --llm_type qwen3 --dataset_type dag \
  --precomputed_embed_keys_path /workspace/dag_kv_ascend/datasets/test_set/2wiki_dev_2hop_tripled_v5-qwen3-4B-Instruct_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /workspace/dag_kv_ascend/datasets/test_set/2wiki_dev_2hop_tripled_v5-qwen3-4B-Instruct_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 --dag_kb_size 1 --query_size 100 --path_attn \
  --step 6800 --t_step 8000 --kb_scale_factor 4 --no-full_eval
```

## HotpotQA

```bash
python /workspace/dag_kv_ascend/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /workspace/dag_kv_ascend/datasets/test_set \
  --test_dataset hotpot_dev_tripled_v5-qwen3-4B-Instruct_dag_aa.jsonl \
  --model_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800 \
  --encoder_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /workspace/dag_kv_ascend/models/qwen3-4B-Instruct \
  --llm_type qwen3 --dataset_type dag \
  --precomputed_embed_keys_path /workspace/dag_kv_ascend/datasets/test_set/hotpot_dev_tripled_v5-qwen3-4B-Instruct_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /workspace/dag_kv_ascend/datasets/test_set/hotpot_dev_tripled_v5-qwen3-4B-Instruct_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 --dag_kb_size 1 --query_size 100 --path_attn \
  --step 6800 --t_step 8000 --kb_scale_factor 4 --no-full_eval
```

## MuSiQue

```bash
python /workspace/dag_kv_ascend/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /workspace/dag_kv_ascend/datasets/test_set \
  --test_dataset musique_dev_tripled_v5-qwen3-4B-Instruct_dag_aa.jsonl \
  --model_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800 \
  --encoder_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /workspace/dag_kv_ascend/models/qwen3-4B-Instruct \
  --llm_type qwen3 --dataset_type dag \
  --precomputed_embed_keys_path /workspace/dag_kv_ascend/datasets/test_set/musique_dev_tripled_v5-qwen3-4B-Instruct_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /workspace/dag_kv_ascend/datasets/test_set/musique_dev_tripled_v5-qwen3-4B-Instruct_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 --dag_kb_size 1 --query_size 100 --path_attn \
  --step 6800 --t_step 8000 --kb_scale_factor 4 --no-full_eval
```

## MintQA (2-hop)

```bash
python /workspace/dag_kv_ascend/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /workspace/dag_kv_ascend/datasets/test_set \
  --test_dataset mintqa_pruned64_hop2_dag_aa.jsonl \
  --model_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800 \
  --encoder_dir /workspace/dag_kv_ascend/models/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_6800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /workspace/dag_kv_ascend/models/qwen3-4B-Instruct \
  --llm_type qwen3 --dataset_type dag \
  --precomputed_embed_keys_path /workspace/dag_kv_ascend/datasets/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /workspace/dag_kv_ascend/datasets/test_set/mintqa_pruned64_hop2_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 --dag_kb_size 1 --query_size 100 --path_attn \
  --step 6800 --t_step 8000 --kb_scale_factor 4 --no-full_eval
```
