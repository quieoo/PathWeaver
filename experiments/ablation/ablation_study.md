性能 vs 时延的Breakdown
    - VectorRAG-as-Text: Baseline RAG
        预期结果：TTFT高，精度差
    - DAG-as-Text：DAG-Retriever检索出来的三元组+图的文字性描述语言，序列进Prompt
        预期结果：TTFT略高，但是低于VectorRAG；精度略差，因为文本化 DAG 并不能稳定让 LLM 用好结构
    - KV-only (KBLaM + 实体检索)：通过实体检索定位到一个子区域，然后全部转换为KV Token注入
        预期结果：无注意力传播，噪音大，精度差
    - Local Propagation
        预期结果：注意力传播，但是噪音强，精度差
    - DAG-Retriever (DAG-KV)
        预期结果：抑制噪音，精度提高，TTFT上升一点点但是可以接受

# VectorRAG-as-Text
````bash
python3 /mnt/n0/PathWeaver/experiments/vector_rag.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/source_data/2wiki_dev_2hop.json \
  --index-path /mnt/n0/PathWeaver/experiments/vector_rag_index/2wiki_bge \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --embedding-model /mnt/n0/models/bge-en-v1.5/ \
  --n-samples 100 \
  --similarity-top-k 64 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.8
````

# DAG-as-Text
````bash
python3 /mnt/n0/PathWeaver/experiments/ablation/ablation_infer.py \
  --dataset-path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl \
  --model-path /mnt/n0/models/qwen3-4B-Instruct \
  --query-size 100 \
  --dag-kb-size 10 \
  --max-output-len 16 \
  --max-model-len 65536 \
  --print-first-prompt
````

# KV-only

## V1-all-triples
````bash
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --dataset_type all_triples \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional.jsonl \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B

````

````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples \
  --test_dataset 2wiki_dev_2hop_compositional.jsonl  \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800 \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type all_triples \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/2wiki_dev_2hop_compositional_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --step 6800 \
  --t_step 8000 \
  --kb_scale_factor 4 

# QPS: 4.79
# Average Latency: 0.2087
# Avg TTFT: 0.0532
# Avg TPOT: 0.0470
# ---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.10372222222222222, 'rouge2': 0.0025, 'rougeL': 0.10466666666666667, 'rougeLsum': 0.10477777777777778, 'exact_match': 0.07, 'f1_overlap': 0.0968888888888889, 'faithfulness01': 0.1}
````

## V2-supporting-facts-triples

````bash
python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --dataset_type all_triples \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B

python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset 2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl  \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800 \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type all_triples \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --step 6800 \
  --t_step 8000 \
  --kb_scale_factor 4 

# ---- [1/1] kb_scale_factor: 4.0, {'rouge1': 0.1974591584885702, 'rouge2': 0.025666666666666664, 'rougeL': 0.19991712372594728, 'rougeLsum': 0.1990294476765065, 'exact_match': 0.12, 'f1_overlap': 0.1979105339105339, 'faithfulness01': 0.14}
````
# local propagation

在graph_gen中增加一个旁路，local_propagation，不做检索，将所有三元组合并成一张图。去环。
````bash
python /mnt/n0/PathWeaver/docs/scripts/graph_gen/DAG_KV_SubgraphRAG_trainable_v5_2.py \
  --mode infer \
  --local_propogation \
  --input /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B.jsonl \
  --output /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_local.jsonl

python /mnt/n0/PathWeaver/docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /mnt/n0/models/qwen-embedding-0.6B/ \
  --dataset_type dag \
  --dataset_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_local.jsonl \
  --batch_size 1024 \
  --progress

python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset 2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_local.jsonl  \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800 \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_local_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_local_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 6800 \
  --t_step 8000 \
  --kb_scale_factor 4 
````


# DAG-KV
````bash
python3 /mnt/n0/PathWeaver/experiments/eval_generation.py generation \
  --dataset_dir /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set \
  --test_dataset 2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa.jsonl  \
  --model_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800 \
  --encoder_dir /mnt/n0/PathWeaver/experiments/train/dag_kv_merged_hybrid_training_v2.1_qwen3_4B_aa/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier-999999KBSizedynamicSepQueryHeadKeyFromkey_qwen-embedding-0.6B_dag_qwen3_step_7800_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /mnt/n0/models/qwen3-4B-Instruct \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_key.npy \
  --precomputed_embed_values_path /mnt/n0/datasets/wiki_hotspot_musique/merged_data/dag-kv/test_set/2wiki_dev_2hop_tripled_v5-qwen3.5-27B_dag_aa_qwen-embedding-0.6B_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --step 6800 \
  --t_step 8000 \
  --kb_scale_factor 4 
````