# PathWeaver

PathWeaver is a research codebase for path-aware knowledge injection and DAG-KV
retrieval-augmented generation. It is built from KBLaM's knowledge-token
training stack, and extends it with graph-structured knowledge units, path
attention, multi-hop reasoning experiments, and multi-round knowledge-editable
evaluation.

The repository still uses the Python package name `kblam`, but the active
experiment workflow is PathWeaver:

- represent each sample as a DAG of key/value knowledge nodes;
- precompute key/value embeddings offline and keep them row-aligned with the
  dataset;
- train adapters that convert embeddings into KB tokens injected into the LLM;
- optionally enable path attention so graph edges can influence attention over
  KB tokens;
- evaluate generation quality, path-attention traces, DAG distractor settings,
  and knowledge updates across rounds.

## Repository Layout

```text
src/kblam/
  kb_encoder.py                    # embedding-to-KB-token adapter
  dag_kv_retriever.py              # DAG-KV dataset parser and retriever
  kblam_attention/kblam_path.py    # path-attention propagation and tracing
  models/                          # Llama, Phi, OLMo, Qwen model adapters
  metrics_evaluator.py             # exact match, F1, ROUGE, faithfulness helpers

experiments/
  train.py                         # main adapter training entrypoint
  eval_generation.py               # generation/debug evaluation entrypoint
  eval_generation_knowledge_editable.py
                                   # round-0 baseline knowledge-editable eval
  Knowledge_editable/              # multi-round edit dataset utilities

docs/scripts/
  embedding_v2.py                  # offline key/value embedding generation
  graph_gen/                       # DAG construction and graph-RAG baselines
  triple_gen/                      # triple extraction utilities

docs/EXPs/
  *.md                             # experiment notes and runnable command logs

tests/
  test_*.py                        # lightweight module and data tests
```

For a broader source map, see `docs/codebase_overview.md`. For experiment
commands, start from `docs/Startup.md` and `docs/EXPs/`.

## Environment

Python 3.10 or 3.11 is expected by `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional experiment dependencies can be installed with:

```bash
pip install -e ".[experiment]"
```

Most training and evaluation runs assume CUDA, local Hugging Face model
directories, and precomputed embedding `.npy` files. Some data-generation and
LLM-judge flows also require OpenAI-compatible or DashScope-compatible API
credentials; keep those in environment variables rather than in committed
commands.

## Data Contracts

PathWeaver supports several legacy dataset formats, but the current DAG-KV path
uses `--dataset_type dag`. Each sample should contain a question/answer pair and
a `dag` object:

```json
{
  "id": "sample-id",
  "question": "question text",
  "answer": "gold answer",
  "dag": {
    "kv_nodes": [
      {
        "key": "the relation/property prompt",
        "value": "knowledge value",
        "score": 0.92
      }
    ],
    "adj": [[0.0]]
  }
}
```

Important invariant: precomputed key/value embeddings are aligned to the
flattened `dag.kv_nodes` order over the full dataset. Do not physically filter,
shuffle, or rewrite rows when reusing existing `.npy` files. If an experiment
needs a subset, filter by sample IDs or query indices at inference time.

For `DAGKVKBRetriever`, the total number of rows in both embedding files must
equal `sum(len(sample["dag"]["kv_nodes"]) for sample in dataset)`. A mismatch is
treated as an error because it means the dataset and embedding files no longer
refer to the same nodes.

## Generate Embeddings

Use `docs/scripts/embedding_v2.py` after DAG construction or whenever the
dataset's `dag.kv_nodes` change.

```bash
python docs/scripts/embedding_v2.py \
  --model_name qwen3-embedding-0.6B \
  --local_model_path /path/to/qwen-embedding-0.6B \
  --dataset_type dag \
  --dataset_path /path/to/round0_dag_aa.jsonl \
  --batch_size 1024 \
  --progress
```

This writes files next to the dataset:

```text
round0_dag_aa_qwen-embedding-0.6B_embd_key.npy
round0_dag_aa_qwen-embedding-0.6B_embd_value.npy
```

The script also supports legacy formats such as `2wiki`, `all_triples`,
`at2qa_2wiki`, `autoschemakg_2wiki`, `musique`, and `synthetic`.

## Train

The main training entrypoint is `experiments/train.py`. A typical DAG-KV
adapter training command looks like:

```bash
python experiments/train.py \
  --seed 1 \
  --B 5 \
  --lr 5e-4 \
  --use_lr_decay \
  --gradient_accm_step 20 \
  --dataset_type dag \
  --sep_query_head \
  --duplicate_true_kb \
  --dynamic_kb_size 10 50 \
  --outlier_num -999999 \
  --kb_token_layer_frequency 3 \
  --path_attn \
  --encoder_spec qwen-embedding-0.6B \
  --key_embd_src key \
  --use_cached_embd \
  --train_data_path /path/to/train_dag.jsonl \
  --train_precomputed_embed_keys_path /path/to/train_embd_key.npy \
  --train_precomputed_embed_values_path /path/to/train_embd_value.npy \
  --test_data_path /path/to/dev_dag.jsonl \
  --test_precomputed_embed_keys_path /path/to/dev_embd_key.npy \
  --test_precomputed_embed_values_path /path/to/dev_embd_value.npy \
  --hf_model_spec /path/to/base-llm \
  --llm_type qwen3 \
  --test_kb_size 10 \
  --test_query_size 100 \
  --test_kb_scale_factor 4 \
  --eval_step 100 \
  --total_steps 8000 \
  --model_save_dir experiments/train/pathweaver_dag \
  --base_embeder_path /path/to/qwen-embedding-0.6B
```

Common knobs:

- `--path_attn` enables graph/path-aware attention over KB tokens.
- `--kb_token_layer_frequency` controls how often KB tokens are injected.
- `--dynamic_kb_size MIN MAX` samples KB sizes during training.
- `--use_cached_embd` uses offline `.npy` embeddings instead of recomputing
  embeddings inside training.
- `--sep_query_head` trains a separate query head when supported by the model
  adapter.

## Evaluate Generation

Use `experiments/eval_generation.py generation` for normal generation
evaluation:

```bash
python experiments/eval_generation.py generation \
  --dataset_dir /path/to/datasets \
  --test_dataset round0_dag_aa.jsonl \
  --model_dir /path/to/checkpoint \
  --encoder_dir /path/to/checkpoint_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /path/to/base-llm \
  --llm_type qwen3 \
  --dataset_type dag \
  --precomputed_embed_keys_path /path/to/round0_embd_key.npy \
  --precomputed_embed_values_path /path/to/round0_embd_value.npy \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --kb_scale_factor 4 \
  --dag_kb_size 1 \
  --save_dir experiments/gen_tmp
```

Useful evaluation flags:

- `--dag_kb_size N` concatenates multiple DAG samples into one inference KB for
  distractor experiments (`1` means no extra DAG samples).
- `--enable_trace --path_attn_trace_path trace.pt` dumps path-attention trace
  records when path attention is enabled.
- `--kb_scale_factor_range START END` sweeps multiplicative KB scaling factors.
- `--full_eval` controls whether full ROUGE/F1/faithfulness evaluation is used.

The script reports exact match, F1 overlap, ROUGE, and faithfulness-style
metrics depending on the selected evaluation mode and installed dependencies.

## Knowledge-Editable Evaluation

`experiments/eval_generation_knowledge_editable.py` wraps the legacy generation
pipeline and adds fixed round-0 baseline metrics for multi-round edits.

The intended semantics are:

1. evaluate round 0 once;
2. evaluate one or more target rounds;
3. compare target-round correctness against the same round-0 baseline;
4. decompose dropped baseline accuracy into stale-answer leakage and other
   errors.

Example:

```bash
python experiments/eval_generation_knowledge_editable.py generation \
  --dataset_dir /path/to/rounds \
  --round0_test_dataset round0_dag_aa.jsonl \
  --round0_precomputed_embed_keys_path /path/to/round0_embd_key.npy \
  --round0_precomputed_embed_values_path /path/to/round0_embd_value.npy \
  --target_datasets round1_dag_aa.jsonl round2_dag_aa.jsonl \
  --target_precomputed_embed_keys_paths /path/to/round1_embd_key.npy /path/to/round2_embd_key.npy \
  --target_precomputed_embed_values_paths /path/to/round1_embd_value.npy /path/to/round2_embd_value.npy \
  --model_dir /path/to/checkpoint \
  --encoder_dir /path/to/checkpoint_encoder/encoder.pt \
  --encoder_spec qwen-embedding-0.6B \
  --llm_base_dir /path/to/base-llm \
  --llm_type qwen3 \
  --dataset_type dag \
  --kb_layer_frequency 3 \
  --kb_size 10 \
  --query_size 100 \
  --path_attn \
  --kb_scale_factor 4 \
  --use_id_intersection
```

`--use_id_intersection` evaluates only sample IDs present in round 0 and all
target rounds, while preserving the original datasets and embedding row order.

For a KV-only cumulative baseline, first materialize cumulative datasets:

```bash
python experiments/Knowledge_editable/build_kv_only_cumulative.py \
  --main-round-files round0_dag_aa.jsonl round1_dag_aa.jsonl round2_dag_aa.jsonl \
  --target-files round1_dag_aa.jsonl round2_dag_aa.jsonl
```

Then regenerate embeddings for the generated `*_kv_only_cumulative.jsonl`
files and evaluate without `--path_attn`.

## Baselines and Utilities

- Vector RAG and graph RAG baselines live under `experiments/` and
  `docs/scripts/graph_gen/`.
- Triple extraction utilities are in `tools/` and `docs/scripts/triple_gen/`.
- `experiments/msa.py` contains the MSA baseline workflow.
- `experiments/parse_log.py`, `docs/scripts/parse_log_metrics.py`, and
  `eval_acc.ipynb` are useful for log and metric inspection.

## Development Checks

Run lightweight tests with:

```bash
pytest tests
```

For script-only changes in an environment with missing optional packages, a
static syntax check is often the quickest sanity check:

```bash
python -m py_compile experiments/eval_generation.py
python -m py_compile experiments/eval_generation_knowledge_editable.py
```

## Project Notes

This repository is derived from the KBLaM implementation:
`KBLaM: Knowledge Base Augmented Language Models` (ICLR 2025). The original
license is MIT; see `LICENSE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and
`SUPPORT.md`.

When adding new experiment variants, prefer small wrapper scripts or new
standalone utilities over changing legacy entrypoints. This keeps old command
logs reproducible while allowing PathWeaver-specific workflows to evolve.
