# Codebase Overview

This document summarizes the key components of the KBLaM repository to help new contributors quickly understand how knowledge base encoding, retrieval, and evaluation are organized.

## Core Library (`src/kblam`)

### GPT and Embedding Clients
- `gpt_session.py` defines thin wrappers around Azure OpenAI chat and embedding endpoints, including retry-aware `GPT` chat calls and `ALI_Embedding` for DashScope-compatible embeddings. The helpers centralize authentication, token handling, and batch embedding generation so higher-level modules can remain backend-agnostic.【F:src/kblam/gpt_session.py†L18-L177】

### Knowledge Base Encoding
- `kb_encoder.py` builds the knowledge token encoder used throughout KBLaM. It selects between sentence-transformer backbones, OpenAI embeddings, or Qwen embeddings, then projects outputs through configurable adapters (identity, linear, or MLP) for keys and values. Special token embeddings (`<KB_BEGIN>`, `<KB_END>`, separators) and optional freezing of the base encoder keep knowledge formatting consistent across models.【F:src/kblam/kb_encoder.py†L17-L176】

### Knowledge Retrieval Utilities
- `kb_retriever.py` orchestrates how encoded knowledge is sampled for training and inference. It can load cached embeddings, fetch on-the-fly encodings, and build per-batch context sets with randomized insertion to mix ground-truth and distractor triples. An alternate path constructs sparse adjacency matrices for two-hop knowledge graphs when working with cached embeddings.【F:src/kblam/kb_retriever.py†L9-L210】

### Evaluation Toolkit
- `metrics_evaluator.py` hosts text normalization plus multiple evaluation strategies: exact match, token-level F1 overlap, local semantic faithfulness via sentence-transformer similarity, and optional cloud-evaluated faithfulness using DashScope or OpenAI GPT endpoints. Graceful fallbacks ensure metrics degrade gracefully when dependencies or API keys are unavailable.【F:src/kblam/metrics_evaluator.py†L1-L113】

## Dataset Generation
- `dataset_generation/gen_synthetic_data.py` extends the GPT wrapper to drive synthetic KB construction. It composes structured prompts to create entity descriptions, derives question/answer/key tuples, and iterates across configurable idea and data-type vocabularies to populate the knowledge base for training.【F:dataset_generation/gen_synthetic_data.py†L14-L120】

## How to Navigate Next
- Start with `train.py` and the `experiments/` scripts to see how the encoder and retriever are wired into end-to-end training.
- The `tests/` directory provides runnable examples for validating embedding behavior and dataset handling.
- Refer back to the sections above to locate the specific module responsible for encoding, retrieval, or evaluation when modifying experiment pipelines.
