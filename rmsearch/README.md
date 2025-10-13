
# RMSearch Package Guide

The `rmsearch` package bundles the search runtime, reward-model trainer, and
support tooling used throughout the RMSearch notebooks. This document maps the
most important entry points to their implementation files so you can jump
directly into the code when customising or debugging the pipeline.

## Download

### For users
```bash
git clone https://github.com/kyotoai/RMSearch.git
```

### For developpers
```bash
git clone --branch develop https://github.com/kyotoai/RMSearch.git
```

## Installation
```bash
pip install -e RMSearch/.
```


Both classes assume you have locally available model checkpoints and a GPU
environment with CUDA-visible devices.

## Core Runtime — [`rmsearch.py`](rmsearch.py)

- `Search.__init__`: Initialises a vLLM `AsyncLLMEngine` (or distributed
  `build_llm` pool) and chat-template formatter for scoring query/key pairs.
- `Search.__call__`: Async top‑k retrieval helper that calls `get_relevance`
  and returns structured results containing `query_id`, `key_id`, and optional
  relevance scores.
- `Search.search_by_df`: Scores every row in a `pandas.DataFrame`; optionally
  performs per-group top‑k selection when the frame contains `group`/`k`
  columns.
- `Search.get_relevance`: Materialises the Cartesian product between queries
  and keys, formats the prompts, and gathers tensor scores with tqdm logging.
- `Search.process` / `process_n_requests`: Internal coroutine(s) that stream
  responses from the async engine and map them to reward scores, supporting
  checkpointing via `save_results`.

These methods are written for coroutine contexts—use `asyncio.run` when calling
from synchronous scripts.

## Reward Model Trainer — [`rmtrain.py`](rmtrain.py)

- `RMTrainer.__init__`: Loads a sequence-classification head plus tokenizer for
  reward modelling; the `model_name` can point to existing fine-tunes.
- `RMTrainer.prepare_dataset`: Converts chat-style preference pairs into
  tokenised tensors (storing them under `base_dir` for reuse).
- `RMTrainer.train`: Wraps TRL’s `RewardTrainer` (via `CustomRewardTrainer`)
  and supports PEFT LoRA adapters by default.
- `CustomRewardTrainer.evaluate`: Overrides TRL’s evaluation to print samples,
  mirroring the notebook behaviour.

The trainer powers CLI scripts (see below) and can be reused programmatically
for custom evaluation or adapter settings.

## Training Pipeline Scripts — [`train/`](train)

- [`train/process_data.py`](train/process_data.py) → `process_data`: Download
  or stub HuggingFace datasets, shuffle, and materialise `df.csv` /
  `df_small.csv`.
- [`train/make_queries.py`](train/make_queries.py) → `make_queries`: Generate
  titles, keywords, relevant/irrelevant questions with an async vLLM backend.
- [`train/judge_dataset.py`](train/judge_dataset.py) → `judge_sentences`:
  Sample sentence pairs per query and request `<ID>` judgements.
- [`train/lora_example.py`](train/lora_example.py) → `make_dataset_list`,
  `train_reward_model`: Convert judge outputs into TRL preference records and
  launch a LoRA fine-tune.
- [`train/utils.py`](train/utils.py): Shared helpers such as
  `setup_async_engine`, the resumable `AllRequests` scheduler,
  `convert_model` (LoRA merge), and small parsing utilities.

Each script exposes a CLI to reproduce the notebook pipeline step-by-step.

## Tag Tree Workflow — [`tree/`](tree)

- [`tree/generate_tag.py`](tree/generate_tag.py) → `generate_tag`: Produce a
  JSON list of tags per key using `LLMWorkerModel` generation workers.
- [`tree/embed_tags.py`](tree/embed_tags.py) → `embed_tags`: Pool embeddings
  for every generated tag, returning a tensor plus `(key_id, tag_index)` map.
- [`tree/build_representative_tags.py`](tree/build_representative_tags.py) →
  `build_representative_tags`: Iteratively fill internal nodes with concise
  representative tags.
- [`tree/assign_key.py`](tree/assign_key.py) → `assign_key_to_tag_tree`: Walk a
  prepared tag tree with a reward model to assign queries to the best paths.
- [`tree/hierarchical_kmeans.py`](tree/hierarchical_kmeans.py) →
  `HierarchicalKMeans`: CPU-bound clustering utility for the initial tree
  layout (use `leaf_members_json()` to export the structure).

These modules compose the “Generate Tag Graph2” workflow from the original
notebook, now exposed as importable functions and CLIs.

## Retrieval Evaluation — [`evaluation/retrieval.py`](evaluation/retrieval.py)

- `retrieval_evaluation`: Reuses `assign_key_to_tag_tree` to collect candidate
  sentences per query before ranking them with a reward model search backend.
  The module ships with a CLI that mirrors the notebook evaluation cells.

## vLLM Backends — [`utils/`](utils)

- [`utils/vllm_generate.py`](utils/vllm_generate.py): Local generation worker
  pool exposing `build_llm` and `generate`.
- [`utils/vllm_embed.py`](utils/vllm_embed.py): Embedding variant that surfaces
  `build_embedding_model` and `embed`.
- [`utils/vllm_reward.py`](utils/vllm_reward.py) / [`utils/vllm_reward2.py`](utils/vllm_reward2.py):
  Pooling helpers optimised for reward-model scoring workloads.
- [`utils/vllm_serve_generate.py`](utils/vllm_serve_generate.py): HTTP client
  that talks to `vllm serve` endpoints while preserving the same call surface.

Select the helper that matches your deployment style (local GPU pool vs.
external `serve` process); all share the `SamplingParams` interface.

## Model Conversion Helper — [`utils.py`](utils.py)

- `convert_model`: Merge LoRA checkpoints into a base reward model and persist
  the combined weights alongside the scalar `score.pt`.
- `revert_model`: Placeholder for future undo functionality.

Call this after fine-tuning to prepare checkpoints for inference-time scoring.

## Example: Async Top‑k Search

```python
import asyncio
from rmsearch import Search

async def main():
    search = Search("/workspace/llama3b-rm-converted-model", tensor_parallel_size=1)
    queries = ["How does graph retrieval work?"]
    keys = ["Graph-based retrieval augmentation", "Best Italian recipes"]
    results = await search(queries, keys, k=1, return_relevance=True)
    print(results[0]["keys"][0])

asyncio.run(main())
```

## Download Reference Checkpoints (GPU environment)

```bash
cd /workspace
pip install "huggingface_hub[hf_transfer]"
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Ray2333/GRM-Llama3.2-3B-rewardmodel-ft --local-dir ./llama3b-rm/
```

```bash
cd /workspace
pip install "huggingface_hub[hf_transfer]"
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./qwen4b/
```


```bash
cd /workspace
pip install "huggingface_hub[hf_transfer]"
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download intfloat/e5-mistral-7b-instruct --local-dir ./e5-mistral7b/
```