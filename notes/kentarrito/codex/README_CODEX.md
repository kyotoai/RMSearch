# RMSearch Codex Overview

## What This Repository Provides
- **RMSearch** exposes a reward-model-driven retrieval engine that can rank candidate keys for a batch of queries using large language model embeddings instead of static semantic vectors.
- Training utilities help convert raw query/key preference data into TRL-ready datasets and fine-tune reward models with LoRA adapters.
- Example pipelines demonstrate how to scale vLLM inference across GPUs, generate hierarchical tag graphs, and orchestrate reward evaluation workloads.

## Core Python Modules
- `rmsearch/rmsearch.py` implements the asynchronous `Search` interface that talks to `AsyncLLMEngine` from vLLM. It formats query/key pairs into prompts, streams pooled embeddings, and optionally applies a saved `score.pt` projection for models converted with `utils.convert_model`.
- `rmsearch/rmtrain.py` provides `RMTrainer` for reward-model fine-tuning. It builds chat-formatted preference datasets (`query`, `chosen_key`, `rejected_key`), caches dataset splits, and wraps TRL's `RewardTrainer` with configurable LoRA adapters.
- `rmsearch/utils.py` hosts conversion helpers to split a reward model into generation weights plus a score head so that vLLM can run embedding inference while reusing the learned scoring layer.

## High-Value Example Scripts
- `examples/generate_tag_graph.py` is a CLI-style pipeline that (1) prompts an LLM for tags, (2) batches SentenceTransformer embeddings with quantisation/back-off, (3) clusters individual tags via torch k-means, and (4) synthesises representative tags per cluster.
- `examples/vllm_reward.py` and `examples/vllm_reward2.py` spin up multi-process vLLM workers for reward scoring, with the latter adding notebook-friendly heartbeats and richer logging.
- `examples/vllm_generate.py` and `examples/vllm_generate5.py` do the same for token generation workloads.
- `examples/deepspeed_test2.py` experiments with torchtune + DeepSpeed LoRA training; paired notes in `notes/deepspeed.md` capture environment hurdles and fixes.
- Multiple Jupyter notebooks (`examples/train_en.ipynb`, `dp_test.ipynb`, etc.) document data preprocessing, evaluation logic, and debugging trails for the search/retrieval experiments.

## Repository Layout
```
rmsearch/        Core library (search + training + utils)
examples/        Experiment scripts and notebooks (tag graph, vLLM workers, training)
notes/           Project notes, communications, roadmap artifacts
images/          Figures for README/demo decks
demo/            Canva slide decks (PDF) referenced by README
tests/           Placeholder test module (currently empty)
```

## Environment & Dependencies
- Target Python ≥ 3.10; `requirements.txt` lists `transformers`, `trl==0.14.0`, `peft`, `vllm`, `sentence_transformers`, etc.
- GPU acceleration is assumed. `Search` requires Ray ≥ 2.22.0 and a CUDA-capable environment for vLLM. Example scripts expect multiple GPUs when `device_groups` > 1.
- Optional extras: DeepSpeed, torchtune, accelerate, and jupyter are referenced in notebooks and scripts but not pinned in `requirements.txt`.

### Quick Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Install platform-specific CUDA wheels (torch, vllm) that match your driver stack before running GPU workloads.

## Search Engine Usage
```python
import asyncio
from rmsearch import Search

search = Search(
    model_name="/path/to/converted-model",  # directory containing weights + score.pt
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
)

queries = ["Why use reward models for retrieval?", "Best agents for graph search"]
keys = ["Reward shaping for LLMs", "Graph retrieval agents", "Classical TF-IDF"]

async def main():
    results = await search(queries, keys, k=2, return_relevance=True)
    for item in results:
        print(item)

asyncio.run(main())
```
Key behaviour:
- `Search.__call__` materialises the full query×key Cartesian product; large batches should chunk inputs or use `search_by_df` with grouping.
- Prompts are produced via the chat template passed at construction. The built-in template slices off the first 17 characters (`<|begin_of_text|>`); adapt this logic for other models.
- When `score.pt` exists alongside the model weights, vLLM only produces pooled embeddings and the saved score head provides scalar rewards.

## Reward Model Training Workflow
1. Assemble a list of preference items with `query`, `chosen_key`, `rejected_key` fields (either strings or chat-turn lists).
2. Call `RMTrainer.prepare_dataset(dataset_list, base_dir=...)` to tokenize, split, and cache the dataset. Custom formatting functions let you change tokenisation or prompt templates.
3. Provide `RewardConfig` or standard `TrainingArguments` plus a `LoraConfig` (defaults target last transformer blocks) to `RMTrainer.train`.
4. Save the fine-tuned model and run `utils.convert_model(model_path)` if you need a `score.pt` for vLLM search.

## Tag Graph Generation Cheatsheet
```bash
# 1. Ask an LLM for tags per key
python examples/generate_tag_graph.py generate-tags \
  --model Qwen2.5-3B-Instruct \
  --keys-file data/keys.txt \
  --out-tag-recs artifacts/tag_records.json

# 2. Embed tags & cache metadata
python examples/generate_tag_graph.py embed-tags \
  --embed-model intfloat/e5-mistral-7b-instruct \
  --tag-recs artifacts/tag_records.json \
  --embeddings-keys-out artifacts/key_emb.pt \
  --embeddings-tags-out artifacts/tag_emb.pt \
  --tag-meta-out artifacts/tag_meta.json \
  --reduce-dim 512

# 3. Cluster individual tags
python examples/generate_tag_graph.py group-tags \
  --tag-recs artifacts/tag_records.json \
  --embeddings-tags artifacts/tag_emb.pt \
  --n-group 100 \
  --centroids-out artifacts/centroids.pt \
  --group-recs-out artifacts/group_records.json \
  --tag-recs-out artifacts/tag_records_with_groups.json

# 4. Generate representative tags (optional)
python examples/generate_tag_graph.py representative-tags \
  --model Qwen2.5-3B-Instruct \
  --group-recs artifacts/group_records.json \
  --tag-recs artifacts/tag_records_with_groups.json \
  --embeddings-tags artifacts/tag_emb.pt \
  --group-recs-out artifacts/group_records_representative.json
```
The `demo` subcommand runs the entire loop on synthetic keys.

## Operational Notes & Gaps
- `rmsearch/rmsearch.py`’s `scheduling_strategy_fn` reads the global `tensor_parallel_size`; ensure it is set at module scope or refactor to use the instance attribute to avoid Ray placement errors.
- Prompt trimming (`prompt[17:]`) is a placeholder tuned for chat models that prefix `<|begin_of_text|>`. Generalise this when swapping templates.
- `Search.get_relevance` currently materialises all pairwise prompts; for large corpora integrate chunking or batched Ray datasets to avoid OOM.
- No automated tests exist (`tests/test.py` is empty); plan unit coverage for dataset formatting, convert_model, and search ranking.
- Packaging metadata in `setup.py` still exposes a placeholder console script (`my-tool`); update before publishing.

## Related Documentation
- `notes/COMUNICATION.md` records hand-offs and dataset artifacts shared among collaborators.
- `notes/deepspeed.md` logs DeepSpeed bring-up steps and fixes for environment-specific crashes.
- `notes/future_application.md` sketches product ideas (SEIMEI, Agent Note).
- `notes/kentarrito.md` is a running engineering diary with debugging milestones.
- `ROADMAP_CODEX.md` (this folder) tracks actionable follow-ups generated by this analysis.

## Where To Improve Next
Prioritise evaluation metrics (nDCG@k, etc.), integrate DeepSpeed configs into the main training workflow, and build reproducible experiments or scripts from the exploratory notebooks.
