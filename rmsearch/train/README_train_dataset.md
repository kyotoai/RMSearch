# RMSearch Training Utilities

The scripts in this directory mirror the data and reward-model preparation
steps from `examples/train_en.ipynb`, but expose them as command-line tools so
you can run the pipeline outside the notebook. Each command writes the same
artifacts the notebook expects, and assumes you have a GPU-enabled environment
with the appropriate models available locally.



## Overview

1. `process_data.py`: download data from huggingface and save it to local volume
2. `make_query_recs.py`: from the data, make queries in records format about each data row. 
3. `filter_query_recs.py`: Filter `query_recs.json` by `query-type` and persist the subset to a new file.
5. `get_top_relevant_keys_embed.py`: Get top-relevant rows from dataframe by embedding model.
6. `sample_dpo_batch.py`: Sample data rows from query_recs
7. `judge_dataset.py`: From query_key_set created from `sample_dpo_batch.py`, make dpo dataset by judging which key is more relevant to query.

* Running order: 1 ~ 7 in order


## Install rmsearch

```bash
git clone --branch develop https://github.com/kyotoai/RMSearch.git
pip install -e RMSearch/.
```


## `process_data.py`

Download a dataset from HuggingFace, shuffle it, and materialise convenient CSV
slices.

```bash
python -m rmsearch.train.process_data \
  --dataset-name HuggingFaceTB/smollm-corpus \
  --output-dir ./data/smollm-corpus \
  --dataset-config cosmopedia-v2 \
  --n-sample 1000 \
  --stream
```
Omit `--n-sample` entirely if you want to materialise the full split.

**Arguments**
- `--dataset-name`: HuggingFace dataset identifier.
- `--output-dir`: Directory where HF `dataset_dict.json` plus `df.csv` / `df_small.csv` are stored.
- `--n-sample`: Optional cap if you only want to persist a sampled subset (applies to the saved dataset and CSVs).
- `--dataset-config`: Optional configuration name if the dataset exposes multiple configs.
- `--split`: Dataset split to load (defaults to `train`).
- `--random-seed`: Shuffle seed.
- `--stream`: Load via the HuggingFace streaming API before materialising rows locally.

**Outputs**
- `<output-dir>/dataset_dict.json` (HF binary format when `datasets` is installed).
- `<output-dir>/df.csv` full sample, `<output-dir>/df_small.csv` subset (default max 10k rows).
- Example row in `df_small.csv`:
  `{"text": "Graph-based retrieval augmentation for enterprise documents"}`

**Notices**
- Requires `datasets` for real downloads; otherwise a stub CSV is produced.
- Set `HF_HUB_OFFLINE=1` (or `HF_DATASETS_OFFLINE=1`) to skip network calls and immediately generate the stub outputs.
- The `HuggingFaceTB/smollm-corpus` snapshot uses the `cosmopedia-v2` configuration; pass `--dataset-config cosmopedia-v2` if you want that specific slice.
- When storage is limited, supply `--n-sample` to only keep that many rows in both the saved DatasetDict and CSVs; omit it to persist the entire split.
- Combine `--stream` with `--n-sample` to limit in-memory buffering; the script dynamically sizes the shuffle buffer so streamed subsets still appear random without staging the entire split on disk.
- Large datasets may need additional disk space.



## `make_query_recs.py`.    

Flatten the generated titles, keywords, questions, and irrelevant questions into
per-query recommendation records while reusing the same vLLM backend.

```bash
python -m rmsearch.train.make_query_recs \
  --input-csv ./data/smollm-corpus/df.csv \
  --text-column text \
  --model-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --batch-size 8 \
  --max-model-len 10000 \
  --output ./data/smollm-corpus/query_recs.json
```

**Arguments**
- Inherits the same CLI as `make_queries.py`; see above for detailed flag descriptions.

**Outputs**
- `{output}`: JSON list where each element contains `query`, `df_id`, and `query-type`, covering every generated title/keyword/question/irrelevant question.
- Example entry:
  ```json
  [
    {"query": "Graph Retrieval Overview", "df_id": 42, "query-type": "titles"},
    {"query": "How does graph retrieval work?", "df_id": 42, "query-type": "questions"}
  ]
  ```

**Notices**
- Shares batching, sampling, and fallback behaviour with `make_queries.py`; refer to that section for runtime considerations.



## `filter_query_recs.py`

Filter `query_recs.json` by `query-type` and persist the subset to a new file.

```bash
python -m rmsearch.train.filter_query_recs \
  --input ./data/smollm-corpus/query_recs.json \
  --output ./data/smollm-corpus/filtered_query_recs.json \
  --filter questions
```

**Arguments**
- `--input`: Path to the JSON list produced by `make_query_recs.py`.
- `--output`: Destination JSON for the filtered records.
- `--filter`: `query-type` value to keep (default `questions`).

**Outputs**
- `{output}`: JSON list containing only entries whose `query-type` matches the provided filter.

**Notices**
- Provide a different `--filter` value (e.g. `titles`, `keywords`, `irr_questions`) to slice other subsets without regenerating queries.




## `get_top_relevant_keys_embed.py`

Embed queries and keys with vLLM, score them with dot-product similarity, and
store the top-N matches per query.

```bash
python -m rmsearch.train.get_top_relevant_keys_embed \
  --queries-json ./data/smollm-corpus/filtered_query_recs.json \
  --keys-csv ./data/smollm-corpus/df_small.csv \
  --key-column text \
  --model-name /workspace/e5-mistral7b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --k-key 100 \
  --similarity-device cuda \
  --output ./data/smollm-corpus/relevance_records_embed.json
```

**Arguments**
- `--queries-json` / `--queries-csv`: Query inputs. The JSON path should point to `filtered_query_recs.json` (or a similar list of objects containing at least a `"query"` field, with optional `df_id` and `query-type`).
- `--keys-json` / `--keys-csv`: Candidate key sentences; use `--key-json-field` / `--key-column` to pick the text field.
- `--model-name`, `--tensor-parallel-size`, `--num-instances`, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`: Embedding worker configuration passed to `vllm_embed`.
- `--query-batch-size`, `--key-batch-size`: Batch sizes for embedding calls.
- `--query-checkpoint`, `--key-checkpoint`: Optional JSONL checkpoints written during embedding.
- `--similarity-device`: Device used for the matrix multiply when ranking (default `cpu`; pass `cuda` for GPU).
- `--k-key`: Number of keys returned per query (default 50).
- `--correct-ids-json`: Optional gold key indices aligned with the query order.
- `--output`: Destination JSON for the relevance records (default `relevance_records_embed.json`).

**Outputs**
- JSON list mirroring the RM-based format with `query`, `query_id`, optional `df_id` / `query_type`, optional `correct_id`, and `"keys"` entries containing `key_id`, `key`, and cosine-like similarity scores.
- Optional embedding checkpoints if the related flags are supplied.

**Notices**
- Embeddings are pulled through the vLLM embedding API (see `rmsearch/utils/vllm_embed.py`); ensure the model exposes embedding heads.
- The similarity computation promotes tensors to the chosen device; large matrices may demand significant memory if you select `cuda`.
- Provide non-empty query and key inputs; the script validates and aborts otherwise.



## `sample_dpo_batch.py`

Sample pairs of relevant/df-sourced keys for DPO-style preference datasets.

```bash
python -m rmsearch.train.sample_dpo_batch \
  --relevance-json ./data/smollm-corpus/relevance_records_embed.json \
  --filtered-queries-json ./data/smollm-corpus/filtered_query_recs.json \
  --source-csv ./data/smollm-corpus/df.csv \
  --output ./data/smollm-corpus/sampled_query_key_set.json
```

**Arguments**
- `--relevance-json`: Optional path to the relevance records (RM or embedding variant). When omitted, two keys are uniformly sampled from the source CSV instead.
- `--filtered-queries-json`: Optional metadata lookup (e.g. `filtered_query_recs.json`) to recover `df_id` / `query-type`.
- `--source-csv`: DataFrame backing the df_id indices (defaults expect `df.csv`).
- `--source-column`: Column within the DataFrame containing the key text (default `text`).
- `--output`: Destination JSON for the sampled pairs (default `./data/smollm-corpus/sampled_query_key_set.json`).
- `--random-seed`: Sampling seed (default 42).

**Outputs**
- `{output}`: JSON list where each entry includes `query`, `query_id`, `keys`, `key_ids`, and the propagated `query-type` when available. When no relevance file is provided, a single placeholder query with two randomly sampled keys is emitted.
- Example:
```
[
  {
    "query_id": 0,
    "query": "...",
    "key_ids": [0,1],
    "keys": [
      "sentence1",
      "sentence2"
    ]
  }
]
```

**Notices**
- Sampling picks one key from the relevance results and one from the original df_id (when available); if no relevance file is supplied, two keys are drawn uniformly from the entire source CSV.



## `judge_dataset.py`

Collect pairwise relevance judgements for candidate sentences, producing the
reward-model preference dataset.

```bash
python -m rmsearch.train.judge_dataset \
  --query-key-set ./data/smollm-corpus/sampled_query_key_set.json \
  --model-name /workspace/qwen4b \
  --progress-dir relevant_file_progress \
  --max-model-len 10000 \
  --output ./exp1/dataset_list_train.json
```

**Arguments**
- `--query-key-set`: JSON generated by `sample_dpo_batch.py` containing query/key pairs (alias: `--query-key-s`).
- `--model-name`: Local vLLM model used to provide pairwise judgements.
- `--tokenizer-name`: Optional tokenizer name (defaults to `--model-name`).
- `--tensor-parallel-size`, `--num-instances`, `--gpu-memory-utilization`: Worker-pool configuration for `rmsearch.utils.vllm_generate`.
- `--max-model-len`, `--dtype`, `--trust-remote-code`: Optional model loader overrides passed to vLLM.
- `--batch-size`, `--temperature`, `--top-p`, `--max-tokens`, `--timeout-s`: Sampling controls for the pairwise judge prompts.
- `--progress-dir`: Optional directory for streaming checkpoints (raw judgements are written to `<progress-dir>/results.json`; leave unset to skip checkpointing).
- `--output`: Destination JSON for the assembled dataset list (default `dataset_list.json`).
- `--restart`: Resume from a previous run in `progress_dir` (requires `--progress-dir`).
- `--sample-pairs`: Number of sentence pairs sampled per query (useful when more than two keys exist).

**Outputs**
- `{output}`: Dataset list JSON suitable for DPO training, containing `chosen_msg`/`rejected_msg` pairs plus metadata.
- `{progress-dir}/results.json`: Raw judgements with prompts and model outputs for resumable execution.

**Notices**
- Reuses the same in-process vLLM worker pool (`rmsearch.utils.vllm_generate`) as `make_queries`; ensure the model fits into GPU memory.
- Random sampling means reruns without `--restart` may yield different pairings (when more than two keys per query are available).




## `lora_example.py`

Fine-tune a reward model using TRL's `RewardTrainer` with LoRA adapters.

```bash
python -m rmsearch.train.lora_example \
  --dataset-list-train ./exp1/dataset_list_train.json \
  --dataset-list-test ./exp1/dataset_list_test.json \
  --model-name /workspace/llama3b-rm \
  --output-dir ./exp1/model1 \
  --wandb-project rmsearch \
  --wandb-run-name exp1-lora
```

**Arguments**
- `--dataset-list-train`: Training preference pairs produced by `judge_dataset.py` (`dataset_list_train.json`).
- `--dataset-list-test`: Optional evaluation preference pairs (`dataset_list_test.json`). When omitted, training runs without evaluation.
- `--model-name`: Base reward model checkpoint or HF Hub path.
- `--output-dir`: Directory where LoRA checkpoints, logs, and tokenizer config are written.
- `--max-length`: Token limit applied during chat-template tokenisation (default `4000`).
- `--max-characters`: Character cap per message before tokenisation (default `4000`).
- `--per-device-train-batch-size` / `--per-device-eval-batch-size`: Batch sizes fed to TRL's `RewardTrainer`.
- `--evaluation-steps`, `--save-steps`, `--logging-steps`, `--num-train-epochs`: Standard TRL scheduling knobs.
- `--wandb-project`, `--wandb-run-name`, `--wandb-tags`: Enable Weights & Biases tracking for the run (omit the project to disable W&B entirely).

**Outputs**
- Saved checkpoints under `output-dir` (e.g. `checkpoint-XXXX`).
- `trainer_state.json` / `trainer_config.json` emitted by TRL in `output-dir`.
- When W&B is enabled, a run with the provided project/run name containing loss curves and evaluation metrics.
- Example dataset entry fed to TRL:
  ```json
  {
    "chosen_msg": [{"role": "user", "content": "...positive sentence..."}],
    "rejected_msg": [{"role": "user", "content": "...negative sentence..."}],
    "chosen_sentence_id": 12,
    "rejected_sentence_id": 45
  }
  ```

**Notices**
- Expects the base reward model weights and tokenizer to reside locally.
- Data is tokenised on the fly—no cached `train_ids`/`test_ids` or dataset directories are created.
- Adjust LoRA modules or training hyperparameters directly in `rmsearch/train/lora_example.py`.
- Long-running GPU job – monitor disk space for checkpoints and keep W&B logging disabled if offline.
