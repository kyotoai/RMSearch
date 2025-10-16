# Process arguana dataset for test dataset of RMSearch training

## Overview

1. `process_data`: download data from huggingface and save it to local volume
2. `make_query_recs`: from the data, make queries in records format about each data row. 
3. `get_top_relevant_keys_embed.py`: Get top-relevant rows from dataframe by embedding model.
4. `sample_dpo_batch.py`: Sample data rows from query_recs
5. `Direct sampled_query_key_set.json -> dataset_list_test.json`: From query_key_set created from `sample_dpo_batch.py`, make dpo dataset.
6. `lora_example.py`: Train reward model for the training.

* Running order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8


## Install rmsearch

```bash
git clone https://github.com/kyotoai/RMSearch.git
pip install -e RMSearch/
```


## `process_data`

Download a dataset from HuggingFace, shuffle it, and materialise convenient CSV
slices.

```bash
curl -L https://huggingface.co/datasets/mteb/arguana/resolve/main/corpus.jsonl -o data/arguana/corpus.jsonl
curl -L https://huggingface.co/datasets/mteb/arguana/resolve/main/queries.jsonl -o data/arguana/queries.jsonl
```

```bash
python -m rmsearch.train.process_data \
  --dataset-name mteb/arguana \
  --split test \
  --n-sample 100 \
  --output-dir ./data/arguana \
  --stream
```

```bash
python3 - <<'PY'
import json
import pandas as pd

df_corpus = pd.read_json('data/arguana/corpus.jsonl', lines=True)
df_queries = pd.read_json('data/arguana/queries.jsonl', lines=True)
df = pd.read_csv("./data/arguana/df.csv")

# Ensure matching dtypes for IDs (optional but safe)
df['_qid'] = df['query-id'].astype(str)
df['_cid'] = df['corpus-id'].astype(str)
q_map = df_queries.assign(_id=df_queries['_id'].astype(str)).set_index('_id')['text']
c_map = df_corpus.assign(_id=df_corpus['_id'].astype(str)).set_index('_id')['text']

df['query'] = df['_qid'].map(q_map)
df['corpus'] = df['_cid'].map(c_map)
df = df.drop(columns=['_qid', '_cid'])

df.to_csv("./data/arguana/df2.csv")
PY
```




## `make_query_recs`

```bash
python3 - <<'PY'
from pathlib import Path
import json
import pandas as pd

input_csv = "data/arguana/df2.csv"
output_path = Path("data/arguana/query_recs.json")

df = pd.read_csv(input_csv)

def _format_prompt(text: str) -> str:
  return (
      "Give me relevant score between query and sentence;\n\n"
      f"Query:{question}\n\n"
      f"Sentence:```{text}```"
  )

query_recs = []
for df_id, query in enumerate(df["query"].to_list()):

  query_recs.append(
      {
          "query": query,
          "df_id": df_id,
          "query-type": "arguana-normal",
      }
  )

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(query_recs, ensure_ascii=False, indent=2))
PY
```


## `get_top_relevant_keys_embed.py`

Embed queries and keys with vLLM, score them with dot-product similarity, and
store the top-N matches per query.

```bash
python -m rmsearch.train.get_top_relevant_keys_embed \
  --queries-json ./data/arguana/query_recs.json \
  --keys-csv ./data/arguana/df2.csv \
  --key-column corpus \
  --model-name /workspace/e5-mistral7b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --k-key 100 \
  --similarity-device cuda \
  --output ./data/arguana/relevance_records_embed.json
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
  --relevance-json ./data/arguana/relevance_records_embed.json \
  --filtered-queries-json ./data/arguana/query_recs.json \
  --source-csv ./data/arguana/df2.csv \
  --source-column corpus \
  --output ./data/arguana/sampled_query_key_set.json
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



## `Direct sampled_query_key_set.json -> dataset_list_test.json`

```bash
python3 - <<'PY'
from pathlib import Path
import json
query_key_set_path = "data/arguana/sampled_query_key_set.json"
output_path = Path("exp1/dataset_list_test.json")

with open(query_key_set_path) as f:
  query_key_set = json.load(f)

def _format_prompt(query: str, key: str) -> str:
  return (
      "Give me relevant score between query and sentence;\n\n"
      f"Query:{query}\n\n"
      f"Sentence:```{key}```"
  )

dataset_list = []
for query_key_dict in query_key_set:
  query_id = query_key_dict["query_id"]
  query = query_key_dict["query"]
  keys = query_key_dict["keys"]
  key_ids = query_key_dict["key_ids"]

  dataset_list.append(
      {
          "query_id": query_id,
          "chosen_msg": [{"role": "user", "content": _format_prompt(query, keys[1])}],
          "rejected_msg": [{"role": "user", "content": _format_prompt(query, keys[0])}],
          "chosen_sentence_id": key_ids[1],
          "rejected_sentence_id": key_ids[0],
      }
  )

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(dataset_list, ensure_ascii=False, indent=2))
print(f"Wrote dataset list with {len(dataset_list)} entries to {output_path}")
PY
```



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
- `--dataset-list-train`: The preference dataset produced by `judge_dataset.py` (`dataset_list_train.json`).
- `--dataset-list-test`: The preference dataset produced by `judge_dataset.py` (`dataset_list_test.json`).
- `--model-name`: Base reward model checkpoint.
- `--num-gpus`: Number of GPUs available for training (passed to `RMTrainer`).
- `--output-dir`: Directory where LoRA checkpoints and logs are written.
- `--base-dir`: Working directory for intermediate preprocessed datasets.

**Outputs**
- Checkpoints under `output-dir` (e.g. `checkpoint-XXXX`).
- Preprocessed dataset shards in `base-dir`.
- TRL training logs under `output-dir`.
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
- Training parameters mirror the notebook; adjust inside the script if you need different LoRA or training hyperparameters.
- Long-running GPU job – monitor disk space for checkpoints.
