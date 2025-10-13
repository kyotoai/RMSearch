# Useful prompts for codex

## README -> code files

```
Refering to rmsearch/---/README.md, make files in ---/ folder. If there is something ambiguous or some logical conflict in the readme, ask some questions to me before you make the files. Also update readme by adding more sentences or improving text for clearer expressions and less ambiguity. Don't change or delete lines in readme so much.
```


## README sample

## Overview

1. `process_data.py`: download data from huggingface and save it to local volume
2. `make_queries.py`: from the data, make queries in dictionary format about each data row. (Old and not used anymore)
3. `make_query_recs.py`: from the data, make queries in records format about each data row. 
4. `filter_query_recs.py`: Filter `query_recs.json` by `query-type` and persist the subset to a new file.
5. `get_top_relevant_keys_rm.py`: Get top-relevant rows from dataframe by reward model. Need to create tag_tree_recs.json by following rmsearch/tree/README.md
6. `get_top_relevant_keys_embed.py`: Get top-relevant rows from dataframe by embedding model.
7. `sample_dpo_batch.py`: Sample data rows from query_recs
8. `judge_dataset.py`: From query_key_set created from `sample_dpo_batch.py`, make dpo dataset by judging which key is more relevant to query.
9. `lora_example.py`: Train reward model for the training.

* Running order: 1 -> 3 -> 4 -> (5 or 6) -> 7 -> 8 -> 9

## `generate_tag.py`

Generate candidate tags for each key using a vLLM generation worker pool.

```bash
python -m rmsearch.tree.generate_tag \
  --keys-file ./data/smollm-corpus/df.csv \
  --key-column text \
  --model-name /workspace/qwen4b \
  --output ./data/smollm-corpus/tag_records.json \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --max-model-len 10_000
```

**Arguments**
- `--keys-file`: Plain-text file with one key (sentence/title) per line.
- `--key-column`: Column name of keys_file if it's a csv file.
- `--output`: Destination JSON file for tag records.
- `--model-name`: Generation model checkpoint.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Control worker topology; `--device-groups` accepts strings like `"0,1;2,3"`.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling and scheduling knobs.
- `--gpu-memory-utilization`, `--max-model-len`, `--dtype`, `--trust-remote-code`: Options forwarded to `vllm.LLM`.

**Outputs**
- `tag_records.json`: List of dictionaries with `key`, `key_id`, and generated `tags`.
- Example entry:
  ```json
  {
    "key": "Graph-based retrieval augmentation",
    "key_id": 0,
    "tags": ["graph retrieval", "enterprise search", "augmentation"]
  }
  ```

**Notices**
- The script creates and tears down the worker pool in-process. Ensure CUDA
  visibility matches the provided device groups.

