# Process arguana dataset for test dataset of RMSearch training

## Overview

1. `process_data`: download data from huggingface and save it to local volume
2. `make_query_recs`: from the data, make queries in records format about each data row. 
3. `get_top_relevant_keys_embed.py`: Get top-relevant rows from dataframe by embedding model.
4. `sample_dpo_batch.py`: Sample data rows from query_recs
5. `Direct sampled_query_key_set.json -> dataset_list_test.json`: From query_key_set created from `sample_dpo_batch.py`, make dpo dataset.
6. `adpo_lora_example.py`: Train reward model for the training.

* Running order: 1 -> 2 -> 3 -> 4 -> 5 -> 6


## Install rmsearch

```bash
git clone https://github.com/kyotoai/RMSearch.git
pip install -e RMSearch/
```


## `process_data`

Download a dataset from HuggingFace, shuffle it, and materialise convenient CSV
slices.

```bash
python -m rmsearch.train.process_data \
  --dataset-name HuggingFaceTB/smollm-corpus \
  --output-dir ./data/smollm-corpus6000 \
  --dataset-config cosmopedia-v2 \
  --n-sample 6000 \
  --stream
```

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


## `make_query_and_less_relevant_keys_recs.py`

Generate a dataset with query and less_relevant_keys. 

```bash
python -m rmsearch.train.make_query_and_less_relevant_keys_recs \
  --input-csv ./data/smollm-corpus/df.csv \
  --text-column text \
  --model-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --n-key-generation 5 \
  --batch-size 8 \
  --max-model-len 10000 \
  --output ./data/smollm-corpus/query_and_less_relevant_keys_recs.json
```


```bash
python -m rmsearch.train.make_query_and_less_relevant_keys_recs-gptoss \
  --input-csv ./data/smollm-corpus/df.csv \
  --text-column text \
  --model-name /workspace/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --n-key-generation 5 \
  --batch-size 8 \
  --max-model-len 10000 \
  --output ./data/smollm-corpus/make_query_and_less_relevant_keys_recs-gptoss.json
```


**Algorithm**
1. Load all keys from df.csv
2. Generate 1 query for every key in df.csv text-column and generate n_key_generation less_relevant_keys to the query which are ranged from a slightly less relevant key to the query than the original key to more irrelevant key. Follow
  * It should generate a different type of query which is very relevant to query. Title, question, 1 sentence, 2~3 sentences, 1 paragraph, 2~3 paragraphs. Make 10 instructions for the difference.
  * You should make n_key_generation less relevant keys in different ways so that the dataset become not biased. Ex. by changine some element of the core sentence of the relevance with the query in the original key, by totally recreating the original key, by inserting some fake sentences, and ... . Make 10 instructions for the difference.
  * For each LLM generation prompt, pick an instruction pair in turn from 100 query & key possibilities (not randomly picking one but picking them in turn) and make the prompt.
  * IN THE PROMPT, MAKE SURE THAT THE KEYS RELEVACE BECOME: keys[0] > keys[1] > ... > keys[n_key_generation-1]. To make the dataset, this is the most important condition. Second, be sure to generate keys which are also relevant to the query to some extent. Don't make some totally irrelevant keys. It's prefarable for the generated keys to contain important terms but the content is slightly off to the query. 

**Arguments**
- Inherits the same CLI as `make_query_recs.py`; see above for detailed flag descriptions.

**Outputs**
- `{output}`: JSON list where each element contains `query`, `df_id`, and `query-type`, covering every generated title/keyword/question/irrelevant question.
- Example entry:
  ```json
  [
    {"query": "Graph Retrieval Overview", "correspond_key":"...", "less_relevant_keys": ["(a slightly less relevant key)", ... , "(less relevant key to some extent)"], "df_id": 42, "query-type": "titles"},
    {"query": "How does graph retrieval work?", "correspond_key":"...", "less_relevant_keys": "(a slightly less relevant key)", ... , "(less relevant key to some extent)"], "df_id": 33, "query-type": "questions"}
  ]
  ```

**Notices**
- Shares batching, sampling, and fallback behaviour with `make_query_recs.py`; refer to that section for runtime considerations.
- Use the same method for LLM generation as `make_query_recs.py`.






## `make_query_dpo_pairs.py`

Generate a dataset with relevance-varient queries for a key. 

```bash
python -m rmsearch.train.make_query_dpo_pairs \
  --input-csv ./data/smollm-corpus/df.csv \
  --text-column text \
  --model-name /workspace/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --n-query-generation 5 \
  --batch-size 20 \
  --max-model-len 10000 \
  --output ./data/smollm-corpus/query_dpo_pairs.json
```

```bash
nohup python -m rmsearch.train.make_query_dpo_pairs_v2 \
  --input-csv ./data/smollm-corpus6000/df.csv \
  --text-column text \
  --model-name /workspace/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --num-instances 3 \
  --n-query-generation 5 \
  --batch-size 20 \
  --max-model-len 10000 \
  --output ./data/smollm-corpus6000/query_dpo_pairs_easy.json \
  > >(tee ./data_generate.log) 2>&1 &
```


**Algorithm**
1. Load all keys from df.csv
2. For every key in df.csv text-column, generate n_query_generation queries ranged from the most relevant query to the key to more irrelevant key. Follow
  * It should generate a different type of queries which varies in terms of their relevance to the key. Randomly pick query type from title/question/single-sentence/several-sentences/single-paragraph/sevral-paragraphs.
  * Make 4 different instructions for making various queries. For each query-type, offer 4 different ways to generate the query. Ex. "Write a concise 5-7 word title capturing the key's core concept with active wording." You can refer to QUERY_VARIATIONS in make_query_and_less_relevant_keys_recs.py.
  * Pick query-type and 4 different instructions randomly and make n_query_generation queries.
  * IN THE PROMPT, MAKE SURE THAT THE QUERIES RELEVACE TO THE KEY BECOME: queries[0] > queries[1] > ... > queries[n_query_generation-1]. To make the qualitative dataset, this is the most important condition. Second, be sure to generate relevant queries, all of which are also relevant to the key to some extent. But queries[i] is slightly more relevant to the key in terms of their relevance to the deep meaning of the key.

**Arguments**
- Inherits the same CLI as `make_query_recs.py`; see above for detailed flag descriptions.

**Outputs**
- `{output}`: JSON list where each element contains `queries`, `key`, `df_id`, and `query-type`, covering every generated title/question/single-sentence/several-sentences/single-paragraph/sevral-paragraphs query.
- Example entry:
  ```json
  [

    {"queries": ["(The most relevant query)", "(Slightly less relevant query)", ... , "(The most irrelevant key)] , "key":"...", "df_id": 42, "query-types": ["title", "question", ...]},

    {"queries": ["(The most relevant query)", "(Slightly less relevant query)", ... , "(The most irrelevant key)] , "key":"...", "df_id": 33, "query-types": ["question", "title", ...]},

    ...
  ]
  ```

**Notices**
- Use the same method for LLM generation as `make_query_and_less_relevant_keys_recs-gptoss,py`.




## `Direct query_dpo_pairs.json -> dataset_list_train.json, dataset_list_test.json`

```bash
python3 - <<'PY'
from pathlib import Path
import json
query_dpo_pairs_path = "data/smollm-corpus/query_dpo_pairs.json"
output_path_train = Path("exp5/dataset_list_train.json")
output_path_test = Path("exp5/dataset_list_test.json")
test_size = 50
n_query = 5

with open(query_dpo_pairs_path) as f:
  query_dpo_pairs = json.load(f)

def _format_prompt(query: str, key: str) -> str:
  return (
      "Give me relevant score between query and sentence;\n\n"
      f"Query:{query}\n\n"
      f"Sentence:```{key}```"
  )

dataset_list = []
n_error = 0
for ds_id, query_dpo_pairs_dict in enumerate(query_dpo_pairs):
  try:
    queries = query_dpo_pairs_dict["queries"]
    key = query_dpo_pairs_dict["key"]
    df_id = query_dpo_pairs_dict["df_id"]
    query_types = query_dpo_pairs_dict["query-types"]

    batch = []
    for query_id, query in enumerate(queries):
      batch.append({"msg": [{"role": "user", "content": _format_prompt(query, key)}], "query_id":query_id, "key_id":df_id, "ds_id":ds_id})

    n_queries = len(queries)

    if n_queries < n_query:
      raise Exception

    dpo_pairs = []
    for c_id in range(n_queries - 1):
      for r_id in range(c_id + 1, n_queries):
        dpo_pairs.append([c_id, r_id])

    dataset_list.append(
        {
            "batch": batch,
            #[
            #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[1])}], "query_id":query_id, "key_id":},
            #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[0])}], "query_id":query_id, "key_id":},
            #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[0])}], "query_id":query_id, "key_id":}
            #],
            "dpo_pairs": dpo_pairs,
            #[
            #  [0,1],  # [(chosen_msg_id), (rejected_msg_id)]
            #  [0,2],
            #  [1,2]
            #]
        }
    )
  
  except Exception as e:
    n_error += 1
    print(e)

print("n_error: ", n_error)

dataset_list_train = dataset_list[:-test_size]
dataset_list_test = dataset_list[-test_size:]

print("len(dataset_list_train): ", len(dataset_list_train))
print("len(dataset_list_test): ", len(dataset_list_test))

output_path_train.parent.mkdir(parents=True, exist_ok=True)
output_path_test.parent.mkdir(parents=True, exist_ok=True)
output_path_train.write_text(json.dumps(dataset_list_train, ensure_ascii=False, indent=2))
output_path_test.write_text(json.dumps(dataset_list_test, ensure_ascii=False, indent=2))
print(f"Wrote dataset list with {len(dataset_list)} entries to {output_path_train} and {output_path_test}")
PY
```



## `adpo_lora_example.py`

Fine-tune a reward model using TRL's `RewardTrainer` with LoRA adapters.

* Login to wandb (-> web service to organize the traini)

  * Wandb home: https://wandb.ai/home
  * Ref: https://docs.wandb.ai/quickstart/

  1. set api key
  ```
  export WANDB_API_KEY=<your_api_key>
  ```

  2. login by command
  ```
  wandb login
  ```

```bash
python -m rmsearch.train.adpo_lora_example \
  --dataset-list-train ./exp2/dataset_list_train.json \
  --dataset-list-test ./exp2/dataset_list_test.json \
  --model-name /workspace/llama3b-rm \
  --output-dir ./exp2/model1 \
  --wandb-project rmsearch \
  --wandb-run-name exp2-adpo-lora
```

## With Accelerate (For Multi GPU)


### Multi-GPU note

✅ Works with 2× GPU for training + eval.

❌ With >2 GPUs on some containers, it hangs on NCCL.

🔁 Workaround: change backend from NCCL → GLOO in the training script:

```python
import torch.distributed as dist
dist.init_process_group("gloo")
```

After switching to GLOO: training runs, but eval still throws tensor-shape errors (not solved yet).

🟢 Simplest stable option right now: use A100 GPUs / containers that already have NCCL distributed set up correctly.

## Accelerate (multi-GPU friendly baseline)

```bash
nohup accelerate launch --config_file ./accelerate_config.yaml \
  -m rmsearch.train.adpo_lora_example \
  --dataset-list-train ./exp2/dataset_list_train.json \
  --dataset-list-test ./exp2/dataset_list_test.json \
  --model-name /workspace/data/llama3b-rm \
  --output-dir ./exp2/model1 \
  > >(tee ./train.log) 2>&1 &
```

## for qwen3 4b Reranker
```bash
nohup accelerate launch --config_file ./accelerate_config.yaml \
  -m rmsearch.train.adpo_lora_example \
  --dataset-list-train ./exp2/dataset_list_train.json \
  --dataset-list-test ./exp2/dataset_list_test.json \
  --model-name /workspace/qwen4b-reranker/ \
  --output-dir ./exp2/model1 \
  --wandb-project rmsearch \
  --wandb-run-name exp2-adpo-lora-qwen4b \
  > >(tee ./train.log) 2>&1 &

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
