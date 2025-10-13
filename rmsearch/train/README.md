# RMSearch Training Utilities

The scripts in this directory mirror the data and reward-model preparation
steps from `examples/train_en.ipynb`, but expose them as command-line tools so
you can run the pipeline outside the notebook. Each command writes the same
artifacts the notebook expects, and assumes you have a GPU-enabled environment
with the appropriate models available locally.



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


## Install rmsearch

```bash
git clone https://github.com/kyotoai/RMSearch.git
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
  --n-sample 10 \
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




## `make_queries.py`

Generate titles, keywords, questions, and irrelevant questions for each source
sentence using the local vLLM worker pool (`rmsearch.utils.vllm_generate`).

```bash
python -m rmsearch.train.make_queries \
  --input-csv ./data/smollm-corpus/df.csv \
  --text-column text \
  --model-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --max-model-len 10000 \
  --batch-size 8 \
  --output ./data/smollm-corpus/query_dict.json
```

**Arguments**
- `--input-csv`: CSV with the source sentences.
- `--text-column`: Column containing the text to analyse.
- `--output`: Where the generated query metadata is written as JSON.
- `--model-name`: Local vLLM model path/name.
- `--tokenizer-name`: Optional tokenizer identifier (defaults to `--model-name`).
- `--tensor-parallel-size`: Number of tensor parallel shards per worker.
- `--num-instances`: Number of worker processes to launch.
- `--gpu-memory-utilization`, `--max-model-len`, `--dtype`, `--trust-remote-code`: Options forwarded to `vllm.LLM`.
- `--batch-size`: Prompts per generation batch.
- `--temperature`, `--top-p`, `--max-tokens`: Sampling controls passed to `vllm.SamplingParams`.
- `--timeout-s`: Optional wall-clock timeout for the job.

**Outputs**
- `{output}`: JSON mapping request indices to generated titles/keywords/questions/irrelevant questions.
- Example entry:
  ```json
  {
    "42": {
      "titles": ["Graph Retrieval Overview"],
      "keywords": ["retrieval", "graph"],
      "questions": ["How does graph retrieval work?"],
      "irr_questions": ["What is your favourite cuisine?"]
    }
  }
  ```

**Notices**
- Requires the generation model weights on local disk; set `CUDA_VISIBLE_DEVICES` to pin GPUs.
- The helper constructs multiprocessing workers; ensure `num_instances * tensor_parallel_size` does not exceed available GPUs.
- When vLLM or the tokenizer cannot be loaded (e.g. CPU-only hosts), the script falls back to deterministic stub outputs so downstream steps continue—check the log for confirmation.
- For quick debugging, work with a small CSV (e.g. copy `df_small.csv` to `df_test.csv` with ~10 rows) before launching long vLLM runs.



## `make_query_recs.py`

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




## `get_top_relevant_keys_rm.py`

Traverse the tag tree with the reward model to score and retrieve the top-N keys
per query.

```bash
python -m rmsearch.train.get_top_relevant_keys_rm \
  --queries-json ./data/smollm-corpus/filtered_query_recs.json \
  --keys-csv ./data/smollm-corpus/df.csv \
  --key-column text \
  --tag-tree ./data/smollm-corpus/tag_tree_recs.json \
  --model-name ./llama3b-rm-converted-model \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --k-tag 2 \
  --k-key 10 \
  --output ./data/smollm-corpus/relevance_records_rm.json
```

**Arguments**
- `--queries-json` / `--queries-csv`: Query inputs. The JSON path should point to `filtered_query_recs.json` (or a similar list of objects containing at least a `"query"` field, with optional `df_id` and `query-type`).
- `--keys-json` / `--keys-csv`: Candidate key sentences; use `--key-json-field` / `--key-column` to pick the text field.
- `--tag-tree`: Base tag tree JSON; the script derives a `tag2key` structure via `assign_key_to_tag_tree`.
- `--tag2key-out`: Optional path to persist the generated tree annotated with `key_ids`.
- `--correct-ids-json`: Optional gold indices matching the query order.
- `--output`: Destination JSON for the relevance records (default `relevance_records_rm.json`).
- `--model-name`, `--tensor-parallel-size`, `--num-instances`, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`: Reward-model worker topology.
- `--k-tag`, `--k-key`: Branching factor per depth and number of keys returned per query.
- `--checkpoint`: Directory for caching reward-model search responses (both assignment and retrieval).

**Outputs**
- Relevance records describing the query text (plus any `df_id` / `query_type` metadata when present), optional `correct_id`, and the scored key list under `"keys"`.
- Optional `tag2key-out` file mirroring the tag tree but annotated with `key_ids`.

**Notices**
- Uses the same vLLM reward-model backend as `rmsearch.tree.search_key`; ensure the model fits the available GPUs.
- Provide either JSON or CSV inputs for queries/keys; the script errors if both variants are omitted.
- Checkpoint caching reuses prior reward-model responses to shorten reruns.




## `get_top_relevant_keys_embed.py`

Embed queries and keys with vLLM, score them with dot-product similarity, and
store the top-N matches per query.

```bash
python -m rmsearch.train.get_top_relevant_keys_embed \
  --queries-json ./data/smollm-corpus/filtered_query_recs.json \
  --keys-csv ./data/smollm-corpus/df_small.csv \
  --key-column text \
  --model-name intfloat/e5-mistral-7b-instruct \
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
  --relevance-json ./data/smollm-corpus/relevance_records_rm.json \
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
  --output ./exp1/dataset_list.json
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
  --dataset-list ./exp2/dataset_list.json \
  --model-name /workspace/llama3b-rm \
  --num-gpus 2 \
  --output-dir ./exp2/model1 \
  --base-dir ./exp2
```

**Arguments**
- `--dataset-list`: The preference dataset produced by `judge_dataset.py` (`dataset_list.json`).
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




## vLLM Serve (`openai/gpt-oss-20b`)

The gpt-oss models can be hosted with `vllm serve`, giving you an OpenAI-compatible endpoint that the RMSearch helpers can call. The first launch downloads the model weights, so ensure you have ~45 GB of free space (or pass `--download-dir` to choose a custom cache location).

**Install dependencies**

```bash
# create or reuse an isolated environment (gpt-oss wheels target Python 3.11 today)
uv python install 3.12
uv venv --python 3.12 .oss
source .oss/bin/activate  # Windows: .oss\Scripts\activate

# install serve + harmony dependencies inside the virtualenv
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

uv pip install openai-harmony
```

If you prefer system-wide installs, append `--system` to both `uv pip` commands. Running them without either a virtual environment or `--system` triggers `error: No virtual environment found`, which simply means `uv` could not detect an active environment.

If dependency resolution fails with:

```
Because there is no version of torch==2.9.0.dev...+cu128 ...
```

it means your Python version does not match the available PyTorch nightly wheels. Recreating the environment with Python 3.11 (shown above) resolves the mismatch because the gpt-oss builds currently depend on the CUDA 12.8 nightly for Python 3.11.

**Launch the server**

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0          # match Harmony sampler expectations
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 5000 \
  --gpu-memory-utilization 0.95
```

The server exposes the OpenAI REST API at `http://127.0.0.1:8000/v1`. Set `OPENAI_API_KEY` (the value is arbitrary for local servers; `EMPTY` works) if you want the helpers to pick it up automatically.

**Query with `rmsearch.utils.vllm_serve_generate`**

```python
import os

from vllm import SamplingParams
from rmsearch.utils.vllm_serve_generate import build_llm, generate

serve = build_llm(
    model_name="openai/gpt-oss-20b",
    tensor_parallel_size=1,
    num_instances=1,
    endpoint_url="http://127.0.0.1:8000/v1",
    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    request_timeout=180.0,
)

try:
    prompts = ["Always answer like a riddle.\nUser: What is the weather in SF?"]
    outputs = generate(serve, prompts, sampling_params=SamplingParams(max_tokens=128))
    print(outputs[0])
finally:
    serve.close()
```

**Structured Harmony conversations (optional)**

When you need fine-grained control over token IDs (e.g. Harmony prefill/stop tokens), work directly with the Python `vllm.LLM` API instead of the HTTP bridge. The sample below mirrors the offline script recommended by vLLM:

```python
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from vllm import LLM, SamplingParams

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Always respond in riddles"),
        ),
        Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
    ]
)

prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

llm = LLM(
    model="openai/gpt-oss-20b",
    trust_remote_code=True,
    gpu_memory_utilization=0.95,
    max_num_batched_tokens=4096,
    max_model_len=5000,
    tensor_parallel_size=1,
)

outputs = llm.generate(
    prompt_token_ids=[prefill_ids],
    sampling_params=SamplingParams(max_tokens=128, stop_token_ids=stop_token_ids),
)
```

Decode the resulting token IDs with `encoding.parse_messages_from_completion_tokens` when you need structured tool-call messages.




## Utility Modules Overview

- `rmsearch/utils/vllm_serve_generate.py`: Client for an external `vllm serve` process. `build_llm` connects to an HTTP endpoint; `generate` mirrors the local batching API while issuing OpenAI-compatible completion requests.
- `rmsearch/utils/vllm_generate.py`: In-process vLLM worker pool that spawns subprocesses, ideal when you have the checkpoint on the same machine and want tight integration.
- `rmsearch/utils/vllm_reward.py`: Embedding-oriented worker pool with helper routines for reward modelling; refer to the module docstring for checkpointing hooks.
- `rmsearch/utils/vllm_embed.py`: Convenience wrapper that swaps generation for pooling to obtain dense embeddings via the same multi-worker infrastructure.




## General Notes

- All scripts expect GPU acceleration and local access to the corresponding
  model weights.
- When running multiple steps consecutively, reuse the same working directory
  layout as the notebook (`/workspace/RMS_exp/data/<name>`), or adjust paths
  consistently.
- The `progress_dir` checkpoints created by generator-backed stages (`make_queries`,
  `judge_dataset`) can be deleted once you no longer need to resume.
