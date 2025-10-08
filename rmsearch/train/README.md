# RMSearch Training Utilities

The scripts in this directory mirror the data and reward-model preparation
steps from `examples/train_en.ipynb`, but expose them as command-line tools so
you can run the pipeline outside the notebook. Each command writes the same
artifacts the notebook expects, and assumes you have a GPU-enabled environment
with the appropriate models available locally.

## Install rmsearch

```bash
git clone https://github.com/kyotoai/RMSearch.git
cd RMSearch
pip install .
```

## `process_data.py`

Download a dataset from HuggingFace, shuffle it, and materialise convenient CSV
slices.

```bash
python -m rmsearch.train.process_data \
  --dataset-name HuggingFaceTB/smollm-corpus \
  --output-dir ./data/smollm-corpus \
  --dataset-config cosmopedia-v2 \
  --n-sample 100000
```
Omit `--n-sample` entirely if you want to materialise the full split.

**Arguments**
- `--dataset-name`: HuggingFace dataset identifier.
- `--output-dir`: Directory where HF `dataset_dict.json` plus `df.csv` / `df_small.csv` are stored.
- `--n-sample`: Optional cap if you only want to persist a sampled subset (applies to the saved dataset and CSVs).
- `--dataset-config`: Optional configuration name if the dataset exposes multiple configs.
- `--split`: Dataset split to load (defaults to `train`).
- `--random-seed`: Shuffle seed.

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
- Large datasets may need additional disk space.

## `make_queries.py`

Generate titles, keywords, questions, and irrelevant questions for each source
sentence using an async vLLM engine.

```bash
python -m rmsearch.train.make_queries \
  --input-csv ./data/smollm-corpus/df_test.csv \
  --text-column text \
  --model-name /workspace/qwen7b \
  --output ./data/smollm-corpus/query_dict.json
```

**Arguments**
- `--input-csv`: CSV with the source sentences.
- `--text-column`: Column containing the text to analyse.
- `--output`: Where the generated query metadata is written as JSON.
- `--model-name`: Async vLLM model path/name.
- `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--gpu-memory-utilization`, `--omp-num-threads`: Engine resource settings; adjust to fit your GPU topology.
- `--max-requests`: Upper bound on concurrent async requests.
- `--progress-dir`: Directory used for on-disk checkpoints (`results.json`, `finished_ids.json`).
- `--restart`: Resume from an existing progress directory instead of starting fresh.

**Outputs**
- `{output}`: JSON mapping request indices to generated titles/keywords/questions/irrelevant questions.
- `{progress-dir}/results.json` and `finished_ids.json` for incremental restarts.
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
- Requires the generation model weights on local disk.
- Async engine uses GPU memory aggressively; tune `gpu_memory_utilization` if you see OOMs.
- If the async engine cannot start (for example on CPU-only hosts), the script now falls back to deterministic stub outputs so downstream steps continue; you can inspect the log message to confirm the fallback.
- For quick debugging, work with a small CSV (e.g. copy `df_small.csv` to `df_test.csv` with ~10 rows) before launching long vLLM runs.

## `judge_dataset.py`

Collect pairwise relevance judgements for candidate sentences, producing the
reward-model preference dataset.

```bash
python -m rmsearch.train.judge_dataset \
  --relevant-json ./data/smollm-corpus/sentences_relevant_to_questions.json \
  --model-name /workspace/qwen7b \
  --progress-dir relevant_file_progress7 \
  --output relevant_file_progress7/results.json
```

**Arguments**
- `--relevant-json`: List of candidate sentences per query (output of the retrieval stage).
- `--model-name`: Async vLLM model used to provide pairwise judgements.
- `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--gpu-memory-utilization`, `--omp-num-threads`: Async engine configuration.
- `--max-requests`: Maximum concurrent requests.
- `--progress-dir`: Directory for streaming checkpoints.
- `--output`: Destination JSON for the completed judgements (defaults to `<progress-dir>/results.json`).
- `--restart`: Resume from a previous run in `progress_dir`.
- `--sample-pairs`: Number of sentence pairs sampled per query.

**Outputs**
- `{output}`: JSON list of adjudicated comparisons with `sentence_ids`, `question`, and `output` fields.
- `{progress-dir}` retains intermediate state for resumable execution.
- Example record:
  ```json
  {
    "request_id": 7,
    "sentence_ids": [123, 987],
    "question": "Explain graph retrieval",
    "output": "<ID>1</ID>"
  }
  ```

**Notices**
- Requires the same async engine as `make_queries`; ensure the model fits into GPU memory.
- Random sampling means reruns without `--restart` may yield different pairings.

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
- The `progress_dir` checkpoints created by async stages (`make_queries`,
  `judge_dataset`) can be deleted once you no longer need to resume.
