# RMSearch vLLM Utilities

This directory hosts interchangeable helpers for running RMSearch workloads on top of vLLM. Each module exposes a consistent `build_llm(...)` factory plus a `generate(...)` (or equivalent) convenience wrapper so downstream code can swap execution modes without code churn.

## `vllm_generate.py`
- **Purpose:** Launches local vLLM engines inside worker subprocesses. Designed for high-throughput text generation when the model weights live on the same machine.
- **Key pieces:**
  - `LLMWorkerModel`: pools dedicated processes, one per GPU group, with round-robin batching, live heartbeat logging, and graceful shutdown.
  - `build_llm(...)`: infers GPU topology (`CUDA_VISIBLE_DEVICES` or `torch.cuda.device_count`) and maps workers to devices; accepts standard vLLM engine kwargs (`max_model_len`, `gpu_memory_utilization`, etc.).
  - `generate(...)`: splits prompts into batches, dispatches them to workers, and rejoins outputs in call order.
- **Typical use:** Creation of query expansions, judgement prompts, or other generation-intensive steps within the training pipeline.

## `vllm_reward.py`
- **Purpose:** Embedding and reward-model evaluation pipeline that mirrors `vllm_generate.py` but tailored for pooling operations and dataset streaming.
- **Highlights:**
  - Integrates TRL/transformers helpers (`AutoTokenizer`, `datasets.Dataset`, `pandas`) for reward-model preprocessing.
  - Adds checkpoint appenders so long-running runs can resume safely.
  - Expands the worker loop with `PoolingParams` and optional JSONL logging of batch progress.
- **Typical use:** Building preference datasets, embedding sentence pairs, and scoring candidates during reward-model training.

## `vllm_embed.py`
- **Purpose:** Lightweight adapter that converts the generation worker pool into an embedding service by swapping `llm.generate` for `llm.encode`. Maintains identical batching and worker semantics to keep call sites simple.
- **Typical use:** Producing dense vector representations of documents or queries for retrieval experiments.

## `vllm_serve_generate.py`
- **Purpose:** Connects to a remote or locally-hosted `vllm serve` instance (e.g., `openai/gpt-oss-20b`) via the OpenAI-compatible REST API. This is the easiest path when another process is already hosting the model.
- **Key pieces:**
  - `LLMServeModel`: thin HTTP client that respects the same interface as `LLMWorkerModel`. Handles health checks, timeout budgeting, exponential backoff, and response parsing.
  - `build_llm(...)`: accepts the same signature as local helpers but forwards configuration to the HTTP endpoint (`endpoint_url`, `api_key`, `request_timeout`, etc.) while translating any sampling defaults.
  - `generate(...)`: batches prompts, translates `SamplingParams` into OpenAI payloads, and aggregates completions back into the original order.
- **Typical use:** Delegating inference to a standalone server process, Docker container, or remote node without changing RMSearch orchestration code.

### Choosing a helper
- Use `vllm_generate.py` when you control GPU allocation locally and want maximum throughput.
- Use `vllm_serve_generate.py` when an OpenAI-compatible server already exposes the model (or when sharing it across multiple clients).
- Use `vllm_reward.py` or `vllm_embed.py` for specialised reward scoring or embedding workloads that need streaming checkpoints or vector outputs.

All modules honour `SamplingParams` from vLLM and return plain text or embeddings in positional order, making it straightforward to slot them into the broader RMSearch data and training pipeline.
