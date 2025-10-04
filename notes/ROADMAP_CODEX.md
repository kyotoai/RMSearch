# RMSearch Codex Roadmap

## Release Milestones
- **M0 – Stabilise core search (Weeks 1–2)**
  - [ ] Validate async search against small benchmark set and document accuracy vs. baseline embedding search.
  - [ ] Replace ad-hoc prompt trimming (`prompt[17:]`) with template-aware preprocessing.
- **M1 – Ship reproducible training loop (Weeks 3–4)**
  - [ ] Publish a scripted pipeline that prepares datasets, fine-tunes a reward model, converts it, and evaluates retrieval quality end-to-end.
  - [ ] Package model artifacts and configs for multi-GPU inference (AsyncLLMEngine + vLLM worker pool).
- **M2 – Graph-enhanced retrieval (Weeks 5–6)**
  - [ ] Productionise `generate_tag_graph.py` (config files, automation, monitoring) and integrate outputs into the search evaluation workflow.
  - [ ] Run scalability study comparing tag-graph search vs. vanilla RMSearch (capture throughput & retrieval quality).

## Search Engine (`rmsearch/rmsearch.py`)
- [ ] Refactor `scheduling_strategy_fn` to read `self.tensor_parallel_size` so Ray placement groups match the instantiated engine configuration.
- [ ] Add batching utilities to `get_relevance` / `search_by_df` to avoid materialising large query×key grids in memory.
- [ ] Formalise the prompt templating contract: accept callable or template object, handle tokenizer prefix stripping generically.
- [ ] Surface instrumentation hooks (timings, token counts, GPU usage) for downstream monitoring.
- [ ] Provide synchronous convenience wrapper that hides asyncio for notebook users.

## Reward Model Training (`rmsearch/rmtrain.py`)
- [ ] Parameterise dataset schema (allow custom column names + metadata) and emit schema versioning in cached datasets.
- [ ] Extend `prepare_dataset` to support preference pairs stored in Arrow/Parquet without loading entire dataset into RAM.
- [ ] Create evaluation callbacks to compute offline retrieval metrics during training.
- [ ] Document LoRA layer selections and expose config knobs for different backbone sizes.
- [ ] Add utilities to export merged adapters + score head for deployment.

## Model Conversion & Packaging (`rmsearch/utils.py`, `setup.py`)
- [ ] Generalise `convert_model` to detect score head attribute names automatically (currently assumes `score`).
- [ ] Implement `revert_model` for round-tripping converted checkpoints.
- [ ] Replace placeholder console entry point (`my-tool`) with a real CLI for search/training automation.
- [ ] Write packaging docs (PyPI/Torch hub) once tests exist.

## Evaluation & Data Assets
- [ ] Reconstruct the evaluation baseline described in `notes/kentarrito.md` with automated scripts (nDCG@k, hit@k, rerank comparison to embedding search).
- [ ] Version and publish the `sentences_relevant_to_questions.json` / `relevance_dict.json` artefacts referenced in `notes/COMUNICATION.md`.
- [ ] Integrate tag graph outputs into evaluation so we can measure improvements over naive agent retrieval.
- [ ] Stand up regression tests that compare new checkpoints against frozen validation sets.

## Infrastructure & Performance
- [ ] Capture DeepSpeed setup (from `examples/deepspeed_test2.py` + `notes/deepspeed.md`) into reproducible configs, ideally with `accelerate` launcher support.
- [ ] Evaluate AsyncLLMEngine vs. custom vLLM worker pools for latency/throughput; converge on one abstraction for production.
- [ ] Containerise the end-to-end pipeline (Docker + optional RunPod template) with GPU requirements documented.
- [ ] Add CI smoke tests (lint, type check, lightweight CPU-only unit tests).

## Documentation & Developer Experience
- [ ] Backfill `tests/test.py` with meaningful unit coverage (tokenisation formatter, dataset split caching, convert_model, search ranking top-k).
- [ ] Convert key notebooks into scripted examples or docs, noting any manual steps that remain.
- [ ] Incorporate figures from `images/` into a concise project overview slide for onboarding.
- [ ] Maintain changelog / weekly update in `notes/` to replace ad-hoc communication threads.

## Research Questions & Open Decisions
- How should RMSearch blend reward scores with classical embedding similarity for cold-start queries?
- What heuristics decide the number of tag clusters (per layer) for agent retrieval?
- Can we reuse the reward model score head for other tasks (reranking generated reasoning steps, summarisation feedback)?
- What is the minimal GPU footprint to serve AsyncLLMEngine with acceptable latency for agentic search chains?
