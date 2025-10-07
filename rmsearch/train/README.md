# RMSearch Training Utilities

The scripts in this directory mirror the data and reward-model preparation
steps from `examples/train_en.ipynb`, but expose them as command-line tools so
you can run the pipeline outside the notebook. Each command writes the same
artifacts the notebook expects, and assumes you have a GPU-enabled environment
with the appropriate models available locally.

## `process_data.py`

Download a dataset from HuggingFace, shuffle it, and materialise convenient CSV
slices.

```bash
python -m rmsearch.train.process_data \
  --dataset-name hatakeyama-llm-team/japanese2010 \
  --output-dir ./data/smollm-corpus \
  --n-sample-train 100000 \
  --n-small-sample 10000
```

**Arguments**
- `--dataset-name`: HuggingFace dataset identifier.
- `--output-dir`: Directory where HF `dataset_dict.json` plus `df.csv` / `df_small.csv` are stored.
- `--dataset-config`: Optional configuration name if the dataset exposes multiple configs.
- `--split`: Dataset split to load (defaults to `train`).
- `--n-sample-train`: Maximum rows kept for training.
- `--n-sample-test`: Held-out evaluation rows (saved under the same directory).
- `--n-small-sample`: Size of the quick iteration CSV (`df_small.csv`).
- `--random-seed`: Shuffle seed.

**Outputs**
- `<output-dir>/dataset_dict.json` (HF binary format when `datasets` is installed).
- `<output-dir>/df.csv` full sample, `<output-dir>/df_small.csv` subset.

**Notices**
- Requires `datasets` for real downloads; otherwise a stub CSV is produced.
- Large datasets may need additional disk space.

## `make_queries.py`

Generate titles, keywords, questions, and irrelevant questions for each source
sentence using an async vLLM engine.

```bash
python -m rmsearch.train.make_queries \
  --input-csv ./data/smollm-corpus/df_small.csv \
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

**Notices**
- Requires the generation model weights on local disk.
- Async engine uses GPU memory aggressively; tune `gpu_memory_utilization` if you see OOMs.

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

**Notices**
- Expects the base reward model weights and tokenizer to reside locally.
- Training parameters mirror the notebook; adjust inside the script if you need different LoRA or training hyperparameters.
- Long-running GPU job – monitor disk space for checkpoints.

## General Notes

- All scripts expect GPU acceleration and local access to the corresponding
  model weights.
- When running multiple steps consecutively, reuse the same working directory
  layout as the notebook (`/workspace/RMS_exp/data/<name>`), or adjust paths
  consistently.
- The `progress_dir` checkpoints created by async stages (`make_queries`,
  `judge_dataset`) can be deleted once you no longer need to resume.
