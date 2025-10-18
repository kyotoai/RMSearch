# RMSearch Training Utilities

## Overview

1. Make sure that you have dataset_list_train.json & dataset_list_test.json
2. Login to wandb (API key is in discord->general->pinned messages)
3. Run `lora_example.py`

## Login to wandb (-> web service to organize the traini)

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

## `lora_example.py`

Fine-tune a reward model using TRL's `RewardTrainer` with LoRA adapters.

```bash
python -m rmsearch.train.lora_example \
  --dataset-list-train ./exp1/dataset_list_train.json \
  --dataset-list-test ./exp1/dataset_list_test.json \
  --model-name /workspace/llama3b-rm \
  --output-dir ./exp1/model1 \
  --wandb-project rmsearch \
  --wandb-run-name example-lora
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



## General Notes

- All scripts expect GPU acceleration and local access to the corresponding
  model weights.
- When running multiple steps consecutively, reuse the same working directory
  layout as the notebook (`/workspace/RMS_exp/data/<name>`), or adjust paths
  consistently.
- The `progress_dir` checkpoints created by generator-backed stages (`make_queries`,
  `judge_dataset`) can be deleted once you no longer need to resume.
