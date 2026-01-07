"""LoRA reward-model training helpers with built-in dataset prep and W&B logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from .utils import extract_int, extract_text

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

__all__ = ["make_dataset_list", "train_reward_model"]

_PROMPT_TEMPLATE = (
    "Give me relevant score between query and sentence;\n\n"
    "Query:{query}\n\n"
    'Sentence:```{sentence}```'
)


def _truncate_message(content: str, limit: int) -> str:
    text = str(content)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# --- replace the old helper that returned list[dict] with this one ---
def _format_preference_pair_standard(
    example: Dict[str, object],
    tokenizer,
    *,
    max_characters: int,
) -> Dict[str, object]:
    """
    Build TRL-standard preference fields as STRINGS:
      {"chosen": "<templated text>", "rejected": "<templated text>"}

    TRL RewardTrainer expects columns 'chosen' and 'rejected' and does tokenization internally.
    Docs: https://huggingface.co/docs/trl/en/reward_trainer  (Dataset formats - Preference)
    """
    chosen_msg = example.get("chosen_msg")
    rejected_msg = example.get("rejected_msg")

    if not isinstance(chosen_msg, list) or not isinstance(rejected_msg, list):
        raise ValueError("Both 'chosen_msg' and 'rejected_msg' must be chat message lists.")

    trimmed_chosen = [
        {**m, "content": _truncate_message(m.get("content", ""), max_characters)}
        for m in chosen_msg
    ]
    trimmed_rejected = [
        {**m, "content": _truncate_message(m.get("content", ""), max_characters)}
        for m in rejected_msg
    ]

    # Convert messages -> single templated strings (NOT tokenized yet)
    chosen_text = tokenizer.apply_chat_template(
        trimmed_chosen, tokenize=False, add_generation_prompt=False
    )
    rejected_text = tokenizer.apply_chat_template(
        trimmed_rejected, tokenize=False, add_generation_prompt=False
    )

    out = {
        "chosen": chosen_text,
        "rejected": rejected_text,
    }
    if "chosen_sentence_id" in example:
        out["chosen_sentence_id"] = example["chosen_sentence_id"]
    if "rejected_sentence_id" in example:
        out["rejected_sentence_id"] = example["rejected_sentence_id"]
    return out


# --- replace your _build_dataset_split with this version ---
def _build_dataset_split(
    records: Sequence[Dict[str, object]],
    tokenizer,
    *,
    max_length: int = 4000,   # forwarded via RewardConfig; not used here directly
    max_characters: int,
) -> Optional[Dataset]:
    if not records:
        return None

    print(f"[prep] received {len(records)} raw preference records")
    ds_in = Dataset.from_list(list(records))
    print(f"[prep] input columns: {ds_in.column_names}")

    def convert_row(example: Dict[str, object]) -> Dict[str, object]:
        return _format_preference_pair_standard(
            example, tokenizer, max_characters=max_characters
        )

    ds = ds_in.map(
        convert_row,
        desc="[prep] building standard 'chosen'/'rejected' strings",
    )

    keep_columns = {"chosen", "rejected", "chosen_sentence_id", "rejected_sentence_id"}
    to_remove = [c for c in ds.column_names if c not in keep_columns]
    if to_remove:
        ds = ds.remove_columns(to_remove)

    print(f"[prep] final columns: {ds.column_names}")
    print(f"[prep] dataset size: {ds.num_rows}")

    # Show a tiny peek so we know we actually have strings
    try:
        ex = ds[0]
        print("[prep] first row keys:", list(ex.keys()))
        print("[prep] chosen[:160]:", (ex["chosen"][:160] + "...") if isinstance(ex.get("chosen"), str) else type(ex.get("chosen")))
        print("[prep] rejected[:160]:", (ex["rejected"][:160] + "...") if isinstance(ex.get("rejected"), str) else type(ex.get("rejected")))
    except Exception as e:
        print(f"[prep] sample print failed (non-fatal): {e}")

    return ds

def make_dataset_list(
    results: Sequence[Dict[str, object]],
    *,
    sentences: Sequence[str],
) -> List[Dict[str, object]]:
    """Convert judge outputs into chat-format preference pairs (implicit prompt)."""

    dataset_list: List[Dict[str, object]] = []

    for result in results:
        output = str(result.get("output", ""))
        sentence_ids = list(result.get("sentence_ids", []))
        question = str(result.get("question", ""))

        chosen_id = extract_text(output, "ID")
        if chosen_id is None:
            chosen_id = extract_int(output[-10:])
        try:
            chosen_idx = int(chosen_id)
        except Exception:
            continue
        if chosen_idx not in (1, 2):
            continue

        if len(sentence_ids) < 2:
            continue

        chosen_sentence_id = sentence_ids[0] if chosen_idx == 1 else sentence_ids[1]
        rejected_sentence_id = sentence_ids[1] if chosen_idx == 1 else sentence_ids[0]
        if chosen_sentence_id >= len(sentences) or rejected_sentence_id >= len(sentences):
            continue

        dataset_list.append(
            {
                "chosen_msg": [
                    {
                        "role": "user",
                        "content": _PROMPT_TEMPLATE.format(
                            query=question,
                            sentence=sentences[chosen_sentence_id],
                        ),
                    }
                ],
                "rejected_msg": [
                    {
                        "role": "user",
                        "content": _PROMPT_TEMPLATE.format(
                            query=question,
                            sentence=sentences[rejected_sentence_id],
                        ),
                    }
                ],
                "chosen_sentence_id": chosen_sentence_id,
                "rejected_sentence_id": rejected_sentence_id,
            }
        )

    # TRL RewardTrainer will consume as:
    # {
    #   "chosen":   [... conversational messages ...],
    #   "rejected": [... conversational messages ...],
    #   (optional metadata)
    # }
    return dataset_list


def train_reward_model(
    dataset_list_train: Sequence[Dict[str, object]],
    *,
    dataset_list_test: Sequence[Dict[str, object]] | None = None,
    model_name: str,
    output_dir: Path = Path("./rm_model"),
    max_length: int = 4000,
    max_characters: int = 4000,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 2,
    evaluation_steps: int = 40,
    save_steps: int = 40,
    logging_steps: int = 1,
    num_train_epochs: int = 50,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
) -> None:
    """Train a reward model using TRL's RewardTrainer with LoRA adapters (built-in collator & dataloader)."""

    from trl import RewardConfig, RewardTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    # If pad_token is missing, fall back to eos
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[tok] pad_token={repr(tokenizer.pad_token)} id={tokenizer.pad_token_id} eos={repr(tokenizer.eos_token)} id={tokenizer.eos_token_id}")

    # --- Load base model in 8-bit & wrap with LoRA ---
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 8-bit weights
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Reward head
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Prepares LayerNorms, gradients, etc. for k-bit finetuning
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "k_proj",
            "q_proj",
            "o_proj",
            "v_proj",
            "down_proj",
            "gate_proj",
            "up_proj",
        ],
        layers_to_transform=[25, 26, 27],  # model-architecture specific
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        # not 100% sure: for AutoModelForSequenceClassification you typically
        # don't need to explicitly save "score"/classifier head via modules_to_save;
        # for CausalLM reward heads you often add modules_to_save=["score"].
    )
    model = get_peft_model(model, peft_config)

    # --- Build datasets in conversational 'chosen'/'rejected' format (no tokenization here) ---
    train_dataset = _build_dataset_split(
        dataset_list_train,
        tokenizer,
        max_length=max_length,
        max_characters=max_characters,
    )

    print("[prep] n_rows:", len(train_dataset))
    print("[prep] columns:", train_dataset.column_names)


    if train_dataset is None:
        raise ValueError("Training dataset is empty; provide at least one preference pair.")

    eval_dataset = _build_dataset_split(
        dataset_list_test or [],
        tokenizer,
        max_length=max_length,
        max_characters=max_characters,
    )

    # --- W&B ---
    wandb_run = None
    report_to: List[str] = []
    if wandb_project:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is required when --wandb-project is specified.") from exc

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=list(wandb_tags) if wandb_tags else None,
            config={
                "model_name": model_name,
                "max_length": max_length,
                "max_characters": max_characters,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "num_train_epochs": num_train_epochs,
                "evaluation_steps": evaluation_steps,
                "save_steps": save_steps,
                "logging_steps": logging_steps,
            },
        )
        report_to = ["wandb"]

    # --- Trainer config (use eval_strategy, not the deprecated evaluation_strategy) ---
    evaluation_strategy = "steps" if (eval_dataset is not None and len(eval_dataset) > 0) else "no"

    training_args = RewardConfig(
        # core
        output_dir=str(output_dir),
        run_name=wandb_run_name,

        # preprocessing knobs used by RewardTrainer internally
        max_length=max_length,           # filter out samples exceeding this after tokenization
        pad_token=tokenizer.pad_token,   # ensure padding is defined
        eos_token=tokenizer.eos_token,
        dataset_num_proc=None,           # set >1 to parallelize TRL's preprocessing if desired

        # batches/optim
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",        # supported by Transformers TrainingArguments
        learning_rate=1e-4,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        warmup_steps=40,
        max_grad_norm=1.0,
        gradient_checkpointing=True,

        # logging & eval & save
        eval_strategy=evaluation_strategy,
        eval_steps=evaluation_steps if evaluation_strategy == "steps" else None,
        # eval_on_start=bool(eval_dataset),
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
        remove_unused_columns=False,     # keep our meta cols if present

        # RM-specific nicety
        # center_rewards_coefficient=1e-2,  # recommended in TRL docs for mean-zero rewards
    )

    print(f"[trainer] eval_strategy={training_args.eval_strategy} "
          f"eval_steps={training_args.eval_steps} "
          f"save_steps={training_args.save_steps} "
          f"logging_steps={training_args.logging_steps} "
          f"max_length={training_args.max_length}")

    # --- RewardTrainer uses its own internal collator & tokenization for preference data ---
    print("[trainer] initializing RewardTrainer (built-in collator/dataloader)")
    from trl import RewardTrainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # quick peek at first processed batch to verify shapes
    print("[trainer] sanity batch preview (one step, no update)")
    try:
        trainer.create_optimizer_and_scheduler(num_training_steps=1)
        dl = trainer.get_train_dataloader()
        first = next(iter(dl))
        # Expect input_ids/attention_mask of concatenated chosen+rejected (2x batch)
        for k in ["input_ids", "attention_mask"]:
            if k in first:
                shape = tuple(first[k].shape)
                print(f"[preview] {k} shape={shape}")
    except Exception as e:
        print(f"[preview] could not preview first batch (non-fatal): {e}")

    print("[train] starting training...")
    trainer.train()

    if eval_dataset is not None and len(eval_dataset) > 0:
        print("[eval] evaluating...")
        trainer.evaluate()

    if wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model using LoRA adapters.")
    parser.add_argument(
        "--dataset-list-train",
        type=Path,
        required=True,
        help="Path to dataset_list_train.json produced by judge_dataset.py.",
    )
    parser.add_argument(
        "--dataset-list-test",
        type=Path,
        help="Optional dataset_list_test.json produced by judge_dataset.py.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/workspace/llama3b-rm",
        help="Base reward model name or path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rm_model"),
        help="Directory where the trained model checkpoints will be stored.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4000,
        help="Maximum number of tokens per preference pair after tokenization.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=8000,
        help="Maximum number of characters kept from each message before tokenization.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=6,
        help="Batch size per device for the training split.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="Batch size per device for the evaluation split.",
    )
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=40,
        help="Frequency (in steps) to evaluate the model when a test split is provided.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=40,
        help="Frequency (in steps) to save checkpoints.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Frequency (in steps) to log training metrics.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=50,
        help="Number of epochs to train the reward model.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name; if omitted, W&B logging is disabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="Optional name for the W&B run.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        help="Optional list of tags to attach to the W&B run.",
    )
    args = parser.parse_args()

    if not args.dataset_list_train.exists():
        raise FileNotFoundError(f"Dataset list not found: {args.dataset_list_train}")

    with args.dataset_list_train.open() as handle:
        dataset_list_train = json.load(handle)

    dataset_list_test = None
    if args.dataset_list_test is not None:
        if not args.dataset_list_test.exists():
            raise FileNotFoundError(f"Dataset list not found: {args.dataset_list_test}")
        with args.dataset_list_test.open() as handle:
            dataset_list_test = json.load(handle)

    from datasets import Dataset

    raw_train = Dataset.from_list(dataset_list_train)
    print("[raw] n_rows:", len(raw_train))
    print("[raw] columns:", raw_train.column_names)
    print("[raw] first row:", raw_train[0])      # direct dict access to first example
    # If you want just the first 2 as a Dataset object (not a dict of lists):
    print("[raw] head(2):", raw_train.select(range(2))[:2])


    train_reward_model(
        dataset_list_train,
        dataset_list_test=dataset_list_test,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        max_characters=args.max_characters,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_steps=args.evaluation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )
