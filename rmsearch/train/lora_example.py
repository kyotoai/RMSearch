"""LoRA reward-model training helpers with built-in dataset prep and W&B logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import extract_int, extract_text

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


def _format_preference_pair(
    example: Dict[str, object],
    tokenizer,
    *,
    max_length: int,
    max_characters: int,
) -> Dict[str, List[int]]:
    chosen_msg = example["chosen_msg"]
    rejected_msg = example["rejected_msg"]

    if not isinstance(chosen_msg, list) or not isinstance(rejected_msg, list):
        raise ValueError("Both 'chosen_msg' and 'rejected_msg' must be chat message lists.")

    trimmed_chosen = [
        {**message, "content": _truncate_message(message.get("content", ""), max_characters)}
        for message in chosen_msg
    ]
    trimmed_rejected = [
        {**message, "content": _truncate_message(message.get("content", ""), max_characters)}
        for message in rejected_msg
    ]

    prompt_plus_chosen = tokenizer.apply_chat_template(trimmed_chosen, tokenize=False)
    prompt_plus_rejected = tokenizer.apply_chat_template(trimmed_rejected, tokenize=False)

    chosen_tokens = tokenizer(
        [prompt_plus_chosen],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    rejected_tokens = tokenizer(
        [prompt_plus_rejected],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    return {
        "input_ids_chosen": chosen_tokens["input_ids"][0],
        "attention_mask_chosen": chosen_tokens["attention_mask"][0],
        "input_ids_rejected": rejected_tokens["input_ids"][0],
        "attention_mask_rejected": rejected_tokens["attention_mask"][0],
    }


def _build_dataset_split(
    records: Sequence[Dict[str, object]],
    tokenizer,
    *,
    max_length: int,
    max_characters: int,
) -> Optional[Dataset]:
    if not records:
        return None

    dataset = Dataset.from_list(list(records))

    def format_example(example: Dict[str, object]) -> Dict[str, List[int]]:
        return _format_preference_pair(
            example,
            tokenizer,
            max_length=max_length,
            max_characters=max_characters,
        )

    tokenized = dataset.map(format_example)

    keep_columns = {
        "input_ids_chosen",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
        "chosen_sentence_id",
        "rejected_sentence_id",
    }
    columns_to_remove = [column for column in tokenized.column_names if column not in keep_columns]
    if columns_to_remove:
        tokenized = tokenized.remove_columns(columns_to_remove)

    return tokenized


def make_dataset_list(
    results: Sequence[Dict[str, object]],
    *,
    sentences: Sequence[str],
) -> List[Dict[str, object]]:
    """Convert judge outputs into chat-format preference pairs."""

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

    # dataset_list (list): preference pairs used by TRL, where each element is
    #   {
    #     "chosen_msg": [{"role": "user", "content": "<prompt with positive sentence>"}],
    #     "rejected_msg": [{"role": "user", "content": "<prompt with negative sentence>"}],
    #     "chosen_sentence_id": <index of the preferred sentence>,
    #     "rejected_sentence_id": <index of the less relevant sentence>
    #   }
    return dataset_list


def train_reward_model(
    dataset_list_train: Sequence[Dict[str, object]],
    *,
    dataset_list_test: Sequence[Dict[str, object]] | None = None,
    model_name: str,
    output_dir: Path = Path("./rm_model"),
    max_length: int = 4000,
    max_characters: int = 4000,
    per_device_train_batch_size: int = 3,
    per_device_eval_batch_size: int = 2,
    evaluation_steps: int = 40,
    save_steps: int = 20,
    logging_steps: int = 1,
    num_train_epochs: int = 50,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
) -> None:
    """Train a reward model using TRL's RewardTrainer with LoRA adapters."""

    from peft import LoraConfig, TaskType, get_peft_model
    from trl import RewardConfig, RewardTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_bos_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

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
        layers_to_transform=[25, 26, 27],
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    train_dataset = _build_dataset_split(
        dataset_list_train,
        tokenizer,
        max_length=max_length,
        max_characters=max_characters,
    )
    if train_dataset is None:
        raise ValueError("Training dataset is empty; provide at least one preference pair.")

    eval_dataset = _build_dataset_split(
        dataset_list_test or [],
        tokenizer,
        max_length=max_length,
        max_characters=max_characters,
    )

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

    evaluation_strategy = "steps" if eval_dataset is not None else "no"
    training_args = RewardConfig(
        output_dir=str(output_dir),
        run_name=wandb_run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy=evaluation_strategy,
        eval_steps=evaluation_steps,
        eval_on_start=bool(eval_dataset),
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    if eval_dataset is not None:
        trainer.evaluate()

    if wandb_project:
        # Close the W&B run so metrics are flushed.
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
        default=4000,
        help="Maximum number of characters kept from each message before tokenization.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=3,
        help="Batch size per device for the training split.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=2,
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
        default=20,
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
