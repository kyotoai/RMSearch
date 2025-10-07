"""LoRA reward-model training helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .utils import extract_int, extract_text

__all__ = ["make_dataset_list", "train_reward_model"]

_PROMPT_TEMPLATE = (
    "Give me relevant score between query and sentence;\n\n"
    "Query:{query}\n\n"
    "Sentence:```{sentence}```"
)


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
    dataset_list: Sequence[Dict[str, object]],
    *,
    model_name: str,
    num_gpus: int = 2,
    output_dir: Path = Path("./rm_model"),
    base_dir: Path = Path("./rm_exp"),
) -> None:
    """Train a reward model using TRL's RewardTrainer with LoRA adapters."""

    from peft import LoraConfig, TaskType
    from rmsearch import RMTrainer
    from trl import RewardConfig, RewardTrainer

    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rmtrainer = RMTrainer(model_name=model_name, num_gpus=num_gpus)
    tokenizer = rmtrainer.tokenizer

    def formatting_func(examples):
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": 4000,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        chosen_msg = examples["chosen_msg"]
        rejected_msg = examples["rejected_msg"]

        if len(chosen_msg[0]["content"]) > 4000:
            chosen_msg[0]["content"] = chosen_msg[0]["content"][:4000] + "..."
        if len(rejected_msg[0]["content"]) > 4000:
            rejected_msg[0]["content"] = rejected_msg[0]["content"][:4000] + "..."

        prompt_plus_chosen = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
        prompt_plus_rejected = tokenizer.apply_chat_template(rejected_msg, tokenize=False)

        chosen_tokens = tokenizer.encode_plus(prompt_plus_chosen, **kwargs)
        rejected_tokens = tokenizer.encode_plus(prompt_plus_rejected, **kwargs)

        return {
            "input_ids_chosen": chosen_tokens["input_ids"][0],
            "attention_mask_chosen": chosen_tokens["attention_mask"][0],
            "input_ids_rejected": rejected_tokens["input_ids"][0],
            "attention_mask_rejected": rejected_tokens["attention_mask"][0],
        }

    formatted_dataset = rmtrainer.prepare_dataset(
        dataset_list,
        base_dir=base_dir,
        test_size=100,
        formatting_func=formatting_func,
    )

    class CustomRewardTrainer(RewardTrainer):
        _tag_names = ["trl", "reward-trainer"]

        def train(self, *args, **kwargs):  # type: ignore[override]
            return super().train(*args, **kwargs)

        def evaluate(self, *args, **kwargs):  # type: ignore[override]
            return super().evaluate(num_print_samples=1, *args, **kwargs)

    training_args = RewardConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=40,
        eval_on_start=True,
        save_steps=20,
        logging_steps=1,
        num_train_epochs=50,
        report_to=None,
        remove_unused_columns=False,
    )

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

    rmtrainer.train(
        formatted_dataset,
        training_args=training_args,
        peft_config=peft_config,
        trainer_cls=CustomRewardTrainer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model using LoRA adapters.")
    parser.add_argument(
        "--dataset-list",
        type=Path,
        required=True,
        help="Path to dataset_list.json produced by make_dataset_list",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/workspace/llama3b-rm",
        help="Base reward model name or path.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs available for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rm_model"),
        help="Directory where the trained model checkpoints will be stored.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("./rm_exp"),
        help="Working directory used for intermediate datasets.",
    )
    args = parser.parse_args()

    if not args.dataset_list.exists():
        raise FileNotFoundError(f"Dataset list not found: {args.dataset_list}")

    with args.dataset_list.open() as handle:
        dataset_list = json.load(handle)

    train_reward_model(
        dataset_list,
        model_name=args.model_name,
        num_gpus=args.num_gpus,
        output_dir=args.output_dir,
        base_dir=args.base_dir,
    )
