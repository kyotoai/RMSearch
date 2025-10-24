
"""LoRA reward-model training helpers with built-in dataset prep and W&B logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardConfig, RewardTrainer

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import extract_int, extract_text

__all__ = ["make_dataset_list", "train_reward_model"]

_PROMPT_TEMPLATE = (
    "Give me relevant score between query and sentence;\n\n"
    "Query:{query}\n\n"
    'Sentence:```{sentence}```'
)

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from transformers import Trainer
from trl.trainer.utils import compute_accuracy

#class CustomRewardTrainer(RewardTrainer):
class CustomRewardTrainer(Trainer):
    _tag_names = ["trl", "reward-trainer"]

    def __init__(self, *args, **kwargs):
        super().__init__(compute_metrics=compute_accuracy, *args, **kwargs)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:

        dpo_pairs_list = inputs["dpo_pairs"]   #[num_gpus]

        #n_chosens = inputs["num_chosen"]  #[num_gpus]
        #n_rejecteds = inputs["num_rejected"]  #[num_gpus]
        #chosen_reject_similarities = inputs["chosen_reject_similarities"]  #[num_gpus, n_chosens, n_rejecteds]

        #print(inputs["input_ids_chosen"].shape)
        #print(inputs["input_ids_rejected"].shape)

        num_gpus, num_batch, max_length = inputs["input_ids"].shape
        input_ids = inputs["input_ids"].reshape(num_gpus*num_batch, max_length)
        attention_mask = inputs["attention_mask"].reshape(num_gpus*num_batch, max_length)
        #print(inputs["input_ids"].shape)
        #attention_mask = inputs["attention_mask"].reshape(inputs["attention_mask"].shape[0]*inputs["attention_mask"].shape[1], inputs["attention_mask"].shape[2])

        all_rewards = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )["logits"]

        #print("1", all_rewards.shape)
        all_rewards = all_rewards.reshape(num_gpus,num_batch)
        #print("2", all_rewards.shape)
        
        #all_chosen_idx = []
        #all_rejected_idx = []
        total_loss = 0
        for i in range(num_gpus):
            rewards = all_rewards[i]
            dpo_pairs = dpo_pairs_list[i]

            #print("rewards: ", rewards)
            #print("dpo_pairs: ", dpo_pairs)

            cr_matrix = torch.zeros(len(dpo_pairs), len(rewards))
            for i, dpo_pair in enumerate(dpo_pairs):
                cr_matrix[i, dpo_pair[0]] = 1
                cr_matrix[i, dpo_pair[1]] = -1
            
            '''
            n_chosen = n_chosens[i]
            n_rejected = n_rejecteds[i]
            n_rows = n_chosen * n_rejected
            n_cols = n_chosen + n_rejected
            chosen_idx = torch.arange(n_chosen).repeat_interleave(n_rejected)   # shape: [n_rows]
            rejected_idx = torch.arange(n_rejected).repeat(n_chosen)   # shape: [n_rows]
            all_chosen_idx.append(chosen_idx)
            all_rejected_idx.append(rejected_idx)
            
            # Create cr_matrix
            cr_matrix = torch.zeros(n_rows, n_cols)
            rows = torch.arange(n_rows)
            cr_matrix[rows, chosen_idx] = 1
            cr_matrix[rows, n_chosen + rejected_idx] = -1
            '''

            cr_matrix = cr_matrix.to(rewards.device)

            # Create coe_vector
            #sims = chosen_reject_similarities[chosen_idx, rejected_idx] # shape: [n_rows]
            #coe_vector = 1-sims

            #print(coe_vector.shape)
            #print(coe_vector)

            # Test cr_matrix
            #test_tensor = torch.ones(rewards.shape).to(cr_matrix.device)
            #result_tensor = torch.matmul(cr_matrix, test_tensor)
            #print("result_tensor.shape: ", result_tensor.shape)
            #print("result_tensor.mean() : ", result_tensor.mean())
            
            # calculate loss, optionally modulate with margin
            if "margin" in inputs:
                loss = -nn.functional.logsigmoid(torch.matmul(cr_matrix, rewards) - inputs["margin"]).mean()
            else:
                loss = -nn.functional.logsigmoid(torch.matmul(cr_matrix, rewards)).mean()

            total_loss += loss
            #all_losses.append(loss)

        #total_loss = torch.tensor(all_losses).mean()
        
        if return_outputs:
            return total_loss, {
                "all_rewards": all_rewards,
                #"all_chosen_idx": all_chosen_idx,
                #"all_rejected_idx": all_rejected_idx,
                "dpo_pairs_list": dpo_pairs_list,
                #"n_chosen":n_chosen,
                #"n_rejected":n_rejected,
            }
        return total_loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        '''
        chosen_rewards = logits_dict["chosen_rewards"]
        rejected_rewards = logits_dict["rejected_rewards"]
        chosen_idx = logits_dict["chosen_idx"]
        rejected_idx = logits_dict["rejected_idx"]
        '''

        all_rewards = logits_dict["all_rewards"]
        dpo_pairs_list = logits_dict["dpo_pairs_list"]
        #all_chosen_idx = logits_dict["all_chosen_idx"]
        #all_rejected_idx = logits_dict["all_rejected_idx"]
        #n_chosen = logits_dict["n_chosen"]
        #n_rejected = logits_dict["n_rejected"]
        
        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        all_logits = torch.tensor([])
        for i in range(len(all_rewards)):
            rewards = all_rewards[i]
            dpo_pairs = dpo_pairs_list[i]

            dpo_pairs_T = torch.tensor(dpo_pairs).transpose(0,1)
            #print("rewards.shape: ", rewards.shape)
            #print("all_chosen_idx[i]: ", all_chosen_idx[i])
            chosen_logits = rewards[dpo_pairs_T[i]].unsqueeze(-1)
            rejected_logits = rewards[dpo_pairs_T[i]].unsqueeze(-1)
            #chosen_logits = rewards[all_chosen_idx[i]].unsqueeze(-1)
            #rejected_logits = rewards[n_chosen + all_rejected_idx[i]].unsqueeze(-1)
            logits = torch.cat((chosen_logits, rejected_logits), dim=1)  # 
            #print("logits.shape: ", logits.shape)
            all_logits = all_logits.to(logits.device)
            all_logits = torch.cat((all_logits, logits), dim=0)

        all_logits = all_logits.softmax(dim=1).detach()
        #print("all_logits: ", all_logits)
        #print("all_logits.shape: ", all_logits.shape)
        #logits = torch.cat((chosen_rewards[chosen_idx], rejected_rewards[rejected_idx]), dim=1)
        #logits = logits.softmax(dim=1).detach()
        '''
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T
        '''

        labels = torch.zeros(all_logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, all_logits, labels

    def train(self, *args, **kwargs): # You need this because it will use RewardTrainer compute_loss method without this. To use a subclass function, some method in the subclass must be called from main directly. 
        return super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)
        #return super(RewardTrainer, self).evaluate(*args, **kwargs)
        #return super().evaluate(num_print_samples=1, *args, **kwargs) # this fell in an error for some reason



'''
# CustomRewardTrainer example
class CustomRewardTrainer(RewardTrainer):
    _tag_names = ["trl", "reward-trainer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs): # You need this because it will use RewardTrainer compute_loss method without this. To use a subclass function, some method in the subclass must be called from main directly. 
        return super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(num_print_samples=1, *args, **kwargs)

'''

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
    batch = example["batch"]
    #ex. batch = [
    #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[1])}], "query_id":query_id, "key_id":},
    #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[0])}], "query_id":query_id, "key_id":},
    #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[0])}], "query_id":query_id, "key_id":}
    #],

    dpo_pairs = example["dpo_pairs"]
    #ex. dpo_pairs = [
    #  [0,1],  # [(chosen_msg_id), (rejected_msg_id)]
    #  [0,2],
    #  [1,2]
    #]

    if not isinstance(batch, list) or not isinstance(dpo_pairs, list):
        raise ValueError("Both 'batch' and 'dpo_pairs' must be lists.")

    prompts = [tokenizer.apply_chat_template(batch_dict["msg"], tokenize=False) for batch_dict in batch]

    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "dpo_pairs": dpo_pairs,
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
        "input_ids",
        "attention_mask",
        "dpo_pairs",
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
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=wandb_run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy=evaluation_strategy,
        eval_steps=evaluation_steps,
        eval_on_start=bool(eval_dataset),
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
        remove_unused_columns=False,
    )

    def custom_data_collator(features):
        batch = {}
        
        # For fields that are tensors, we stack them.
        
        tensor_fields = [
            "input_ids", "attention_mask",
        ]
        '''
        tensor_fields = [
            "input_ids_chosen", "attention_mask_chosen",
            "input_ids_rejected", "attention_mask_rejected"
        ]
        '''
        
        for field in tensor_fields:
            batch[field] = torch.stack([torch.tensor(f[field]) for f in features])  #[num_gpus, num_advice_per_batch, max_length]
        
        # For the original prompts (strings), we simply collect them in a list.
        non_tensor_fields = ["dpo_pairs"]
        for field in non_tensor_fields:
            batch[field] = [f[field] for f in features]
        
        return batch

    trainer = CustomRewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,
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
