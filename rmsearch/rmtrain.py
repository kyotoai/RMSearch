from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType
import accelerate
from peft import get_peft_model

import json
import os
import random
import pandas as pd
from operator import itemgetter
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification,AutoTokenizer,TrainingArguments,AutoConfig,AutoModelForCausalLM



class CustomRewardTrainer(RewardTrainer):
    _tag_names = ["trl", "reward-trainer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs): # You need this because it will use RewardTrainer compute_loss method without this. To use a subclass function, some method in the subclass must be called from main directly. 
        return super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(num_print_samples=1, *args, **kwargs)


class RMTrainer:

    def __init__(self,
             model_name = "llama3b-rm",
             num_gpus = None,
        ):

        self.num_gpus = num_gpus
        self.model_name = model_name
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = config.eos_token_id
        self.tokenizer = tokenizer
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)


    def prepare_dataset(self,
                        dataset_list,
                        base_dir = "./RMSearch_exp",
                        test_size = None,
                       ):

        """
        dataset_list should be like
        [
            {
                “query”:[{“system”:”…”}, …],
                “chosen_key”:[{“assistant”:”…”},…],
                “rejected_key”:[{“assistant”:”…”},…],
                **kwargs,
            }
            ...
        ]
        """

        dataset_save_path = f"{base_dir}/dataset"
        train_ids_save_path = f"{base_dir}/train_ids.json"
        test_ids_save_path = f"{base_dir}/test_ids.json"
        
        dataset = Dataset.from_list(dataset_list)
        #print(dataset.to_pandas())

        if not os.path.exists(dataset_save_path):
            
            def formatting_func(examples):
                kwargs = {"padding": "max_length", "truncation": True, "max_length": 4000, "return_tensors": "pt", "add_special_tokens":False}
                query = examples['query']
                chosen_key = examples['chosen_key']
                rejected_key = examples['rejected_key']
            
                if type(query)==list and type(chosen_key)==list and type(rejected_key)==list:
                    chosen_message = query + chosen_key
                    rejected_message = query + rejected_key

                elif type(query)==str and type(chosen_key)==str and type(rejected_key)==str:
                    chosen_message = [
                        {'role': 'user', 'content': f"Give me a key to the query below;\n\nQuery:{query}"},
                        {'role': 'assistant', 'content': f"{chosen_key}"}
                    ]

                    rejected_message = [
                        {'role': 'user', 'content': f"Give me a key to the query below;\n\nQuery:{query}"},
                        {'role': 'assistant', 'content': f"{rejected_key}"}
                    ]

                else:
                    raise Exception("query must be str or list like [{'role':'', }]")

                prompt_plus_chosen_response = self.tokenizer.apply_chat_template(chosen_message, tokenize=False)
                prompt_plus_rejected_response = self.tokenizer.apply_chat_template(rejected_message, tokenize=False)
                
                #prompts = chosen_prompts+rejected_prompts
                #inputs = tokenizer(prompts, **kwargs)
        
                #chosen_reject_similarities = advice_similarities[chosen_ids][:, rejected_ids]
        
                tokens_chosen = self.tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
                tokens_rejected = self.tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
                
                return {
                    "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                    "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
                }
        
                '''
                return {
                    "num_chosen":len(chosen_prompts), "num_rejected":len(rejected_prompts),
                    "input_ids":inputs["input_ids"], "attention_mask":inputs["attention_mask"],
                    
                    #"chosen_reject_similarities":chosen_reject_similarities,
                }
                '''
        
            formatted_dataset = dataset.map(formatting_func)
        
            try:
                with open(train_ids_save_path) as f:
                    train_indices = json.load(f)
                with open(test_ids_save_path) as f:
                    test_indices = json.load(f)
                    
                #formatted_dataset['test'] = formatted_dataset.select(test_indices)
                #formatted_dataset['train'] = formatted_dataset.select(train_indices)
        
                from datasets import DatasetDict
                formatted_dataset = DatasetDict({
                    "train": formatted_dataset.select(train_indices),
                    "test": formatted_dataset.select(test_indices)
                })
        
            except:
                
                # Get the total number of samples in the dataset
                total_samples = len(formatted_dataset)

                if not test_size:
                    test_size = int(total_samples*0.1)
                
                # Generate random indices for the test set using PyTorch
                test_indices = torch.randperm(total_samples)[:test_size]
                
                # Create the train set by excluding the test indices using set difference
                all_indices = set(range(total_samples))
                test_indices_set = set(test_indices.tolist())
                train_indices = list(all_indices - test_indices_set)
        
                from datasets import DatasetDict
                formatted_dataset = DatasetDict({
                    "train": formatted_dataset.select(train_indices),
                    "test": formatted_dataset.select(test_indices.tolist())
                })
        
                with open(train_ids_save_path, "w") as f:
                    json.dump(train_indices, f)
                with open(test_ids_save_path, "w") as f:
                    json.dump(test_indices.tolist(), f)
                
                #formatted_dataset['test'] = formatted_dataset.select(test_indices.tolist())
                #formatted_dataset['train'] = formatted_dataset.select(train_indices)
        
            #formatted_dataset = formatted_dataset.train_test_split(test_size=test_size)
            formatted_dataset.save_to_disk(dataset_save_path)
        else:
            print(f"Existed: {dataset_save_path}")
            formatted_dataset = load_from_disk(dataset_save_path)
            

        return formatted_dataset

    
    def train(self,
              formatted_dataset,
              training_args = None,
              peft_config = None,
             ):

        # Configuring the training arguments
        if not training_args:
            training_args = RewardConfig(  #TrainingArguments(   #CustomRewardTrainer( #
                output_dir=model_save_dir,
                per_device_train_batch_size=batch_size_per_device,
                per_device_eval_batch_size=eval_batch_size_per_device,
                evaluation_strategy="steps",
                eval_steps=20,
                eval_on_start=True,
                save_steps=20,
                logging_steps=1,
                num_train_epochs = 3,
                report_to=None,
                remove_unused_columns=False,
            )
        
        
        if not peft_config:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                target_modules=["k_proj","q_proj","o_proj", "v_proj","down_proj","gate_proj","up_proj",],
                layers_to_transform=[25,26,27],
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
            )
        
        self.model = get_peft_model(self.model, peft_config)
        
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
            non_tensor_fields = ["num_chosen", "num_rejected", "problem_id"]
            for field in non_tensor_fields:
                batch[field] = [f[field] for f in features]
            
            return batch
        
        if self.num_gpus*training_args.per_device_train_batch_size != 1:
            num_trash = len(formatted_dataset["train"])%(self.num_gpus*training_args.per_device_train_batch_size)
            formatted_dataset["train"] = formatted_dataset["train"].select(range(len(formatted_dataset["train"])-num_trash))
            num_trash = len(formatted_dataset["test"])%(self.num_gpus*training_args.per_device_train_batch_size)
            formatted_dataset["test"] = formatted_dataset["test"].select(range(len(formatted_dataset["test"])-num_trash))
            
        # Loading the RewardTrainer from TRL
        trainer = CustomRewardTrainer(
        #trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset["train"],
            eval_dataset=formatted_dataset["test"],
            #data_collator=custom_data_collator,
            #peft_config=peft_config,
        )
        
        accelerator = trainer.accelerator
        self.model = self.model.to(accelerator.device)
        
        train_output = trainer.train()





