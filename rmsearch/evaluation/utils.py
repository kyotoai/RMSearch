from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from pathlib import Path
import time, os
import torch
import argparse

'''

Examples:

1. When converting checkpoint to model compatible with rmsearch.utils.vllm_reward.py
```bash
python -m rmsearch.evaluation.utils \
  --type checkpoint \
  --check-point-path /workspace/Prakhar/exp5/model1/checkpoint-400 \
  --base-model-path /workspace/qwen4b-reranker \
  --model-path /workspace/qwen4b-reranker-exp5-model1-400
```

2. When converting a model (reranker, reward, etc.) from huggingface to converted-model compatible with rmsearch.utils.vllm_reward.py
```bash
python -m rmsearch.evaluation.utils \
  --type model \
  --model-path /workspace/qwen4b-reranker
```

'''


# You should custom the following function depending on your model

# step 1. See score name in the model
# step 2. 
def convert_model(model_name, keep_original_model=False):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)

    if not keep_original_model:
        save_dir = model_name
        score_save_path = f"{save_dir}/score.pt"
    else:
        save_dir = f"{model_name}-converted-model"
        score_save_path = f"{save_dir}/score.pt"

    print(f"Save Converted Model in {save_dir}")
    tokenizer.save_pretrained(save_dir)

    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    torch.save(reward_model.score.weight.data, score_save_path)
    del reward_model
    
    generate_model = AutoModelForCausalLM.from_pretrained(model_name)
    generate_model.save_pretrained(save_dir)
    del generate_model

def revert_model(model_name, keep_converted_model=False):
    # Not implemented yet.
    pass

def convert_checkpoint(base_model_path, checkpoint_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left",add_eos_token=False,add_bos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    # Load the pre-trained model without the LM head.
    # AutoModelForCausalLM usually refers to models with the LM head included, so you'd typically use a more specific base model class.
    #base_model = AutoModelForCausalLM.from_config(config).base_model
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=1)

    lora_model = PeftModel.from_pretrained(model, checkpoint_path)

    reward_model = lora_model.merge_and_unload()

    score_save_path = f"{model_path}/score.pt"

    tokenizer.save_pretrained(model_path)
    reward_model.save_pretrained(model_path)

    print("reward_model: ", reward_model)
    
    torch.save(reward_model.score.weight.data, score_save_path)
    del reward_model

    generate_model = AutoModelForCausalLM.from_pretrained(model_path)
    generate_model.save_pretrained(model_path)
    del generate_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate query sequences with controlled relevance drift for source keys.")
    parser.add_argument("--type", type=str, required=True, help="checkpoint or model.")
    parser.add_argument("--check-point-path", type=Path, default=None, help="Checkpoint path after you train a reward model.")
    parser.add_argument("--base-model-path", type=Path, default=None, help="Base reward model path to add lora weight to.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to save the converted reward model.")
    parser.add_argument("--keep-original", type=bool, default=True, help="Path to save the converted reward model.")
    args = parser.parse_args()

    if args.type == "checkpoint":
        if not args.base_model_path or not args.base_model_path or not args.model_path:
            raise Exception("provide check-point-path, base-model-path and model-path for type:checkpoint")
        convert_checkpoint(args.base_model_path, args.check_point_path, args.model_path)
    elif args.type == "model":
        if not args.keep_original or not args.model_path:
            raise Exception("provide check-point-path, base-model-path and model-path for type:checkpoint")
        convert_model(args.model_path, keep_original_model=args.keep_original)
    else:
        raise Exception("type should be checkpoint or model")

