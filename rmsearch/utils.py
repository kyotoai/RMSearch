from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import time, os
import torch

# You should custom the following function depending on your model

# step 1. See score name in the model
# step 2. 
def convert_model(model_name, keep_original_model=False):
    
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)

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