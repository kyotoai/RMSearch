# RMSearch Training Utilities

## General Notes

- For collaborators, make sure that run all the commands in Runpod: /workspace/(your_name)/ folder
- All scripts expect GPU acceleration and local access to the corresponding
  model weights.

## Prerequisites

1. Git clone and install packages
```bash
git clone --branch develop https://github.com/kyotoai/RMSearch.git
pip install -e RMSearch/.
```

2. (Optional) Download generate, embedding and reward model

### llama 3b Reward Model

```bash
cd /workspace
pip install "huggingface_hub[hf_transfer]"
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Ray2333/GRM-Llama3.2-3B-rewardmodel-ft --local-dir ./llama3b-rm/
```

### Qwen3 4b Instruct Model

```bash
cd /workspace
pip install "huggingface_hub[hf_transfer]"
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./qwen4b/
```


### e5 Mistral 7b Model (float16)

```bash
cd /workspace
pip install -U "huggingface_hub[hf_transfer]" && pip install -U hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download intfloat/e5-mistral-7b-instruct \
  --local-dir ./e5-mistral7b \
  --include "model-*.safetensors" "model.safetensors.index.json" \
           "config.json" "config_sentence_transformers.json" \
           "tokenizer.json" "tokenizer.model" "tokenizer_config.json" \
           "special_tokens_map.json" "added_tokens.json" \
           "sentence_bert_config.json" "modules.json" "1_Pooling/*"
```



## Overview

1. **Collect Dataset**

  * Train dataset — [`README_train_dataset.md`](README_train_dataset.md)
  * Test dataset — [`README_test_dataset.md`](README_test_dataset.md)

2. **Train Reward Model**

  * [`README_training.md`](README_training.md)

3. **Train Reward Model With Advanced DPO Batching Method**

  * [`README_adpo_training.md`](README_adpo_training.md)