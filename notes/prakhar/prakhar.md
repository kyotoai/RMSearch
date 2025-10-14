
### Deepspeed 

Quatized the model, (8 bit)

Next check num of layers trainable without OOM with Zero 0

Command 

accelerate launch --config_file deepspeed_zero.yaml -m rmsearch.train.lora_example \
  --dataset-list ./exp1/dataset_list.json \
  --model-name /workspace/llama3b-rm \
  --num-gpus 2 \
  --output-dir ./exp1/model1 \
  --base-dir ./exp1

  Error with deepspeed, run with device_map auto in config (no acelerate):

  training starts, 
  All layers (0-28) 
  Batch size: 8 (10 works too)
 Fit in 48 GB GPU