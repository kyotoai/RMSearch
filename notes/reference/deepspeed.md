# Deepspeed

Deepspeed Env
1. Deploy “winglian/axolotl-cloud:main-latest” from edit template
2. Open web terminal and get authorization token by `jupyter server list`



!!!
No matter how much I tried, I couldn’t solve this issue in torch 2.2.0 env
https://github.com/huggingface/transformers/issues/25582

so I used winglian/axolotl-cloud:main-latest template instead and could solve the issue

Using web terminal is better


deepspeed_test1.py
-> running this asks to create wandb account but you don’t need to listen and select 3


https://github.com/deepspeedai/DeepSpeed/issues/3062

AssertionError: Not enough buffers 0 for swapping 1

このエラーが治らない

3/1
治った！！
`Increase buffer_count in offload_param to 10, 15, and 20.`
ここでbuffer_countを２０にしたら治った

deepspeedが動かせるようにはなったが、model parallelはできていないとおもう
https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Getting_Started_with_DeepSpeed/Getting_Started_with_DeepSpeed.html#getting-started-with-deepspeed
ここにあるnum_gpuの調節でなんとかしてみる。

またメモリ調節は自動的についている可能性もある。もう少しdeepspeedについて学ばないといけない。

公式doc:
https://www.deepspeed.ai/getting-started/


3/2

できた！！
deepspeed --num_gpus=2 deepspeed_test2.py
コマンドを実行する時にモデルを幾つのgpuにdistributeしたいかをnum_gpusで指定する。
-> いや違ったこれはただ使うGPUの数を指定するだけ。パラレルの数ではない

https://www.deepspeed.ai/tutorials/automatic-tensor-parallelism/
this might be the solution

deepspeed deepspeed_test2.py --ds_inference

the log looks like the model tensor is loaded into multiple gpus!!

but I got this error 
https://github.com/pytorch/torchtune/issues/2093
which is hard to fix

pip install omegaconf
pip install torchdata
pip install torchtune
pip install torchao

accelerate config

accelerate launch deepspeed_test2.py
accelerate launch deepspeed_test2.py --config /root/.cache/huggingface/accelerate/default_config.yaml

axolotle
https://axolotl-ai-cloud.github.io/axolotl/docs/multi-gpu.html




