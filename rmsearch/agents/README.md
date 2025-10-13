# RMSearch agent implementation for practical use

## Overview

1. make_agents.py (based on agents.md from developper, llm makes agents)
2. make_evaluation_dataset_*.py (ex. excel, code, search, ...)
-> llm_inference.py (makes rm_dataset.json)
-> train reward model 


## `make_agents.py`

Generate agents to make inference.

```bash
python -m rmsearch.agents.make_agents \
  --prompts-dir RMSearch/rmsearch/agents/generate_agents/example/ \
  --n-agents 1 \
  --output ./agents/agent_records.json \
  --model-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --max-model-len 10_000
```

**Arguments**
- `--prompts-dir`: Directory with markdown prompt files. From the prompts, agents are generated
- `--n-agents`: Number of agents generated from a prompt file. Default set to 5.
- `--output`: Destination JSON file for agents.
- `--model-name`: Generation model checkpoint.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Control worker topology; `--device-groups` accepts strings like `"0,1;2,3"`.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling and scheduling knobs.
- `--gpu-memory-utilization`, `--max-model-len`, `--dtype`, `--trust-remote-code`: Options forwarded to `vllm.LLM`.

**Outputs**
- `agent_records.json`: List of dictionaries with `agent`, `agent_id` and `gen_prompt_file`.
- Example entry:
  ```json
  {
    "agent": "(prompt to llm)",
    "agent_id": 0,
    "gen_prompt_file": "/example/code_agents.md"
  }
  ```

**Notices**
- It's using utils/vllm_generate.py
- Basically each file inside the dir corresponds to an agent group. Agent group is like agents about coding or agents about searching. Inside the file, it describes how each agent in an agent group works. One agent corresponds to one prompt of LLM and output. For example, in code agents group, you can add something like agent to search keywords inside a dir, agent to search all file names, or planning how to code with instructions. 
- Inside ./agents/generate_agents/example/, there are sample files code_agents.md, instructions.md, web_search.md and thinking.md.
    - code_agents.md: Contains all python function calls to analyze code
    - instructions.md: One or a several lines of instructions to be added to other agents.
    - web_search.md: With google_api, search docments in web.
    - thinking.md: Let llm generates thingking. Basically they all have instructions section to be searched from instruction agents by reward model. Ex. to plan, to summarize, to think next action etc.





## `make_evaluation_dataset_code.py`

```bash
python -m rmsearch.agents.make_evaluation_dataset_code \
  --code-dir ./code \
  --output ./agents/dataset.json \
  --model-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --max-model-len 10_000
```

**Arguments**
- `--code-dir`: Code files are inside the directory.
```
|- code1
|   |- ...
|- code2
|   |- ...
```
- `--output`: Destination JSON file for dataset.
- `--model-name`: Generation model checkpoint.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Control worker topology; `--device-groups` accepts strings like `"0,1;2,3"`.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling and scheduling knobs.
- `--gpu-memory-utilization`, `--max-model-len`, `--dtype`, `--trust-remote-code`: Options forwarded to `vllm.LLM`.

**Outputs**
- `datatset.json`: List of dictionaries with `task`, `dir_path` and `correct_answer`.
- Example JSON
```
[
    {
        "task": "Implement ...?",
        "dir_path": "./agents/code/gkvp",
        "correct_answer": "..."
    },
    ...
]
```

**Notices**
- This file intentionally deletes some functions inside code and make task to code the deleted part.




## `llm_inference.py`

```bash
python -m rmsearch.agents.llm_inference \
  --code-dir ./agents/code \
  --dataset ./agents/dataset.json \
  --output ./agents/inference_log \
  --inference_out ./agents/result.json \
  --model-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --max-model-len 10_000
```

**Arguments**
- `--code-dir`: Code files are inside the directory.
```
|- code1
|   |- ...
|- code2
|   |- ...
```
- `--output`: Destination JSON file for dataset.
- `--model-name`: Generation model to check if the output is correct.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Control worker topology; `--device-groups` accepts strings like `"0,1;2,3"`.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling and scheduling knobs.
- `--gpu-memory-utilization`, `--max-model-len`, `--dtype`, `--trust-remote-code`: Options forwarded to `vllm.LLM`.

**Outputs**
- `result.json`: List of dictionaries with `question`, `dir_path`, `output`, `correct_answer` and `correctness`.
- Example output JSON
```
[
    {
        "question": "How to implement ...?",
        "dir_path": "./agents/code/gkvp",
        "inference_out":"./agents/inference_log/0.parquet
        "output": "...",
        "correct_answer": "...",
        "correctness": True or False,
    },
    ...
]
```

**Notices**
- It's using utils/vllm_generate.py and utils/vllm_reward.py
- inference_out has graph structure to save each agent's input and output, and which agents the outputs are passed to.
