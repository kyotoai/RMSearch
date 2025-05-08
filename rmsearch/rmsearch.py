import os, asyncio
import pandas as pd
import itertools
from datasets import Dataset
import torch
import traceback
from typing import Any, Dict, List
import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, PoolingParams, SamplingParams, AsyncEngineArgs, AsyncLLMEngine
assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import time

class pooler_config:
    def __init__(self):
        self.pooling_type = "LAST"
        self.normalize = False
        self.softmax = False
        self.softmax = False
        self.step_tag_id = None
        self.returned_token_ids = None

pooler_config_ = pooler_config()

# Read one text file from S3. Ray Data supports reading multiple files
# from cloud storage (such as JSONL, Parquet, CSV, binary format).
#ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")

# For tensor_parallel_size > 1, we need to create placement groups for vLLM
# to use. Every actor has to have its own placement group.
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


#top_advice_rewards, advice_indices = torch.topk(reshaped_advice_rewards, k=num_retrival) # [len(problems), num_advices]
#torch.save(advice_indices, retrieved_advice_ids_save_path)  #all_advice_indices: [len(test_problem_ids), num_retrival]


class Search:

    def __init__(
        self,
        model_name,
        tensor_parallel_size = 1,
        pipeline_parallel_size = 1,
        llm_template = None,
    ):

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        Search.tokenizer = tokenizer

        self.model_unsupported = False

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size

        if not llm_template:
            def llm_template_func(query, key):
                message = [
                  {'role': 'user', 'content': f"Give me a key that matches the query below;\n\nQuery:{query}"},
                  {'role': 'assistant', 'content': f"{key}"}
                ]
                prompt = tokenizer.apply_chat_template(message, tokenize=False)
                return prompt

            self.llm_template = llm_template_func

        else:
            self.llm_template = llm_template

        
        try:
            
            self.engine_args = AsyncEngineArgs(
                model = model_name,
                dtype="bfloat16",
                tensor_parallel_size = tensor_parallel_size,
                pipeline_parallel_size = pipeline_parallel_size,
                distributed_executor_backend = "mp",
                gpu_memory_utilization=0.95,
                task="embed",
                override_pooler_config=pooler_config_,
            )

            if "engine" not in dir(Search):
                Search.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            
        except Exception as e:
            if "Model architectures" in f"{e}":
                save_dir = f"{model_name}-converted-model"
                score_save_path = f"{model_name}-converted-score.pt"
                self.model_unsupported = True
                if not os.path.exists(save_dir):
                    tokenizer.save_pretrained(save_dir)

                    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
                    self.score = reward_model.score.weight.data
                    torch.save(reward_model.score.weight.data, score_save_path)
                    del reward_model
                    
                    generate_model = AutoModelForCausalLM.from_pretrained(model_name)
                    generate_model.save_pretrained(save_dir)
                    del generate_model

                else:
                    self.score = torch.load(score_save_path)
                
                model_name = save_dir

                self.engine_args = AsyncEngineArgs(
                    model = model_name,
                    dtype="bfloat16",
                    tensor_parallel_size = tensor_parallel_size,
                    pipeline_parallel_size = pipeline_parallel_size,
                    distributed_executor_backend = "mp",
                    gpu_memory_utilization=0.95,
                    task="embed",
                    override_pooler_config=pooler_config_,
                )

                if "engine" not in dir(Search):
                    Search.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

            else:
                raise Exception(e)

        #self.PoolingParams = PoolingParams()

        

        resources_kwarg: Dict[str, Any] = {}
        if tensor_parallel_size == 1:
            # For tensor_parallel_size == 1, we simply set num_gpus=1.
            resources_kwarg["num_gpus"] = 1
        else:
            # Otherwise, we have to set num_gpus=0 and provide
            # a function that will create a placement group for
            # each instance.
            resources_kwarg["num_gpus"] = 0
            resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

        self.resources_kwarg = resources_kwarg

    async def __call__(self,
               queires,
               keys,
                k=5,
                return_relevance=False,
                query_id = None):

        self.query_id = query_id

        relevance = await self.get_relevance(queires, keys)
        top_relevance, top_key_ids = torch.topk(relevance, k=k)

        return_dicts = []
        for query_id, query in enumerate(queires):
            return_dict = {"query":query, "query_id":query_id, "keys":[]}
            for i in range(len(top_key_ids[query_id])):
                torch_key_id = top_key_ids[query_id, i]
                key_id = torch_key_id.item()
                if return_relevance:
                    return_dict["keys"].append({"key_id":key_id, "key":keys[key_id], "relevance":relevance[query_id, key_id].item()})
                else:
                    return_dict["keys"].append({"key_id":key_id, "key":keys[key_id]})
                    
            return_dicts.append(return_dict)

        return return_dicts  # [{"query":, "query_id":, "keys":[{"key_id":, "key":,} ...]}, ... ]


    async def get_relevance(self,
                      queries,
                      keys,
                     ):

        # Generate the Cartesian product
        query_ids = list(range(len(queries)))
        combinations = list(itertools.product(query_ids, keys))
        df = pd.DataFrame(combinations, columns=['query_id', 'key'])
        df['query'] = df['query_id'].apply(lambda idx: queries[idx])
        
        from datasets import Dataset
        dataset1 = Dataset.from_pandas(df)
        
        def format(row):
            query = row["query"]
            key = row["key"]
            prompt = self.llm_template(query, key)
            prompt = prompt[17:]  # to eliminate <|begin_of_text|> because vllm automatically add it to prompt  ####### need to be modified accordingly
            row["prompt"] = prompt
            return row
        
        formatted_dataset = dataset1.map(format)
        df_formatted = formatted_dataset.to_pandas()
        list_of_prompts = df_formatted[['prompt']].to_dict('records')  # [{"prompt":".."}, ...]

        total_num_tokens = 0
        for prompt_dict in list_of_prompts:
            inputs = Search.tokenizer(prompt_dict["prompt"], return_tensors = "pt")
            total_num_tokens += len(inputs["input_ids"][0])

        mean_num_tokens = total_num_tokens/len(list_of_prompts)

        start = time.time()
        
        rewards = await asyncio.gather(
            *[self.process(prompt_dict["prompt"], i) for i, prompt_dict in enumerate(list_of_prompts)]
        )

        end = time.time()

        print()
        print("----------")
        print("total number of inputs : ", len(list_of_prompts))
        print("mean number of tokens : ", mean_num_tokens)
        print("calculation time(s) : ", end - start)

        relevance = torch.tensor(rewards).reshape(len(queries), len(keys))
        
        return relevance

    async def process(self, prompt, request_id):

        if self.model_unsupported:

            results_generator = self.engine.encode(prompt, 
                                        PoolingParams(), 
                                        request_id
                                        )

            final_output = None
            async for request_output in results_generator:
                """
                if await request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await engine.abort(request_id)
                    # Return or raise an error
                    raise Exception()
                """
                final_output = request_output

            #embedding = final_output.outputs.embedding
            embedding = final_output.outputs.data
            embedding = torch.tensor(embedding).float()

            reward = torch.matmul(torch.tensor(embedding).unsqueeze(0), self.score.to(embedding.device).transpose(0,1))

        else:
            reward = torch.ones(1,1)  # for now
        
        return reward
    
    
        
      