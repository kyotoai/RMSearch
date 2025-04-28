import pandas as pd
import itertools
from datasets import Dataset
import torch

from typing import Any, Dict, List
import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

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


class rmsearch:

    def __init__(
        self,
        model_name = "llama3b-rm",
        tensor_parallel_size = 1,
        num_instances = 1,
        llm_template = None,
    ):

        self.llm = LLM(model=model_name,
                       dtype="bfloat16",
                       #dtype="float32",
                       tensor_parallel_size = tensor_parallel_size,
                       num_instances = num_instances,
                       task="embed",
                       override_pooler_config=pooler_config_,)

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.num_instances = num_instances
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

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

    
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        outputs = self.llm.embed(batch["prompt"])

        embeddings = []
        for output in outputs:
            embeddings.append(output.outputs.embedding)
            #generated_text.append(' '.join([o.text for o in output.outputs]))

        return {
            "embeddings": embeddings
        }

    def search(self,
               queires,
               keys,
                k=5,
                return_relevance=False,):

        relevance = self.get_relevance(queires, keys)
        top_relevance, top_key_ids = torch.topk(relevance, k=k)

        return_dicts = []
        for query_id, query in queires:
            return_dict = {"query":query, "query_id":query_id, "keys":[]}
            for i, torch_key_id in top_key_ids:
                key_id = torch_key_id.item()
                if return_relevance:
                    return_dict["keys"].append({"key_id":key_id, "key":keys[key_id], "relevance":relevance[query_id, key_id].item()})
                else:
                    return_dict["keys"].append({"key_id":key_id, "key":keys[key_id]})
                    
            return_dicts.append(return_dict)

        return return_dicts


    def get_relevance(self,
                      queires,
                      keys,
                     ):
        
        # Generate the Cartesian product
        query_ids = list(range(len(queires)))
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
        
        ds = ray.data.from_items(list_of_prompts)
        #ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
        
        batch_size = len(list_of_prompts)//self.num_instances
        
        # Apply batch inference for all input data.
        ds = ds.map_batches(
            rmsearch,
            # Set the concurrency to the number of LLM instances.
            concurrency=self.num_instances,
            # Specify the batch size for inference.
            batch_size=batch_size,
            **resources_kwarg,
        )
        
        # Peek first 10 results.
        # NOTE: This is for local testing and debugging. For production use case,
        # one should write full result out as shown below.
        # outputs = ds.take(limit=10)
        outputs = ds.take_all()  # [{"embeddings":(2d list)}, ...]
        
        #outputs = model.embed(prompts)
        
        dataset1 = Dataset.from_list(outputs)
        print("Putting All Embeddings onto GPU...")
        all_embeddings = torch.tensor(dataset1["embeddings"]).to("cuda")
        #all_embeddings = torch.tensor([outputs[i].outputs.embedding for i in range(len(outputs))]).to("cuda")
        rm_head = torch.load(score_path, weights_only=True)
        rm_head = rm_head.to(all_embeddings.device)
        print("Calculating matmul...")
        advice_rewards = torch.matmul(all_embeddings, rm_head.transpose(0,1)).squeeze()
        relevance = advice_rewards.reshape(len(problems), len(advices))  # [len(problems), len(advices)]
        print("Obtained Relevance")
        
        return relevance
    
    
        
      