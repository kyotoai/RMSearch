# vllm_model.py

"""sample.py
import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # do this ONCE per kernel before using mp

from vllm import SamplingParams
from examples.vllm_reward import build_llm, embed_with_model

#model_name = "/workspace/qwen7b"
#model_name = "/workspace/llama3b-rm"
model_name = "/workspace/llama3b-rm-converted-model"
device_groups = [[0], [1]]  # 2 workers on GPU 0 and 1

model = build_llm(
    model_name=model_name,
    tensor_parallel_size=len(device_groups[0]),
    num_instances=len(device_groups),
    device_groups=device_groups,
    max_model_len=2500,
    max_num_seqs=64,
    gpu_memory_utilization=0.90,
    runner="pooling",
)

try:
    #sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    prompts = ["Hello, my name is", "The capital of France is"] * 2
    outs = embed_with_model(model, prompts, batch_size=4, timeout_s=180)
    for i, (inp, out) in enumerate(zip(prompts, outs)):
        print(f"[#{i}] {inp!r} -> {out!r}")

    outs = embed_with_model(model, prompts, batch_size=4, timeout_s=180)
    for i, (inp, out) in enumerate(zip(prompts, outs)):
        print(f"[#{i}] {inp!r} -> {out!r}")
finally:
    model.close()
"""


import os, time, uuid, signal, traceback, queue
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from vllm import LLM, SamplingParams, PoolingParams
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import pandas as pd

def _worker_main(
    worker_id: int,
    device_ids: List[int],
    model: str,
    llm_kwargs: Dict[str, Any],
    task_q: mp.Queue,
    result_q: mp.Queue,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        llm = LLM(model=model, tensor_parallel_size=len(device_ids), **llm_kwargs)
        if os.path.exists(f"{model}/score.pt"):
            score = torch.load(f"{model}/score.pt", weights_only = True)
        else:
            score = None

        while True:
            task = task_q.get()
            if task is None:
                break
            job_id, item_idx, kind, payload = task
            try:
                if score != None:
                    prompts = payload["prompts"]
                    pp = payload["pooling_params"]
                    # If pickling SamplingParams ever bites, reconstruct:
                    # if isinstance(sp, dict): sp = SamplingParams(**sp)
                    outputs = llm.encode(prompts, pooling_task="embed")
                    
                    #print(outputs)
                    embeds = torch.stack([out.outputs.data for out in outputs])
                    common_dtype = torch.float32
                    common_device = embeds.device  # or torch.device("cuda") if you prefer
                    embeds = embeds.to(dtype=common_dtype, device=common_device)
                    score  = score.to(dtype=common_dtype, device=common_device)
                    
                    #print(score, embeds)
                    rewards = torch.matmul(score, embeds.transpose(0,1))[0]
                    #print(rewards)
                    result_q.put((job_id, item_idx, {"outputs": rewards}))
                else:  # Not Implemented Yet
                    prompts = payload["prompts"]
                    outputs = llm.reward(prompts)
                    result_q.put((job_id, item_idx, {"outputs": outputs}))
            except Exception as e:
                result_q.put((job_id, item_idx, {"error": f"{type(e).__name__}: {e}"}))

    except KeyboardInterrupt:
        pass
    except Exception:
        tb = traceback.format_exc()
        try:
            result_q.put(("__init__", -1, {"fatal_error": tb}))
        except Exception:
            pass
        raise

class LLMWorker:
    def __init__(
        self,
        model: str,
        device_groups: List[List[int]],
        max_request_per_worker: int = 16,
        **llm_kwargs,
    ):
        self.ctx = mp.get_context("spawn")
        self.model = model
        self.device_groups = device_groups
        self.llm_kwargs = llm_kwargs
        self.result_q: mp.Queue = self.ctx.Queue()
        self.task_queues: List[mp.Queue] = [
            self.ctx.Queue(maxsize=max_request_per_worker) for _ in device_groups
        ]
        self.procs: List[mp.Process] = []
        self._rr = 0
        self.requests = []
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", add_eos_token=True, add_bos_token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        for wid, devs in enumerate(device_groups):
            p = self.ctx.Process(
                target=_worker_main,
                args=(wid, devs, model, llm_kwargs, self.task_queues[wid], self.result_q),
                daemon=False,
            )
            p.start()
            self.procs.append(p)

    def _next_queue(self) -> mp.Queue:
        q = self.task_queues[self._rr]
        self._rr = (self._rr + 1) % len(self.task_queues)
        return q

    def encode(
        self,
        prompts: List[str],
        pooling_params = None,
        batch_size: int = 8,
        timeout_s: Optional[float] = None,
    ) -> List[str]:
        if pooling_params is None:
            pooling_params = PoolingParams()

        job_id = f"job-{uuid.uuid4().hex}"
        indexed = list(enumerate(prompts))
        chunks: List[List[Tuple[int, str]]] = [
            indexed[i:i+batch_size] for i in range(0, len(indexed), batch_size)
        ]
        pending = len(chunks)

        for item_idx, chunk in enumerate(chunks):
            _, batch_prompts = zip(*chunk)
            payload = {"prompts": list(batch_prompts), "pooling_params": pooling_params}
            self._next_queue().put((job_id, item_idx, "reward", payload))

        chunk_results: Dict[int, List[str]] = {}
        deadline = time.time() + timeout_s if timeout_s else None

        while pending > 0:
            remaining = max(0.0, (deadline - time.time())) if deadline else None
            try:
                rid, item_idx, payload = self.result_q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("Timed out waiting for worker results.")
            if rid == "__init__" and "fatal_error" in payload:
                raise RuntimeError(f"Worker failed to initialize:\n{payload['fatal_error']}")
            if rid != job_id:
                continue
            if "error" in payload:
                raise RuntimeError(f"Worker error in batch {item_idx}: {payload['error']}")
            chunk_results[item_idx] = payload["outputs"]
            pending -= 1

        outputs: List[Optional[str]] = [None] * len(prompts)
        for item_idx, chunk in enumerate(chunks):
            idxs, _ = zip(*chunk)
            texts = chunk_results[item_idx]
            for i, t in zip(idxs, texts):
                outputs[i] = t

        outputs = torch.stack(outputs)  # torch.tensor([reward1, reward2, ... ])
        
        return outputs  # type: ignore

    def close(self, kill: bool = False):
        for q in self.task_queues:
            try: q.put(None)
            except Exception: pass
        for p in self.procs:
            if kill and p.is_alive():
                try: os.kill(p.pid, signal.SIGKILL)
                except Exception: pass
            else:
                p.join(timeout=15)

def build_llm(
    model_name: str,
    tensor_parallel_size: int,
    num_instances: int,
    device_groups: Optional[List[List[int]]] = None,
    **llm_kwargs,
) -> LLMWorker:
    if device_groups is None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            gpus = [int(x) for x in vis.split(",") if x.strip() != ""]
        else:
            import torch
            gpus = list(range(torch.cuda.device_count()))
        expected = tensor_parallel_size * num_instances
        if len(gpus) < expected:
            raise ValueError(f"Need {expected} GPUs, have {len(gpus)} (gpus={gpus})")
        device_groups = [
            gpus[i*tensor_parallel_size:(i+1)*tensor_parallel_size]
            for i in range(num_instances)
        ]
    return LLMWorker(model=model_name, device_groups=device_groups, **llm_kwargs)

def embed_with_model(model: LLMWorker, prompts: List[str], **gen_kwargs) -> List[str]:
    return model.encode(prompts, **gen_kwargs)

def search(model: LLMWorker, requests: List[Dict[str, Any]], llm_template, **gen_kwargs) -> List[str]:
    # requests: [{"query":"", "keys":["", ...], }, ...]

    # requests = [{"query": "...", "keys": ["k1","k2", ...]}, ...]
    df = pd.DataFrame(requests)
    df["query_id"] = range(len(df))
    
    # explode keys
    df = df.explode("keys", ignore_index=True).rename(columns={"keys": "key"})
    
    # assign sequential key_id inside each query_id
    df["key_id"] = df.groupby("query_id").cumcount()
    
    # reorder columns
    df = df[["query_id", "query", "key_id", "key"]]

    '''
    # Step 1. Put into DataFrame
    df = pd.DataFrame(requests).reset_index(names="request_id")
    
    # Step 2. Explode queries
    qdf = df[["request_id", "queries"]].explode("queries").reset_index(drop=True)
    qdf["query_id"] = qdf.groupby("request_id").cumcount()
    qdf = qdf.rename(columns={"queries": "query"})
    
    # Step 3. Explode keys
    kdf = df[["request_id", "keys"]].explode("keys").reset_index(drop=True)
    kdf["key_id"] = kdf.groupby("request_id").cumcount()
    kdf = kdf.rename(columns={"keys": "key"})
    
    # Step 4. Cross join everything
    cross = qdf.merge(kdf, how="cross")
    
    # Step 5. Filter so that only rows with same request_id remain
    df = cross[cross["request_id_x"] == cross["request_id_y"]] \
        .drop(columns=["request_id_y"]) \
        .rename(columns={"request_id_x": "request_id"})
    '''

    '''
    example:
    requests = [
        {"queries":["q1","q2"], "keys":["k1","k2"]},
        {"queries":["q3"], "keys":["k3","k4","k5"]}
    ]
    
    df =
        request_id query  query_id key  key_id
    0            0    q1         0  k1       0
    1            0    q1         0  k2       1
    5            0    q2         1  k1       0
    6            0    q2         1  k2       1
    12           1    q3         0  k3       0
    13           1    q3         0  k4       1
    14           1    q3         0  k5       2
    '''
    

    PREFIX = "<|begin_of_text|>"
    
    def strip_prefix(s: str, prefix: str = PREFIX) -> str:
        return s[len(prefix):] if isinstance(s, str) and s.startswith(prefix) else s
    
    # Build prompts and strip the prefix (like your `prompt[17:]`)
    df["prompt"] = (
        df.apply(lambda row: llm_template(row), axis=1)
                    .apply(strip_prefix)
    )

    rewards = model.encode(df["prompt"], **gen_kwargs)
    relevance = torch.stack(rewards).reshape(len(queries), len(keys))

    topn = 2  # choose how many rows per group

    # Sort and pick topn per request
    df = (
        df.sort_values(["query_id", "reward"], ascending=[True, False])
                .groupby("query_id")
                .head(topn)
    )

    '''
    # Collapse into result list per request
    df = (
        df.groupby("request_id")
        .apply(lambda g: g[["key", "key_id", "reward"]].to_dict("records"))
        .reset_index(name="result")
    )'''

    result = (
        df.groupby("query_id")
        .apply(lambda g: {
            "query_id": g.name,
            "query": g["query"].unique()[0], #.tolist(),
            "keys": g[["key", "key_id", "reward"]].to_dict("records")
        })
        .tolist()
    )

    
    '''
    example:
    result = 
    [
      {"request_id": 0,
       "keys": [
           {"key": "k3", "key_id": 2, "reward": 0.9},
           {"key": "k1", "key_id": 0, "reward": 0.8}
       ]},
      {"request_id": 1,
       "keys": [
           {"key": "k4", "key_id": 0, "reward": 0.7},
           {"key": "k6", "key_id": 2, "reward": 0.6}
       ]}
    ]
    '''
    

    '''
    dataset1 = Dataset.from_pandas(final_df)

    def format(row):
        prompt = llm_template(row)
        prompt = prompt[17:]  # to eliminate <|begin_of_text|> because vllm automatically add it to prompt  ####### need to be modified accordingly
        row["prompt"] = prompt
        return row
    
    formatted_dataset = dataset1.map(format)
    df_formatted = formatted_dataset.to_pandas()
    list_of_prompts = df_formatted[['prompt']].to_dict('records')  # [{"prompt":".."}, ...]
    

    #total_num_tokens = 0
    #for prompt_dict in list_of_prompts:
    #    inputs = Search.tokenizer(prompt_dict["prompt"], return_tensors = "pt")
    #    total_num_tokens += len(inputs["input_ids"][0])

    #mean_num_tokens = total_num_tokens/len(list_of_prompts)

    start = time.time()

    if disable_log:
        rewards = await asyncio.gather(
            *[self.process(prompt_dict["prompt"], i, progress_bar) 
              for i, prompt_dict in enumerate(list_of_prompts)]
        )

    else:
        rewards = await tqdm_asyncio.gather(
            *[self.process(prompt_dict["prompt"], i, progress_bar) 
              for i, prompt_dict in enumerate(list_of_prompts)],
            desc="Searching: "
        )
    
    #rewards = await asyncio.gather(
    #    *[self.process(prompt_dict["prompt"], i) for i, prompt_dict in enumerate(tqdm(list_of_prompts, desc="Searching"))]
    #)

    end = time.time()
    

    #print()
    #print("----------")
    #print("total number of inputs : ", len(list_of_prompts))
    #print("mean number of tokens : ", mean_num_tokens)
    #print("calculation time(s) : ", end - start)

    relevance = torch.stack(rewards).reshape(len(queries), len(keys))


    relevance = await self.get_relevance(queires, keys, disable_log, progress_bar)
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
    '''

    return result
