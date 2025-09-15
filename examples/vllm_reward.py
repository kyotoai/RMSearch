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
from tqdm import tqdm

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

        llm = LLM(model=model, tensor_parallel_size=len(device_ids), disable_log_stats=True, **llm_kwargs)
        llm.llm_engine.log_stats = False

        if os.path.exists(f"{model}/score.pt"):
            score = torch.load(f"{model}/score.pt", weights_only = True)
        else:
            score = None

        start = time.time()
        n_finished_item = 0
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
                    outputs = llm.encode(prompts, pooling_task="embed") #, use_tqdm=False)
                    
                    #print(outputs)
                    embeds = torch.stack([out.outputs.data for out in outputs])
                    common_dtype = torch.float32
                    common_device = embeds.device  # or torch.device("cuda") if you prefer
                    embeds = embeds.to(dtype=common_dtype, device=common_device)
                    score  = score.to(dtype=common_dtype, device=common_device)
                    
                    #print(score, embeds)
                    rewards = torch.matmul(score, embeds.transpose(0,1))[0]
                    #print(rewards)
                    #result_q.put((job_id, item_idx, {"outputs": rewards}))
                    result_q.put((job_id, item_idx, worker_id, {"outputs": rewards}))
                    n_finished_item += 1
                    if n_finished_item % 10 == 0:
                        wrap = time.time()
                        print(f"worker_id: {worker_id},  n_finished_item: {n_finished_item},  wrap: {wrap-start} s")

                else:  # Not Implemented Yet
                    prompts = payload["prompts"]
                    outputs = llm.reward(prompts)
                    #result_q.put((job_id, item_idx, {"outputs": outputs}))
                    result_q.put((job_id, item_idx, worker_id, {"outputs": outputs}))

            except Exception as e:
                #result_q.put((job_id, item_idx, {"error": f"{type(e).__name__}: {e}"}))
                result_q.put((job_id, item_idx, worker_id, {"error": f"{type(e).__name__}: {e}"}))


    except KeyboardInterrupt:
        pass
    except Exception:
        tb = traceback.format_exc()
        try:
            #result_q.put(("__init__", -1, {"fatal_error": tb}))
            result_q.put(("__init__", -1, worker_id, {"fatal_error": tb}))

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

    '''
    def _next_queue(self) -> mp.Queue:
        q = self.task_queues[self._rr]
        self._rr = (self._rr + 1) % len(self.task_queues)
        return q
    '''

    def _next_queue(self) -> Tuple[int, mp.Queue]:
        wid = self._rr
        q = self.task_queues[wid]
        self._rr = (self._rr + 1) % len(self.task_queues)
        return wid, q

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

        assigned_counts = {wid: 0 for wid in range(len(self.task_queues))}
        assigned_worker: Dict[int, int] = {}

        print("3")

        for item_idx, chunk in enumerate(chunks):
            _, batch_prompts = zip(*chunk)
            payload = {"prompts": list(batch_prompts), "pooling_params": pooling_params}
            #self._next_queue().put((job_id, item_idx, "reward", payload))
            wid, q = self._next_queue()
            assigned_worker[item_idx] = wid
            assigned_counts[wid] += 1
            q.put((job_id, item_idx, "reward", payload))

        # One tqdm bar per worker/process
        bars = {
            wid: tqdm(total=cnt, position=wid, desc=f"Worker {wid}", leave=False)
            for wid, cnt in assigned_counts.items() if cnt > 0
        }

        chunk_results: Dict[int, List[str]] = {}
        deadline = time.time() + timeout_s if timeout_s else None

        while pending > 0:
            remaining = max(0.0, (deadline - time.time())) if deadline else None
            try:
                #rid, item_idx, payload = self.result_q.get(timeout=remaining)
                rid, item_idx, wid, payload = self.result_q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("Timed out waiting for worker results.")
            if rid == "__init__" and "fatal_error" in payload:
                raise RuntimeError(f"Worker failed to initialize:\n{payload['fatal_error']}")
            if rid != job_id:
                continue
            if "error" in payload:
                raise RuntimeError(f"Worker error in batch {item_idx}: {payload['error']}")
            
            # Update the corresponding worker's progress bar
            if wid in bars:
                bars[wid].update(1)
                print(f"wid:{wid}, item_idx:{item_idx}")

            chunk_results[item_idx] = payload["outputs"]
            pending -= 1

        for b in bars.values():
            b.close()

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

def search(model: LLMWorker, requests: List[Dict[str, Any]], llm_template, topk = 10, **gen_kwargs) -> List[str]:
    
    # requests = [{"query": "...", "keys": ["k1","k2", ...]}, ...]
    df = pd.DataFrame(requests)
    df["query_id"] = range(len(df))
    
    # explode keys
    df = df.explode("keys", ignore_index=True).rename(columns={"keys": "key"})
    
    # assign sequential key_id inside each query_id
    df["key_id"] = df.groupby("query_id").cumcount()
    
    # reorder columns
    df = df[["query_id", "query", "key_id", "key"]]

    #PREFIX = "<|begin_of_text|>"
    
    #def strip_prefix(s: str, prefix: str = PREFIX) -> str:
    #    return s[len(prefix):] if isinstance(s, str) and s.startswith(prefix) else s

    dataset1 = Dataset.from_pandas(df)

    def format(row):
        prompt = llm_template(row)
        prompt = prompt[17:]  # to eliminate <|begin_of_text|> because vllm automatically add it to prompt  ####### need to be modified accordingly
        row["prompt"] = prompt
        return row
    
    formatted_dataset = dataset1.map(format)
    df = formatted_dataset.to_pandas()
    
    #list_of_prompts = df_formatted[['prompt']].to_dict('records')  # [{"prompt":".."}, ...]
    
    # Build prompts and strip the prefix (like your `prompt[17:]`)
    #df["prompt"] = df.apply(lambda row: llm_template(row)[17:], axis=1)

    rewards = model.encode(df["prompt"], **gen_kwargs)

    df["relevance"] = rewards.numpy()

    # Sort and pick topn per request
    df = (
        df.sort_values(["query_id", "relevance"], ascending=[True, False])
                .groupby("query_id")
                .head(topk)
    )

    result = (
        df.groupby("query_id")
        .apply(lambda g: {
            "query_id": g.name,
            "query": g["query"].unique()[0], #.tolist(),
            "keys": g[["key", "key_id", "relevance"]].to_dict("records")
        })
        .tolist()
    )

    return result
