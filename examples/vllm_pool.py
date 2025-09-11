# vllm_pool.py
import os, time, uuid, signal, traceback, queue
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from vllm import LLM, SamplingParams, PoolingParams
import torch

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
                if kind == "embed":
                    prompts = payload["prompts"]
                    pp = payload["pooling_params"]
                    # If pickling SamplingParams ever bites, reconstruct:
                    # if isinstance(sp, dict): sp = SamplingParams(**sp)
                    outputs = llm.encode(prompts, pooling_task="embed")

                    common_dtype = torch.float32
                    common_device = embeds.device  # or torch.device("cuda") if you prefer
                    
                    embeds = embeds.to(dtype=common_dtype, device=common_device)
                    score  = score.to(dtype=common_dtype, device=common_device)
                    
                    #print(outputs)
                    embeds = torch.stack([out.outputs.data for out in outputs])
                    #print(score, embeds)
                    rewards = torch.matmul(score, embeds.transpose(0,1))[0]
                    #print(rewards)
                    result_q.put((job_id, item_idx, {"outputs": rewards}))
                elif kind == "reward":  # Not Implemented Yet
                    prompts = payload["prompts"]
                    outputs = llm.reward(prompts)
                    result_q.put((job_id, item_idx, {"outputs": outputs}))
                else:
                    raise ValueError(f"Unknown task kind: {kind}")
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

class LLMWorkerPool:
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
            self._next_queue().put((job_id, item_idx, "embed", payload))

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
) -> LLMWorkerPool:
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
    return LLMWorkerPool(model=model_name, device_groups=device_groups, **llm_kwargs)

def embed_with_pool(pool: LLMWorkerPool, prompts: List[str], **gen_kwargs) -> List[str]:
    return pool.encode(prompts, **gen_kwargs)
