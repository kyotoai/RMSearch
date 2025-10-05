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


import json
import os, time, uuid, signal, traceback, queue, sys, threading
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from vllm import LLM, SamplingParams, PoolingParams
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import pandas as pd
from tqdm import tqdm  # kept (not used in the new log system, but left to avoid breaking imports)


def _append_checkpoint(path: Optional[str], record: Dict[str, Any]) -> None:
    """Append batch information to a JSONL checkpoint file."""
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

# ---------------------------- Jupyter log board ----------------------------
class _NotebookBoard:
    """Minimal live log board that avoids clearing once an error occurs."""

    def __init__(self, num_workers: int):
        self.state = ["initializing…" for _ in range(num_workers)]
        self._have_clear = False
        self._suppress_clear = False
        self._error_message: Optional[str] = None
        try:
            from IPython.display import clear_output  # type: ignore

            self._clear_output = clear_output
            self._have_clear = True
        except Exception:
            self._have_clear = False

    def update(self, wid: int, text: str):
        if 0 <= wid < len(self.state):
            self.state[wid] = text
        self.render()

    def disable_clear(self) -> None:
        self._suppress_clear = True

    def set_error(self, message: str) -> None:
        self._error_message = message
        self.disable_clear()
        self.render()

    def render(self) -> None:
        if self._have_clear and not self._suppress_clear:
            try:
                self._clear_output(wait=True)
            except Exception:
                pass
        if self._error_message is not None:
            print("== Worker error ==")
            print(self._error_message)
        else:
            print("== Worker logs ==")
            for i, line in enumerate(self.state):
                print(f"[Worker {i}] {line}")
        sys.stdout.flush()


def _worker_main(
    worker_id: int,
    device_ids: List[int],
    model: str,
    llm_kwargs: Dict[str, Any],
    task_q: mp.Queue,
    result_q: mp.Queue,
):
    """
    Worker with heartbeat logging. Sends logs as 4-tuples:
    ('__log__', -1, worker_id, {'msg': ...})
    Results are 4-tuples:
    (job_id, item_idx, worker_id, {'outputs': ... | 'error': ...})
    """
    total_batches = 0
    done = 0
    phase = "starting"
    start_time = None
    stop_evt = threading.Event()

    def send_log(msg: str):
        try:
            result_q.put(("__log__", -1, worker_id, {"msg": msg}), block=False)
        except Exception:
            # last resort
            print(f"[Worker {worker_id}] {msg}", flush=True)

    def heartbeat():
        last = None
        while not stop_evt.is_set():
            elapsed = (time.time() - start_time) if start_time else 0.0
            rate = (done / elapsed) if elapsed > 0 else 0.0
            pct = (100.0 * done / total_batches) if total_batches else 0.0
            rem = max(total_batches - done, 0)
            eta = (rem / rate) if (rate > 0 and total_batches) else 0.0
            msg = f"{phase}: {done}/{total_batches} ({pct:4.1f}%) · {rate:0.2f} batch/s · ETA {eta:0.1f}s"
            if msg != last:
                send_log(msg)
                last = msg
            stop_evt.wait(1.0)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

        phase = "loading model"
        send_log("loading model…")
        llm = LLM(model=model, tensor_parallel_size=len(device_ids), disable_log_stats=True, **llm_kwargs)
        llm.llm_engine.log_stats = False
        phase = "idle"
        send_log("model ready")

        # score.pt is optional
        if os.path.exists(f"{model}/score.pt"):
            score = torch.load(f"{model}/score.pt", weights_only=True)
        else:
            score = None

        start_time = time.time()
        threading.Thread(target=heartbeat, daemon=True).start()

        start = time.time()
        n_finished_item = 0

        while True:
            task = task_q.get()
            if task is None:
                phase = "stopping"
                send_log("shutting down…")
                break

            job_id, item_idx, kind, payload = task
            try:
                if kind == "init":
                    total_batches = int(payload.get("total_batches", 0))
                    done = 0
                    start_time = time.time()
                    phase = "idle"
                    send_log(f"initialized: 0/{total_batches} (0.0%)")
                    # ack (keep 4-tuple shape)
                    result_q.put((job_id, item_idx, worker_id, {"ok": True}), block=False)
                    continue

                if score is not None:
                    # Embedding + reward path
                    prompts = payload["prompts"]
                    pp = payload.get("pooling_params", None)  # kept for compatibility
                    phase = "generating"
                    outputs = llm.encode(prompts, pooling_task="embed", use_tqdm=False)

                    embeds = torch.stack([out.outputs.data for out in outputs])
                    common_dtype = torch.float32
                    common_device = embeds.device
                    embeds = embeds.to(dtype=common_dtype, device=common_device)
                    score_device = common_device
                    score_dtype = common_dtype
                    score_local = score.to(dtype=score_dtype, device=score_device)

                    rewards = torch.matmul(score_local, embeds.transpose(0, 1))[0]
                    result_q.put((job_id, item_idx, worker_id, {"outputs": rewards}))
                    n_finished_item += 1
                    done += 1
                    # keep original periodic print (unchanged) + heartbeat will also update
                    if n_finished_item % 10 == 0:
                        wrap = time.time()
                        print(f"worker_id: {worker_id},  n_finished_item: {n_finished_item},  wrap: {wrap-start} s")

                    phase = "idle"

                else:
                    # Reward API path (not implemented previously for tqdm; keep semantics)
                    prompts = payload["prompts"]
                    phase = "generating"
                    outputs = llm.reward(prompts)
                    result_q.put((job_id, item_idx, worker_id, {"outputs": outputs}))
                    done += 1
                    phase = "idle"

            except Exception as e:
                result_q.put((job_id, item_idx, worker_id, {"error": f"{type(e).__name__}: {e}"}))
                phase = "error"
                send_log(f"error on batch {item_idx}: {type(e).__name__}: {e}")

    except KeyboardInterrupt:
        phase = "interrupted"
        send_log("keyboard interrupt")
    except Exception:
        tb = traceback.format_exc()
        try:
            result_q.put(("__init__", -1, worker_id, {"fatal_error": tb}))
        except Exception:
            pass
        raise
    finally:
        stop_evt.set()


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
        checkpoint_path: Optional[str] = None,
    ) -> List[str]:
        if pooling_params is None:
            pooling_params = PoolingParams()

        job_id = f"job-{uuid.uuid4().hex}"
        indexed = list(enumerate(prompts))
        chunks: List[List[Tuple[int, str]]] = [
            indexed[i:i+batch_size] for i in range(0, len(indexed), batch_size)
        ]
        pending = len(chunks)

        # Precompute per-worker totals for 'init' (round-robin)
        W = len(self.task_queues)
        per_worker_totals = [0] * W
        for i in range(pending):
            per_worker_totals[i % W] += 1

        # Send 'init' to each worker with its total batch count (non-blocking if possible)
        init_job_id = f"init-{uuid.uuid4().hex}"
        for wid, q in enumerate(self.task_queues):
            payload = {"total_batches": per_worker_totals[wid]}
            q.put((init_job_id, wid, "init", payload))

        # Board to render logs in Jupyter; prints elsewhere
        board = _NotebookBoard(num_workers=W)
        board.render()

        # Non-blocking dispatch + interleaved log polling
        to_submit = list(range(pending))  # chunk indices not yet submitted
        rr = 0  # independent round-robin pointer for submission
        chunk_results: Dict[int, Any] = {}
        deadline = time.time() + timeout_s if timeout_s else None
        poll = 0.2  # seconds

        print("3")  # keep original debug print

        def try_submit(next_idx: int) -> bool:
            nonlocal rr
            wid = rr
            q = self.task_queues[wid]
            _, batch_prompts = zip(*chunks[next_idx])
            payload = {"prompts": list(batch_prompts), "pooling_params": pooling_params}
            msg = (job_id, next_idx, "reward", payload)
            try:
                q.put_nowait(msg)
                rr = (rr + 1) % W
                return True
            except queue.Full:
                return False

        while len(chunk_results) < pending:
            # 1) Submit without blocking as much as possible
            progressed = False
            while to_submit:
                idx = to_submit[0]
                if try_submit(idx):
                    to_submit.pop(0)
                    progressed = True
                else:
                    break  # some queues are full; go poll logs/results

            # 2) Poll logs/results briefly so UI doesn't stall
            if deadline:
                remaining = max(0.0, deadline - time.time())
                tmo = min(poll, remaining)
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for worker results.")
            else:
                tmo = poll

            try:
                rid, item_idx, wid, payload = self.result_q.get(timeout=tmo)
            except queue.Empty:
                # no messages this tick; refresh board to keep it alive
                board.render()
                continue

            if rid == "__init__" and "fatal_error" in payload:
                error_msg = f"Worker failed to initialize:\n{payload['fatal_error']}"
                board.set_error(error_msg)
                raise RuntimeError(error_msg)

            if rid == "__log__":
                board.update(wid, payload.get("msg", ""))
                continue

            if rid != job_id:
                # ignore non-job messages (e.g., init acks)
                continue

            if "error" in payload:
                error_msg = f"Worker error in batch {item_idx}: {payload['error']}"
                board.set_error(error_msg)
                raise RuntimeError(error_msg)

            # normal batch result
            batch_outputs = payload["outputs"]
            chunk_results[item_idx] = batch_outputs

            if checkpoint_path:
                original_indices, _ = zip(*chunks[item_idx])
                if isinstance(batch_outputs, torch.Tensor):
                    serializable = batch_outputs.detach().cpu().tolist()
                else:
                    try:
                        serializable = [float(x) for x in batch_outputs]
                    except Exception:
                        serializable = list(batch_outputs)
                _append_checkpoint(
                    checkpoint_path,
                    {
                        "job_id": job_id,
                        "batch_index": item_idx,
                        "indices": list(original_indices),
                        "outputs": serializable,
                    },
                )

        # Reconstruct outputs in original order
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


def search(
    model: LLMWorker,
    requests: List[Dict[str, Any]],
    llm_template,
    topk: int = 10,
    query_batch_size: int = 128,
    **gen_kwargs,
) -> List[str]:

    # requests = [{"query": "...", "keys": ["k1","k2", ...]}, ...]
    df = pd.DataFrame(requests)
    df["query_id"] = range(len(df))
    
    # explode keys
    df = df.explode("keys", ignore_index=True).rename(columns={"keys": "key"})
    
    # assign sequential key_id inside each query_id
    df["key_id"] = df.groupby("query_id").cumcount()
    
    # reorder columns
    df = df[["query_id", "query", "key_id", "key"]]

    dataset1 = Dataset.from_pandas(df)

    def format(row):
        prompt = llm_template(row)
        prompt = prompt[17:]  # to eliminate <|begin_of_text|> because vllm automatically add it to prompt
        row["prompt"] = prompt
        return row

    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")

    records: List[Dict[str, Any]] = []
    num_rows = len(dataset1)

    for start in range(0, num_rows, query_batch_size):
        end = min(start + query_batch_size, num_rows)
        batch_indices = list(range(start, end))
        formatted_batch = dataset1.select(batch_indices).map(format)
        prompts = formatted_batch["prompt"]

        batch_rewards = model.encode(prompts, **gen_kwargs)
        if isinstance(batch_rewards, torch.Tensor):
            batch_scores = batch_rewards.detach().cpu().numpy()
        else:
            batch_scores = batch_rewards

        for idx in range(len(formatted_batch)):
            row = formatted_batch[idx]
            score = batch_scores[idx]
            records.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "key_id": row["key_id"],
                    "key": row["key"],
                    "relevance": float(score.item() if hasattr(score, "item") else score),
                }
            )

    if not records:
        return []

    df = pd.DataFrame.from_records(
        records,
        columns=["query_id", "query", "key_id", "key", "relevance"],
    )

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
            "query": g["query"].unique()[0],
            "keys": g[["key", "key_id", "relevance"]].to_dict("records")
        })
        .tolist()
    )

    return result
