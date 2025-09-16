# --- import at top ---
import queue as _q  # distinguish from variable name 'queue' in stdlib
import threading
from IPython.display import clear_output

import os, time, uuid, signal, traceback, queue, sys, threading
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from vllm import LLM, SamplingParams

# ... (rest of your imports and code) ...




# ---------------- Worker: keep lightweight heartbeat logging ----------------
def _worker_main(
    worker_id: int,
    device_ids: List[int],
    model: str,
    llm_kwargs: Dict[str, Any],
    task_q: mp.Queue,
    result_q: mp.Queue,
):
    total_batches = 0
    done = 0
    phase = "starting"
    start_time = None

    stop_evt = threading.Event()

    def send_log(msg: str):
        try:
            result_q.put(("__log__", worker_id, {"msg": msg}), block=False)
        except Exception:
            # last resort; may not show reliably in Jupyter
            print(f"[Worker {worker_id}] {msg}", flush=True)

    def heartbeat():
        last = None
        while not stop_evt.is_set():
            elapsed = (time.time() - start_time) if start_time else 0.0
            rate = done / elapsed if elapsed > 0 else 0.0
            pct = (100.0 * done / total_batches) if total_batches else 0.0
            rem = max(total_batches - done, 0)
            eta = rem / rate if rate > 0 and total_batches else 0.0
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
        llm = LLM(model=model, tensor_parallel_size=len(device_ids), **llm_kwargs)
        phase = "idle"
        send_log("model ready")

        start_time = time.time()
        threading.Thread(target=heartbeat, daemon=True).start()

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
                    result_q.put((job_id, item_idx, {"ok": True}), block=False)
                    continue

                if kind == "generate":
                    phase = "generating"
                    prompts = payload["prompts"]
                    sp = payload["sampling_params"]
                    outputs = llm.generate(prompts, sp)
                    texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
                    # include worker id so parent can manage inflight if needed
                    result_q.put((job_id, item_idx, {"texts": texts, "wid": worker_id}), block=False)
                    done += 1
                    phase = "idle"

                else:
                    raise ValueError(f"Unknown task kind: {kind}")

            except Exception as e:
                result_q.put((job_id, item_idx, {"error": f"{type(e).__name__}: {e}", "wid": worker_id}), block=False)
                phase = "error"
                send_log(f"error on batch {item_idx}: {type(e).__name__}: {e}")

    finally:
        stop_evt.set()

# ---------------- Parent: non-blocking dispatcher + interleaved log polling ----------------

class _NotebookBoard:
    def __init__(self, num_workers: int):
        self.state = ["initializing…" for _ in range(num_workers)]
    def update(self, wid: int, text: str):
        if 0 <= wid < len(self.state):
            self.state[wid] = text
        self.render()
    def render(self):
        try:
            clear_output(wait=True)
        except Exception:
            pass
        print("== Worker logs ==")
        for i, line in enumerate(self.state):
            print(f"[Worker {i}] {line}")

class LLMWorkerModel:
    # ... __init__ unchanged ...
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

        for wid, devs in enumerate(device_groups):
            p = self.ctx.Process(
                target=_worker_main,
                args=(wid, devs, model, llm_kwargs, self.task_queues[wid], self.result_q),
                daemon=False,
            )
            p.start()
            self.procs.append(p)


    def _send_worker_inits(self, chunk_count: int):
        W = len(self.task_queues)
        counts = [0] * W
        for i in range(chunk_count):
            counts[i % W] += 1
        job_id = f"init-{uuid.uuid4().hex}"
        for wid, (q, total) in enumerate(zip(self.task_queues, counts)):
            payload = {"total_batches": total}
            q.put((job_id, wid, "init", payload))
        # don't drain here; let main loop handle logs uniformly

    def _next_queue(self) -> mp.Queue:
        q = self.task_queues[self._rr]
        self._rr = (self._rr + 1) % len(self.task_queues)
        return q

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        batch_size: int = 8,
        timeout_s: Optional[float] = None,
    ) -> List[str]:
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=32, temperature=0.7, top_p=0.95)

        job_id = f"job-{uuid.uuid4().hex}"
        indexed = list(enumerate(prompts))
        chunks: List[List[Tuple[int, str]]] = [
            indexed[i:i + batch_size] for i in range(0, len(indexed), batch_size)
        ]
        total_chunks = len(chunks)

        board = _NotebookBoard(num_workers=len(self.task_queues))
        self._send_worker_inits(chunk_count=total_chunks)
        board.render()

        # scheduling state
        to_submit = list(range(total_chunks))   # indices of chunks not yet submitted
        submitted = 0
        completed = 0
        chunk_results: Dict[int, List[str]] = {}

        # Helper to try enqueue without blocking; returns True if enqueued
        def try_submit(next_idx: int) -> bool:
            q = self._next_queue()
            _, batch_prompts = zip(*chunks[next_idx])
            payload = {"prompts": list(batch_prompts), "sampling_params": sampling_params}
            msg = (job_id, next_idx, "generate", payload)
            try:
                q.put_nowait(msg)  # <-- never block here
                return True
            except _q.Full:
                return False

        # Main loop: interleave submission attempts and queue polling
        poll = 0.2
        start = time.time()
        deadline = start + timeout_s if timeout_s else None

        while completed < total_chunks:
            # 1) Try to submit as many as possible without blocking
            progressed = False
            while to_submit:
                next_idx = to_submit[0]
                if try_submit(next_idx):
                    to_submit.pop(0)
                    submitted += 1
                    progressed = True
                else:
                    break  # queue(s) full for now; go poll messages

            # 2) Poll for logs/results briefly
            tmo = poll
            if deadline:
                remain = max(0.0, deadline - time.time())
                if remain == 0:
                    raise TimeoutError("Timed out waiting for worker results.")
                tmo = min(tmo, remain)

            try:
                rid, slot, payload = self.result_q.get(timeout=tmo)
            except _q.Empty:
                # No messages — still render board to keep UI “alive”
                board.render()
                continue

            # Handle messages
            if rid == "__init__" and "fatal_error" in payload:
                raise RuntimeError(f"Worker failed to initialize:\n{payload['fatal_error']}")

            if rid == "__log__":
                board.update(slot, payload.get("msg", ""))
                continue

            if rid != job_id:
                # ignore other job IDs
                continue

            if "error" in payload:
                raise RuntimeError(f"Worker error in batch {slot}: {payload['error']}")

            # Normal batch result
            chunk_results[slot] = payload["texts"]
            completed += 1
            board.render()

        # Reorder to original prompt order
        outputs: List[Optional[str]] = [None] * len(prompts)
        for item_idx, chunk in enumerate(chunks):
            idxs, _ = zip(*chunk)
            texts = chunk_results[item_idx]
            for i, t in zip(idxs, texts):
                outputs[i] = t
        return outputs  # type: ignore

    def close(self, kill: bool = False):
        for q in self.task_queues:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            if kill and p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception:
                    pass
            else:
                p.join(timeout=15)



# -------------------------- Convenience functions --------------------------
def build_llm(
    model_name: str,
    tensor_parallel_size: int,
    num_instances: int,
    device_groups: Optional[List[List[int]]] = None,
    **llm_kwargs,
) -> LLMWorkerModel:
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
            gpus[i * tensor_parallel_size:(i + 1) * tensor_parallel_size]
            for i in range(num_instances)
        ]
    return LLMWorkerModel(model=model_name, device_groups=device_groups, **llm_kwargs)

def generate(model: LLMWorkerModel, prompts: List[str], **gen_kwargs) -> List[str]:
    return model.generate(prompts, **gen_kwargs)

