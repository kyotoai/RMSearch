"""Parallel embedding helper built from vllm_generate5.py.

This module mirrors the batching/worker orchestration from
`examples/vllm_generate5.py`, but swaps generation calls for
`LLM.get_embeddings` as described in the official vLLM embedding docs:
https://docs.vllm.ai/en/latest/usage/embeddings.html.
"""

import os
import signal
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import queue as _q  # keep non-blocking put semantics clear

from IPython.display import clear_output
from vllm import LLM


# ---------------- Worker ----------------------------------------------------

def _ensure_cpu_list(vector: Any) -> List[float]:
    """Return a CPU list regardless of tensor/array implementation."""
    try:
        if hasattr(vector, "detach"):
            vector = vector.detach()
        if hasattr(vector, "cpu"):
            vector = vector.cpu()
        elif hasattr(vector, "to"):
            vector = vector.to("cpu")
        if hasattr(vector, "tolist"):
            return list(vector.tolist())
    except Exception:
        pass
    return [float(x) for x in vector]


def _worker_main(
    worker_id: int,
    device_ids: List[int],
    model: str,
    llm_kwargs: Dict[str, Any],
    task_q: mp.Queue,
    result_q: mp.Queue,
    output_to_cpu: bool,
) -> None:
    total_batches = 0
    done = 0
    phase = "starting"
    start_time = None

    stop_evt = threading.Event()

    def send_log(msg: str) -> None:
        try:
            result_q.put(("__log__", worker_id, {"msg": msg}), block=False)
        except Exception:
            print(f"[Worker {worker_id}] {msg}", flush=True)

    def heartbeat() -> None:
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
        local_kwargs = dict(llm_kwargs)
        local_kwargs["task"] = "embed"
        local_kwargs["enforce_eager"] = True
        #local_kwargs.setdefault(task, "embed")
        #local_kwargs.setdefault(enforce_eager, True)
        tensor_parallel = local_kwargs.pop("tensor_parallel_size", len(device_ids))
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel, **local_kwargs)
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

                if kind == "embed":
                    phase = "embedding"
                    inputs = payload["inputs"]
                    outputs = llm.embed(inputs)
                    embeds_list = []
                    for output in outputs:
                        embeds = output.outputs.embedding
                        if output_to_cpu:
                            embeds = _ensure_cpu_list(embeds)
                        embeds_list.append(embeds)
                    #vectors = _normalize_embedding_output(outputs)
                    result_q.put(
                        (job_id, item_idx, {"embeddings": embeds_list, "wid": worker_id}),
                        block=False,
                    )
                    done += 1
                    phase = "idle"

                else:
                    raise ValueError(f"Unknown task kind: {kind}")

            except Exception as exc:  # surface worker failure upstream for visibility
                result_q.put(
                    (
                        job_id,
                        item_idx,
                        {"error": f"{type(exc).__name__}: {exc}", "wid": worker_id},
                    ),
                    block=False,
                )
                phase = "error"
                send_log(f"error on batch {item_idx}: {type(exc).__name__}: {exc}")

    finally:
        stop_evt.set()


def _normalize_embedding_output(
    payload: Sequence[Any],
) -> List[List[float]]:
    """Return a plain list-of-list of floats regardless of vLLM return type."""
    if not payload:
        return []
    first = payload[0]
    if hasattr(first, "embedding"):
        return [list(getattr(item, "embedding")) for item in payload]
    if isinstance(first, dict) and "embedding" in first:
        return [list(item["embedding"]) for item in payload]  # type: ignore[index]
    return [list(item) for item in payload]  # assume sequence of floats


# ---------------- Parent ----------------------------------------------------

class _NotebookBoard:
    def __init__(self, num_workers: int):
        self.state = ["initializing…" for _ in range(num_workers)]

    def update(self, wid: int, text: str) -> None:
        if 0 <= wid < len(self.state):
            self.state[wid] = text
        self.render()

    def render(self) -> None:
        try:
            pass
            clear_output(wait=True)
        except Exception:
            pass
        print("== Worker logs ==")
        for idx, line in enumerate(self.state):
            print(f"[Worker {idx}] {line}")


class EmbeddingWorkerModel:
    def __init__(
        self,
        model: str,
        device_groups: List[List[int]],
        max_request_per_worker: int = 16,
        *,
        output_to_cpu: bool = False,
        **llm_kwargs: Any,
    ) -> None:
        self.ctx = mp.get_context("spawn")
        self.model = model
        self.device_groups = device_groups
        self.llm_kwargs = llm_kwargs
        self.output_to_cpu = output_to_cpu
        self.result_q: mp.Queue = self.ctx.Queue()
        self.task_queues: List[mp.Queue] = [
            self.ctx.Queue(maxsize=max_request_per_worker) for _ in device_groups
        ]
        self.procs: List[mp.Process] = []
        self._rr = 0

        for wid, devs in enumerate(device_groups):
            process = self.ctx.Process(
                target=_worker_main,
                args=(
                    wid,
                    devs,
                    model,
                    llm_kwargs,
                    self.task_queues[wid],
                    self.result_q,
                    self.output_to_cpu,
                ),
                daemon=False,
            )
            process.start()
            self.procs.append(process)

    def _send_worker_inits(self, chunk_count: int) -> None:
        workers = len(self.task_queues)
        counts = [0] * workers
        for idx in range(chunk_count):
            counts[idx % workers] += 1
        job_id = f"init-{uuid.uuid4().hex}"
        for wid, (queue_obj, total) in enumerate(zip(self.task_queues, counts)):
            queue_obj.put((job_id, wid, "init", {"total_batches": total}))

    def _next_queue(self) -> mp.Queue:
        queue_obj = self.task_queues[self._rr]
        self._rr = (self._rr + 1) % len(self.task_queues)
        return queue_obj

    def embed(
        self,
        inputs: List[str],
        batch_size: int = 8,
        timeout_s: Optional[float] = None,
    ) -> List[List[float]]:
        job_id = f"job-{uuid.uuid4().hex}"
        indexed = list(enumerate(inputs))
        chunks: List[List[Tuple[int, str]]] = [
            indexed[idx:idx + batch_size] for idx in range(0, len(indexed), batch_size)
        ]
        total_chunks = len(chunks)

        board = _NotebookBoard(num_workers=len(self.task_queues))
        self._send_worker_inits(chunk_count=total_chunks)
        board.render()

        to_submit = list(range(total_chunks))
        completed = 0
        chunk_results: Dict[int, List[List[float]]] = {}

        def try_submit(next_idx: int) -> bool:
            queue_obj = self._next_queue()
            batch_inputs = [text for _, text in chunks[next_idx]]
            message = (job_id, next_idx, "embed", {"inputs": batch_inputs})
            try:
                queue_obj.put_nowait(message)
                return True
            except _q.Full:
                return False

        poll = 0.2
        start = time.time()
        deadline = start + timeout_s if timeout_s else None

        while completed < total_chunks:
            while to_submit:
                next_idx = to_submit[0]
                if try_submit(next_idx):
                    to_submit.pop(0)
                else:
                    break

            timeout = poll
            if deadline is not None:
                remaining = max(0.0, deadline - time.time())
                if remaining == 0:
                    raise TimeoutError("Timed out waiting for worker results.")
                timeout = min(timeout, remaining)

            try:
                rid, slot, payload = self.result_q.get(timeout=timeout)
            except _q.Empty:
                board.render()
                continue

            if rid == "__log__":
                board.update(slot, payload.get("msg", ""))
                continue

            if rid != job_id:
                continue

            if "error" in payload:
                raise RuntimeError(f"Worker error in batch {slot}: {payload['error']}")

            chunk_results[slot] = payload["embeddings"]
            completed += 1
            board.render()

        outputs: List[Optional[List[float]]] = [None] * len(inputs)
        for batch_idx, chunk in enumerate(chunks):
            ids = [idx for idx, _ in chunk]
            vectors = chunk_results[batch_idx]
            for original_idx, vector in zip(ids, vectors):
                outputs[original_idx] = vector
        missing = [idx for idx, vec in enumerate(outputs) if vec is None]
        if missing:
            raise RuntimeError(f"Missing embeddings for original indices: {missing}")
        return [vec for vec in outputs if vec is not None]

    def close(self, kill: bool = False) -> None:
        for queue_obj in self.task_queues:
            try:
                queue_obj.put(None)
            except Exception:
                pass
        for process in self.procs:
            if kill and process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except Exception:
                    pass
            else:
                process.join(timeout=15)


# ---------------- Convenience helpers --------------------------------------

def build_embedding_model(
    model_name: str,
    tensor_parallel_size: int,
    num_instances: int,
    device_groups: Optional[List[List[int]]] = None,
    *,
    output_to_cpu: bool = False,
    **llm_kwargs: Any,
) -> EmbeddingWorkerModel:
    if device_groups is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpus = [int(token) for token in visible.split(",") if token.strip()]
        else:
            import torch

            gpus = list(range(torch.cuda.device_count()))
        expected = tensor_parallel_size * num_instances
        if len(gpus) < expected:
            raise ValueError(f"Need {expected} GPUs, have {len(gpus)} (gpus={gpus})")
        device_groups = [
            gpus[idx * tensor_parallel_size:(idx + 1) * tensor_parallel_size]
            for idx in range(num_instances)
        ]
    return EmbeddingWorkerModel(
        model=model_name,
        device_groups=device_groups,
        tensor_parallel_size=tensor_parallel_size,
        output_to_cpu=output_to_cpu,
        **llm_kwargs,
    )


def embed(model: EmbeddingWorkerModel, inputs: List[str], **embed_kwargs: Any) -> List[List[float]]:
    return model.embed(inputs, **embed_kwargs)
