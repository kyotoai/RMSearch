# --- import at top ---
import argparse
import queue as _q  # distinguish from variable name 'queue' in stdlib
import threading

import os, time, uuid, signal, traceback, queue, sys, threading
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from vllm import LLM, SamplingParams

from rmsearch._display import resolve_clear_output, should_enable_board, should_use_tqdm

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
import re

try:
    from vllm.inputs import TokensPrompt
except Exception:
    from vllm.inputs.data import TokensPrompt

_clear_output = resolve_clear_output()
_USE_TQDM = should_use_tqdm()



def extract_harmony_final(text: str) -> str:
    # 1) official Harmony tag, per cookbook
    # https://cookbook.openai.com/articles/openai-harmony
    tag = "<|channel|>final<|message|>"
    if tag in text:
        tail = text.split(tag, 1)[1]
    else:
        # 2) flattened vLLM style like your example:
        # "analysis....assistantfinal- ...."
        m = re.search(r"assistantfinal-?\s*", text)
        if not m:
            return text.strip()
        tail = text[m.end():]

    # trim common Harmony end markers
    for stop in ("<|return|>", "<|end|>", "<|stop|>"):
        i = tail.find(stop)
        if i != -1:
            tail = tail[:i]
            break

    return tail.strip()


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
        llm = LLM(model=model, tensor_parallel_size=len(device_ids),
            trust_remote_code=True,
             **llm_kwargs)

        # --- Harmony setup for GPT-OSS ---
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        # ----------------------------------
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

                    raw_prompts = payload["prompts"] # plain user strings from parent

                    sp = payload["sampling_params"]

                    # outputs = llm.generate(prompts, sp, use_tqdm=_USE_TQDM)
                        # 1) build Harmony conversations -> token IDs
                    # prompt_token_ids = []
                    # for p in raw_prompts:
                    #     convo = Conversation.from_messages([
                    #         Message.from_role_and_content(
                    #             Role.SYSTEM,
                    #             SystemContent.new(),          # you can add Reasoning: high here if you want
                    #         ),
                    #         # optional developer block:
                    #         # Message.from_role_and_content(
                    #         #     Role.DEVELOPER,
                    #         #     DeveloperContent.new().with_instructions("You are a helpful assistant."),
                    #         # ),
                    #         Message.from_role_and_content(Role.USER, p),
                    #     ])
                    #     prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
                    #     prompt_token_ids.append(prefill_ids)

                    # # 2) make sure we stop on Harmony assistant end markers
                    # if not getattr(sp, "stop_token_ids", None):
                    #     sp.stop_token_ids = stop_token_ids

                    # # 3) call vLLM with token-ids, not strings  ← this is the big change
                    # outputs = llm.generate(
                    #     prompt_token_ids=prompt_token_ids,
                    #     sampling_params=sp,
                    #     use_tqdm=_USE_TQDM,
                    # )
                        # 1) build Harmony-prefilled token IDs for every prompt
                    prompt_token_ids_list = []
                    for p in raw_prompts:
                        convo = Conversation.from_messages([
                            # system can be empty; you can later stick your RMSearch system here
                            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                            Message.from_role_and_content(Role.USER, p),
                        ])
                        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
                        prompt_token_ids_list.append(prefill_ids)

                    # 2) make sure Harmony stop tokens are set
                    if not getattr(sp, "stop_token_ids", None):
                        sp.stop_token_ids = stop_token_ids

                    # 3) vLLM 0.11.0 wants *inputs*, not keyword prompt_token_ids  :contentReference[oaicite:6]{index=6}
                    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids_list]

                    outputs = llm.generate(
                        inputs,                   # <-- positional, NOT prompt_token_ids=...
                        sampling_params=sp,
                        use_tqdm=_USE_TQDM,
                    )

                    # print("Outputs", outputs)
                    # texts = [o.outputs[0].text if o.outputs else "" for o in outputs]

                    raw_texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
                    texts = [extract_harmony_final(t) for t in raw_texts]
                    # print(texts)
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
        self._enabled = should_enable_board()
        self._dirty = True
        self._error_message: Optional[str] = None
    def update(self, wid: int, text: str):
        if 0 <= wid < len(self.state):
            self.state[wid] = text
            self._dirty = True
        self.render()
    def set_error(self, message: str):
        self._error_message = message
        self._dirty = True
        self.render(force=True)
    def render(self, force: bool = False):
        if not (force or self._enabled or self._error_message):
            return
        if not force and not self._dirty:
            return
        if self._enabled:
            _clear_output(wait=True)
        if self._error_message is not None:
            print("== Worker error ==")
            print(self._error_message)
        else:
            print("== Worker logs ==")
            for i, line in enumerate(self.state):
                print(f"[Worker {i}] {line}")
        self._dirty = False

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
        self.llm_kwargs = dict(llm_kwargs)
        self.result_q: mp.Queue = self.ctx.Queue()
        self.task_queues: List[mp.Queue] = [
            self.ctx.Queue(maxsize=max_request_per_worker) for _ in device_groups
        ]
        self.procs: List[mp.Process] = []
        self._rr = 0

        for wid, devs in enumerate(device_groups):
            p = self.ctx.Process(
                target=_worker_main,
                args=(wid, devs, model, dict(self.llm_kwargs), self.task_queues[wid], self.result_q),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a quick vLLM generation demo with LLMWorkerModel.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/gpt-oss-20b",
        help="Path or identifier of the model to load.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs per worker.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="Number of worker processes to launch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts to send per batch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate per prompt.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout (seconds) for the overall generation job.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory fraction to target when loading the model.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Optional list of prompts; defaults to built-in examples when omitted.",
    )
    args = parser.parse_args()

    fallback_prompts = [
        "Summarise the advantages of retrieval-augmented generation in three bullet points.",
        "Provide a short haiku about GPU clusters.",
        "Explain what RMSearch does in one sentence.",
        "List two tips for managing async vLLM workloads.",
    ]
    prompts = args.prompts if args.prompts else fallback_prompts

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"Loading model {args.model_path!r} with {args.num_instances} instance(s)…")
    llm = build_llm(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=10000,
    )

    try:
        print(f"Generating {len(prompts)} prompt(s) with batch size {args.batch_size}…")
        outputs = generate(
            llm,
            prompts,
            sampling_params=sampling,
            batch_size=args.batch_size,
            timeout_s=args.timeout,
        )
        for idx, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
            separator = "-" * 60
            print(separator)
            print(f"Prompt {idx}:\n{prompt}\n")
            print(f"Completion:\n{output.strip()}")
        print("-" * 60)
        print("Done.")
    finally:
        llm.close()
