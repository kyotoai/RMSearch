"""Shared training utilities ported from the original notebook pipeline."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ENGINE_CACHE: Dict[Tuple[str, int, int, int, float], Tuple[Any, Any]] = {}


def setup_async_engine(
    model_name: str,
    *,
    tensor_parallel_size: int = 2,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    omp_num_threads: int = 4,
    disable_log_stats: bool = True,
) -> Tuple[Any, Any]:
    """Lazy-load and cache an AsyncLLMEngine/tokenizer pair.

    Returns ``(engine, tokenizer)`` so callers can reuse the same backend.
    """

    cache_key = (model_name, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, gpu_memory_utilization)
    if cache_key in _ENGINE_CACHE:
        return _ENGINE_CACHE[cache_key]

    os.environ.setdefault("OMP_NUM_THREADS", str(omp_num_threads))

    from transformers import AutoTokenizer  # Imported lazily to keep optional dependency lightweight
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=disable_log_stats,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine.log_requests = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    _ENGINE_CACHE[cache_key] = (engine, tokenizer)
    return engine, tokenizer


class AllRequests:
    """Notebook-friendly prompt scheduler with checkpointing support."""

    def __init__(self, max_request: int, *, engine_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.max_request = max_request
        self.engine_kwargs = engine_kwargs or {}
        self.requests: List[Dict[str, Any]] = []
        self.request_ids: List[int] = []
        self.request_id = 0
        self.results: List[Dict[str, Any]] = []
        self.finished_ids: List[int] = []
        self.progress_bar = None

    def add(self, request: Dict[str, Any]) -> None:
        self.requests.append(request)
        self.request_ids.append(self.request_id)
        self.request_id += 1

    async def process(
        self,
        *,
        model_name: Optional[str] = None,
        max_tokens: int = 3000,
        temperature: float = 0.4,
        save_dir: str = "progress_log",
        restart: bool = False,
    ) -> List[Dict[str, Any]]:
        from pathlib import Path

        from tqdm import tqdm
        from vllm import SamplingParams

        if model_name is not None:
            engine_kwargs = dict(self.engine_kwargs)
            engine_kwargs["model_name"] = model_name
        else:
            engine_kwargs = dict(self.engine_kwargs)
            if "model_name" not in engine_kwargs:
                raise ValueError("model_name must be supplied via engine_kwargs or process(model_name=...)")

        engine, _ = setup_async_engine(**engine_kwargs)

        os.makedirs(save_dir, exist_ok=True)
        save_path = Path(save_dir)

        if restart:
            finished_path = save_path / "finished_ids.json"
            results_path = save_path / "results.json"
            if finished_path.exists() and results_path.exists():
                self.finished_ids = json.loads(finished_path.read_text())
                self.results = json.loads(results_path.read_text())
                keep_requests: List[Dict[str, Any]] = []
                keep_ids: List[int] = []
                for rid, request in zip(self.request_ids, self.requests):
                    if rid in self.finished_ids:
                        continue
                    keep_ids.append(rid)
                    keep_requests.append(request)
                self.request_ids = keep_ids
                self.requests = keep_requests
        else:
            for filename in ("finished_ids.json", "results.json"):
                path = save_path / filename
                if path.exists():
                    path.unlink()

        total = len(self.requests)
        self.progress_bar = tqdm(total=total, desc="Processing Requests")

        async def worker() -> None:
            while self.requests:
                request_dict = self.requests.pop(0)
                request_id = self.request_ids.pop(0)
                prompt = request_dict["prompt"]

                generator = engine.generate(
                    prompt,
                    SamplingParams(temperature=temperature, max_tokens=max_tokens),
                    request_id,
                )
                final_output = None
                async for request_output in generator:
                    final_output = request_output

                output_text = final_output.outputs[0].text if final_output else ""
                record = dict(request_dict)
                record["output"] = output_text
                self.results.append(record)
                self.finished_ids.append(request_id)

                (save_path / "results.json").write_text(json.dumps(self.results))
                (save_path / "finished_ids.json").write_text(json.dumps(self.finished_ids))

                self.progress_bar.update(1)

        await asyncio.gather(*[worker() for _ in range(max(self.max_request, 1))])
        self.progress_bar.close()
        return self.results


def extract_text(text: str, tag_name: str) -> Optional[str]:
    import re

    pattern = rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else None


def extract_int(text: str) -> Optional[int]:
    import re

    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def datasetdict_to_pandas(dataset_dict: Any):  # pragma: no cover - convenience bridge
    import pandas as pd

    frames = []
    for split_name, split_ds in dataset_dict.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def convert_model(
    base_model_name: str,
    checkpoint_path: str,
    *,
    output_dir: Path,
    score_filename: str = "score.pt",
) -> Path:
    """Merge LoRA adapters into the base reward model weights and save artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / score_filename

    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left", add_eos_token=False, add_bos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
    lora_model = PeftModel.from_pretrained(model, checkpoint_path)
    reward_model = lora_model.merge_and_unload()

    tokenizer.save_pretrained(output_dir)
    reward_model.save_pretrained(output_dir)
    torch.save(reward_model.score.weight.data, score_path)
    del reward_model

    generate_model = AutoModelForCausalLM.from_pretrained(output_dir)
    generate_model.save_pretrained(output_dir)
    del generate_model

    return output_dir


if __name__ == "__main__":
    sample = "<tag>Hello</tag>"
    print("extract_text:", extract_text(sample, "tag"))
    print("extract_int:", extract_int("Pair 42"))
    print("convert_model would save to:", (Path("./demo_out")))
