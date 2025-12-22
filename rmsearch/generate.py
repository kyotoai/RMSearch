"""
FastAPI service that loads a GPT-OSS generation model at startup and exposes
only a single generation endpoint:

  - POST /generate → text generation via the GPT-OSS model.

Run:
  nohup uvicorn rmsearch.generate:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

* When killing nohop
`ps aux | grep uvicorn`
`kill -9 <pid>`

Example usage:
----------------------------------------------------------------------------
# 1) Python (async) call to /generate
import asyncio
import httpx

async def main():
    payload = {
        "prompts": ["Suggest three remote team rituals."],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:8000/generate", json=payload)
        resp.raise_for_status()
        print(resp.json())

asyncio.run(main())

# 2) cURL request to /generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompts": ["List two prompts for evaluating factual recall."],
        "temperature": 0.2,
        "max_tokens": 3000
      }'
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from vllm import SamplingParams

from rmsearch.utils.vllm_generate_gptoss import build_llm as build_generate_llm
from rmsearch.utils.vllm_generate_gptoss import generate as run_generate


# ── Environment helpers ──────────────────────────────────────────────────────
def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer (got {raw!r}).") from exc


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be numeric (got {raw!r}).") from exc


def _parse_device_groups(var_name: str) -> Optional[List[List[int]]]:
    raw = os.getenv(var_name)
    if not raw:
        return None
    groups: List[List[int]] = []
    for chunk in raw.split("|"):
        stripped = [piece.strip() for piece in chunk.split(",") if piece.strip()]
        if not stripped:
            continue
        groups.append([int(piece) for piece in stripped])
    if not groups:
        raise ValueError(f"{var_name} must contain at least one device id.")
    return groups


def _visible_gpus() -> List[int]:
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(part.strip()) for part in visible.split(",") if part.strip()]

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is unset and torch is unavailable; cannot "
            "determine GPU inventory."
        ) from exc

    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("No CUDA devices detected.")
    return list(range(count))


def _flatten(groups: Iterable[Iterable[int]]) -> List[int]:
    return [idx for group in groups for idx in group]


def _slice_groups(
    device_ids: List[int],
    tensor_parallel_size: int,
    num_instances: int,
) -> List[List[int]]:
    expected = tensor_parallel_size * num_instances
    if len(device_ids) < expected:
        raise ValueError(
            f"Need {expected} GPU ids, only received {len(device_ids)}: {device_ids}"
        )
    groups: List[List[int]] = []
    for i in range(num_instances):
        start = i * tensor_parallel_size
        groups.append(device_ids[start : start + tensor_parallel_size])
    return groups


def _validate_group_layout(
    groups: List[List[int]],
    tensor_parallel_size: int,
    num_instances: int,
    label: str,
) -> None:
    if len(groups) != num_instances:
        raise ValueError(
            f"{label} must provide {num_instances} groups; received {len(groups)}"
        )
    for group in groups:
        if len(group) != tensor_parallel_size:
            raise ValueError(
                f"{label} group {group} must contain {tensor_parallel_size} GPUs."
            )


# ── Configuration defaults (kept consistent with your original env names) ─────
GEN_MODEL_NAME = os.getenv("MULTI_GEN_MODEL_NAME", "/workspace/gpt-oss-20b")

GEN_TENSOR_PARALLEL = _int_env("MULTI_GEN_TENSOR_PARALLEL", 1)
GEN_NUM_INSTANCES = _int_env("MULTI_GEN_INSTANCES", 1)
GEN_DEFAULT_BATCH_SIZE = _int_env("MULTI_GEN_DEFAULT_BATCH_SIZE", 4)
GEN_GPU_MEMORY_UTIL = _float_env("MULTI_GEN_GPU_MEMORY_UTILIZATION", 0.90)

GEN_DEVICE_GROUPS = _parse_device_groups("MULTI_GEN_DEVICE_GROUPS")

TOTAL_EXPECTED_GPUS = GEN_TENSOR_PARALLEL * GEN_NUM_INSTANCES
if TOTAL_EXPECTED_GPUS < 1:
    raise ValueError(
        "Configuration must allocate at least 1 GPU for the generation service."
    )


def _resolve_device_groups() -> List[List[int]]:
    gen_groups = GEN_DEVICE_GROUPS
    available = _visible_gpus()

    if gen_groups is not None:
        _validate_group_layout(
            gen_groups,
            GEN_TENSOR_PARALLEL,
            GEN_NUM_INSTANCES,
            "MULTI_GEN_DEVICE_GROUPS",
        )
        return gen_groups

    needed = GEN_TENSOR_PARALLEL * GEN_NUM_INSTANCES
    if len(available) < needed:
        raise ValueError(
            f"Not enough GPUs to allocate generation model: need {needed}, "
            f"have {len(available)} (all GPUs={available})."
        )
    gen_slice = available[:needed]
    return _slice_groups(gen_slice, GEN_TENSOR_PARALLEL, GEN_NUM_INSTANCES)


# ── Generation model wrapper ─────────────────────────────────────────────────
class Generator:
    """Thin proxy around the GPT-OSS generation worker."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        num_instances: int,
        *,
        device_groups: List[List[int]],
        **llm_kwargs: Any,
    ) -> None:
        self.model = build_generate_llm(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            num_instances=num_instances,
            device_groups=device_groups,
            gpu_memory_utilization=GEN_GPU_MEMORY_UTIL,
            **llm_kwargs,
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling_params: SamplingParams,
        batch_size: Optional[int],
        timeout_s: Optional[float],
    ) -> List[str]:
        return run_generate(
            self.model,
            list(prompts),
            sampling_params=sampling_params,
            batch_size=batch_size or GEN_DEFAULT_BATCH_SIZE,
            timeout_s=timeout_s,
        )

    def close(self, *, kill: bool = False) -> None:
        self.model.close(kill=kill)


# ── Pydantic schemas ────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompts: List[str]
    temperature: Optional[float] = Field(default=0.7, ge=0.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=3000, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    timeout_s: Optional[float] = Field(default=180.0, ge=0.0)

    @validator("prompts")
    def _non_empty_prompts(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("prompts must contain at least one entry.")
        return value


class GenerateOut(BaseModel):
    prompt_id: int
    prompt: str
    text: str


# ── FastAPI lifecycle ────────────────────────────────────────────────────────
app = FastAPI()

_engine_lock = asyncio.Lock()
_generation_engine: Optional[Generator] = None


async def _initialise_engine() -> Generator:
    global _generation_engine
    if _generation_engine is not None:
        return _generation_engine

    gen_groups = _resolve_device_groups()

    async def _build_generator() -> Generator:
        return await asyncio.to_thread(
            Generator,
            GEN_MODEL_NAME,
            GEN_TENSOR_PARALLEL,
            GEN_NUM_INSTANCES,
            device_groups=gen_groups,
        )

    _generation_engine = await _build_generator()
    return _generation_engine


async def _get_generation_engine() -> Generator:
    global _generation_engine
    if _generation_engine is not None:
        return _generation_engine
    async with _engine_lock:
        if _generation_engine is None:
            _generation_engine = await _initialise_engine()
    return _generation_engine


async def _shutdown_engine() -> None:
    global _generation_engine
    if _generation_engine is not None:
        await asyncio.to_thread(_generation_engine.close)
        _generation_engine = None


@app.on_event("startup")
async def _startup() -> None:
    async with _engine_lock:
        await _initialise_engine()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _shutdown_engine()


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=List[GenerateOut])
async def generate_endpoint(req: GenerateRequest) -> List[GenerateOut]:
    engine = await _get_generation_engine()
    sampling_kwargs: Dict[str, Any] = {}
    if req.temperature is not None:
        sampling_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        sampling_kwargs["top_p"] = req.top_p
    if req.max_tokens is not None:
        sampling_kwargs["max_tokens"] = req.max_tokens
    sampling_params = SamplingParams(**sampling_kwargs)

    try:
        outputs = await asyncio.to_thread(
            engine.generate,
            req.prompts,
            sampling_params=sampling_params,
            batch_size=req.batch_size,
            timeout_s=req.timeout_s,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [
        GenerateOut(prompt_id=i, prompt=prompt, text=text)
        for i, (prompt, text) in enumerate(zip(req.prompts, outputs))
    ]


__all__ = ["app"]
