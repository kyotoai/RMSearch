"""
Combined FastAPI service that loads a reward model and a GPT-OSS generation
model at startup so they can be served simultaneously. The reward model logic
mirrors ``rmsearch.Search`` while generation is delegated to
``utils.vllm_generate``.

By default the service expects at least four GPUs and splits them so each model
consumes two (can be overridden via environment variables). Example launch:

    `nohup uvicorn rmsearch.multi_service:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &`

Endpoints:
  - POST /rmsearch   → relevance ranking using the reward model.
  - POST /generate   → text generation via the GPT-OSS model.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from vllm import SamplingParams

from rmsearch.utils.vllm_reward import build_llm as build_reward_llm
from rmsearch.utils.vllm_reward import search as reward_search
from rmsearch.utils.vllm_generate_gptoss import build_llm as build_generate_llm
from rmsearch.utils.vllm_generate_gptoss import generate as run_generate

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # stable ordering
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"       # <- same as `export CUDA_VISIBLE_DEVICES=0,1`


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
    except Exception as exc:  # pragma: no cover - torch is expected in runtime env
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


# ── Configuration defaults ───────────────────────────────────────────────────
REWARD_MODEL_NAME = os.getenv(
    "MULTI_REWARD_MODEL_NAME",
    os.getenv("RMSEARCH_MODEL_NAME", "/workspace/qwen4b-reward-converted-model"),
)
GEN_MODEL_NAME = os.getenv("MULTI_GEN_MODEL_NAME", "/workspace/gpt-oss-20b")

REWARD_TENSOR_PARALLEL = _int_env("MULTI_REWARD_TENSOR_PARALLEL", 1)
REWARD_PIPELINE_PARALLEL = _int_env("MULTI_REWARD_PIPELINE_PARALLEL", 1)
REWARD_QUERY_BATCH_SIZE = _int_env("MULTI_REWARD_QUERY_BATCH_SIZE", 128)
REWARD_GPU_MEMORY_UTIL = _float_env("MULTI_REWARD_GPU_MEMORY_UTILIZATION", 0.90)

GEN_TENSOR_PARALLEL = _int_env("MULTI_GEN_TENSOR_PARALLEL", 1)
GEN_NUM_INSTANCES = _int_env("MULTI_GEN_INSTANCES", 1)
GEN_DEFAULT_BATCH_SIZE = _int_env("MULTI_GEN_DEFAULT_BATCH_SIZE", 4)
GEN_GPU_MEMORY_UTIL = _float_env("MULTI_GEN_GPU_MEMORY_UTILIZATION", 0.90)

REWARD_DEVICE_GROUPS = _parse_device_groups("MULTI_REWARD_DEVICE_GROUPS")
GEN_DEVICE_GROUPS = _parse_device_groups("MULTI_GEN_DEVICE_GROUPS")

TOTAL_EXPECTED_GPUS = (
    REWARD_TENSOR_PARALLEL * REWARD_PIPELINE_PARALLEL
    + GEN_TENSOR_PARALLEL * GEN_NUM_INSTANCES
)
if TOTAL_EXPECTED_GPUS < 2:
    raise ValueError(
        "Configuration must allocate more than 1 GPU in total for the combined "
        "service. Increase tensor/instance counts."
    )


def _resolve_device_groups() -> Tuple[List[List[int]], List[List[int]]]:
    reward_groups = REWARD_DEVICE_GROUPS
    gen_groups = GEN_DEVICE_GROUPS
    available = _visible_gpus()

    if reward_groups is not None:
        _validate_group_layout(
            reward_groups,
            REWARD_TENSOR_PARALLEL,
            REWARD_PIPELINE_PARALLEL,
            "MULTI_REWARD_DEVICE_GROUPS",
        )
    if gen_groups is not None:
        _validate_group_layout(
            gen_groups,
            GEN_TENSOR_PARALLEL,
            GEN_NUM_INSTANCES,
            "MULTI_GEN_DEVICE_GROUPS",
        )

    used = set(_flatten(reward_groups or [])) | set(_flatten(gen_groups or []))

    if reward_groups is None:
        remaining = [gpu for gpu in available if gpu not in used]
        needed = REWARD_TENSOR_PARALLEL * REWARD_PIPELINE_PARALLEL
        if len(remaining) < needed:
            raise ValueError(
                f"Not enough GPUs to allocate reward model: need {needed}, "
                f"have {len(remaining)} available (all GPUs={available})."
            )
        reward_slice = remaining[:needed]
        reward_groups = _slice_groups(
            reward_slice, REWARD_TENSOR_PARALLEL, REWARD_PIPELINE_PARALLEL
        )
        used.update(_flatten(reward_groups))

    if gen_groups is None:
        remaining = [gpu for gpu in available if gpu not in used]
        needed = GEN_TENSOR_PARALLEL * GEN_NUM_INSTANCES
        if len(remaining) < needed:
            raise ValueError(
                f"Not enough GPUs to allocate generation model: need {needed}, "
                f"have {len(remaining)} available (all GPUs={available})."
            )
        gen_slice = remaining[:needed]
        gen_groups = _slice_groups(
            gen_slice,
            GEN_TENSOR_PARALLEL,
            GEN_NUM_INSTANCES,
        )

    # Ensure there is no GPU reuse across the two models.
    all_used = _flatten(reward_groups) + _flatten(gen_groups)
    if len(all_used) != len(set(all_used)):
        raise ValueError(
            "GPU device groups overlap between reward and generation models. "
            "Adjust MULTI_*_DEVICE_GROUPS to avoid shared GPUs."
        )
    return reward_groups, gen_groups


# ── Reward model wrapper ─────────────────────────────────────────────────────
class Search:
    """Formats requests for the reward model and proxies to vLLM."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        *,
        device_groups: List[List[int]],
        query_batch_size: int = REWARD_QUERY_BATCH_SIZE,
        **llm_kwargs: Any,
    ) -> None:
        self.model = build_reward_llm(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            num_instances=pipeline_parallel_size,
            device_groups=device_groups,
            runner="pooling",
            gpu_memory_utilization=REWARD_GPU_MEMORY_UTIL,
            **llm_kwargs,
        )
        self.tokenizer = self.model.tokenizer
        self.query_batch_size = query_batch_size

    def _format_chat_history(self, query: "ChatQuery") -> str:
        segments = [f"{turn.role}: {turn.content}" for turn in query.message]
        return "\n".join(segments)

    def _normalise_query(self, query: "QueryInput") -> str:
        if isinstance(query, str):
            return query
        if isinstance(query, ChatQuery):
            return self._format_chat_history(query)
        raise TypeError(f"Unsupported query type: {type(query)!r}")

    def _llm_template(self, row: Dict[str, Any]) -> str:
        message = [
            {
                "role": "user",
                "content": (
                    "Provide a relevance score between the query and the sentence.\n\n"
                    f"Query: {row['query']}\n\n"
                    f"Sentence: {row['key']}"
                ),
            }
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False)

    async def __call__(
        self,
        queries: Sequence["QueryInput"],
        keys: Sequence[str],
        *,
        k: int,
        batch_size: Optional[int] = None,
        query_batch_size: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not keys:
            raise ValueError("keys must contain at least one entry.")

        normalized_queries = [self._normalise_query(query) for query in queries]
        requests = [{"query": q, "keys": list(keys)} for q in normalized_queries]

        effective_query_batch = query_batch_size or self.query_batch_size
        if batch_size is not None:
            gen_kwargs["batch_size"] = batch_size

        def _run_search() -> List[Dict[str, Any]]:
            return reward_search(
                self.model,
                requests,
                self._llm_template,
                topk=k,
                query_batch_size=effective_query_batch,
                **gen_kwargs,
            )

        return await asyncio.to_thread(_run_search)

    def close(self, *, kill: bool = False) -> None:
        self.model.close(kill=kill)


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
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatQuery(BaseModel):
    message: List[ChatMessage]


QueryInput = Union[str, ChatQuery]


class SearchRequest(BaseModel):
    queries: List[QueryInput]
    keys: Optional[List[str]] = None
    k: int = Field(default=5, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    query_batch_size: Optional[int] = Field(default=None, ge=1)


class KeyOut(BaseModel):
    key_id: int
    key: str
    relevance: float


class QueryOut(BaseModel):
    query: str
    query_id: int
    keys: List[KeyOut]


DEFAULT_KEYS: List[str] = (
    ["LLM is Large Language Model which can be made ..." * 7, "Japanese capital is ..." * 7]
    * 5
)


class GenerateRequest(BaseModel):
    prompts: List[str]
    temperature: Optional[float] = Field(default=0.7, ge=0.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=256, ge=1)
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


# ── FastAPI lifecycle --------------------------------------------------------
app = FastAPI()

_engine_lock = asyncio.Lock()
_search_engine: Optional[Search] = None
_generation_engine: Optional[Generator] = None


async def _initialise_engines() -> Tuple[Search, Generator]:
    global _search_engine, _generation_engine
    if _search_engine is not None and _generation_engine is not None:
        return _search_engine, _generation_engine

    reward_groups, gen_groups = _resolve_device_groups()

    async def _build_search() -> Search:
        return await asyncio.to_thread(
            Search,
            REWARD_MODEL_NAME,
            REWARD_TENSOR_PARALLEL,
            REWARD_PIPELINE_PARALLEL,
            device_groups=reward_groups,
        )

    async def _build_generator() -> Generator:
        return await asyncio.to_thread(
            Generator,
            GEN_MODEL_NAME,
            GEN_TENSOR_PARALLEL,
            GEN_NUM_INSTANCES,
            device_groups=gen_groups,
        )

    search_obj, generator_obj = await asyncio.gather(_build_search(), _build_generator())
    _search_engine, _generation_engine = search_obj, generator_obj
    return search_obj, generator_obj


async def _get_search_engine() -> Search:
    global _search_engine, _generation_engine
    if _search_engine is not None:
        return _search_engine
    async with _engine_lock:
        if _search_engine is None or _generation_engine is None:
            _search_engine, _generation_engine = await _initialise_engines()
    return _search_engine


async def _get_generation_engine() -> Generator:
    global _generation_engine, _search_engine
    if _generation_engine is not None:
        return _generation_engine
    async with _engine_lock:
        if _generation_engine is None or _search_engine is None:
            _search_engine, _generation_engine = await _initialise_engines()
    return _generation_engine


async def _shutdown_engines() -> None:
    global _search_engine, _generation_engine
    if _search_engine is not None:
        await asyncio.to_thread(_search_engine.close)
        _search_engine = None
    if _generation_engine is not None:
        await asyncio.to_thread(_generation_engine.close)
        _generation_engine = None


@app.on_event("startup")
async def _startup() -> None:
    async with _engine_lock:
        await _initialise_engines()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _shutdown_engines()


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/rmsearch", response_model=List[QueryOut])
async def rmsearch_endpoint(req: SearchRequest) -> List[QueryOut]:
    keys = req.keys or DEFAULT_KEYS
    engine = await _get_search_engine()

    results = await engine(
        req.queries,
        keys,
        k=req.k,
        batch_size=req.batch_size,
        query_batch_size=req.query_batch_size,
    )

    response: List[QueryOut] = []
    for row in results:
        key_objs = [
            KeyOut(
                key_id=int(key_info["key_id"]),
                key=key_info["key"],
                relevance=float(key_info.get("relevance", 0.0)),
            )
            for key_info in row["keys"]
        ]
        response.append(
            QueryOut(
                query=row["query"],
                query_id=int(row["query_id"]),
                keys=key_objs,
            )
        )

    return response


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
