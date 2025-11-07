"""RMSearch FastAPI service exposing embedding-based relevance scoring.

Run with:

    nohup uvicorn rmsearch.embed_search:app --host 0.0.0.0 --port 8000 \
        > embed_server.log 2>&1 &

Example request:
    curl -X POST http://localhost:8000/embed \
      -H "Content-Type: application/json" \
      -d '{"queries": ["What is LLM?"], "keys": ["LLM means large language model."]}'
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rmsearch.utils.vllm_embed import build_embedding_model, embed as embed_with_model


# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _parse_device_groups(
    spec: Optional[str],
    tensor_parallel_size: int,
    num_instances: int,
) -> Optional[List[List[int]]]:
    if not spec:
        return None
    groups: List[List[int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        group = [int(token) for token in chunk.split(",") if token.strip()]
        if group:
            groups.append(group)
    if not groups:
        return None
    if len(groups) != num_instances:
        raise ValueError(
            f"Expected {num_instances} device groups, got {len(groups)} "
            f"(spec={spec!r})"
        )
    for group in groups:
        if len(group) != tensor_parallel_size:
            raise ValueError(
                "Each device group must contain exactly "
                f"{tensor_parallel_size} devices (got {group})"
            )
    return groups


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}."
        )
    return float(math.fsum(ax * bx for ax, bx in zip(a, b)))


# ── Configuration defaults (overridable via environment variables) ───────────
DEFAULT_EMBED_MODEL_NAME = os.getenv(
    "RMSEARCH_EMBED_MODEL_NAME",
    "intfloat/e5-mistral-7b-instruct",
)
DEFAULT_EMBED_TENSOR_PARALLEL = int(os.getenv("RMSEARCH_EMBED_TENSOR_PARALLEL", "1"))
DEFAULT_EMBED_INSTANCES = int(os.getenv("RMSEARCH_EMBED_INSTANCES", "1"))
DEFAULT_WORKER_BATCH_SIZE = int(os.getenv("RMSEARCH_EMBED_BATCH_SIZE", "32"))
_embed_timeout_env = os.getenv("RMSEARCH_EMBED_TIMEOUT_S")
DEFAULT_EMBED_TIMEOUT: Optional[float] = (
    float(_embed_timeout_env) if _embed_timeout_env else None
)
DEFAULT_DEVICE_GROUPS = _parse_device_groups(
    os.getenv("RMSEARCH_EMBED_DEVICE_GROUPS"),
    DEFAULT_EMBED_TENSOR_PARALLEL,
    DEFAULT_EMBED_INSTANCES,
)


# ── Fallback keys (used when the caller omits `keys`) ────────────────────────
DEFAULT_KEYS: List[str] = (
    ["LLM is Large Language Model which can be made ..." * 7,
     "Japanese capital is ..." * 7] * 5
)


# ── Request / Response schemas ───────────────────────────────────────────────
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


# ── Embedding-based search helper ────────────────────────────────────────────
class EmbedSearch:
    """Compute relevance via dot products of query/key embeddings."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        num_instances: int,
        *,
        worker_batch_size: int = DEFAULT_WORKER_BATCH_SIZE,
        device_groups: Optional[List[List[int]]] = None,
        timeout_s: Optional[float] = DEFAULT_EMBED_TIMEOUT,
        **llm_kwargs: Any,
    ) -> None:
        self.pool = build_embedding_model(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            num_instances=num_instances,
            device_groups=device_groups,
            output_to_cpu=True,
            **llm_kwargs,
        )
        self.worker_batch_size = worker_batch_size
        self.timeout_s = timeout_s

    def _format_chat_history(self, query: ChatQuery) -> str:
        segments = [f"{turn.role}: {turn.content}" for turn in query.message]
        return "\n".join(segments)

    def _normalise_query(self, query: QueryInput) -> str:
        if isinstance(query, str):
            return query
        if isinstance(query, ChatQuery):
            return self._format_chat_history(query)
        raise TypeError(f"Unsupported query type: {type(query)!r}")

    def _embed_sync(
        self,
        texts: Sequence[str],
        batch_size: Optional[int],
    ) -> List[List[float]]:
        payload = list(texts)
        if not payload:
            return []
        effective_batch = batch_size or self.worker_batch_size
        return embed_with_model(
            self.pool,
            payload,
            batch_size=effective_batch,
            timeout_s=self.timeout_s,
        )

    async def __call__(
        self,
        queries: Sequence[QueryInput],
        keys: Sequence[str],
        *,
        k: int,
        batch_size: Optional[int] = None,
        query_batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not queries:
            raise ValueError("queries must contain at least one entry.")
        if not keys:
            raise ValueError("keys must contain at least one entry.")

        normalized_queries = [self._normalise_query(query) for query in queries]
        key_strings = list(keys)

        key_batch = batch_size or self.worker_batch_size
        query_batch = query_batch_size or key_batch

        query_vecs, key_vecs = await asyncio.gather(
            asyncio.to_thread(self._embed_sync, normalized_queries, query_batch),
            asyncio.to_thread(self._embed_sync, key_strings, key_batch),
        )

        if len(key_vecs) != len(key_strings):
            raise RuntimeError(
                f"Key embedding mismatch: expected {len(key_strings)}, got {len(key_vecs)}"
            )
        if len(query_vecs) != len(normalized_queries):
            raise RuntimeError(
                f"Query embedding mismatch: expected {len(normalized_queries)}, got {len(query_vecs)}"
            )

        key_records = list(zip(range(len(key_strings)), key_strings, key_vecs))
        top_limit = min(k, len(key_records))

        output: List[Dict[str, Any]] = []
        for q_idx, (query_str, q_vec) in enumerate(zip(normalized_queries, query_vecs)):
            scored: List[Dict[str, Any]] = []
            for key_id, key_text, key_vec in key_records:
                relevance = _dot(q_vec, key_vec)
                scored.append(
                    {"key_id": key_id, "key": key_text, "relevance": float(relevance)}
                )
            scored.sort(key=lambda item: item["relevance"], reverse=True)
            output.append(
                {
                    "query": query_str,
                    "query_id": q_idx,
                    "keys": scored[:top_limit],
                }
            )
        return output

    def close(self, *, kill: bool = False) -> None:
        self.pool.close(kill=kill)


# ── Search engine lifecycle helpers ──────────────────────────────────────────
embed_engine: Optional[EmbedSearch] = None
_embed_lock = asyncio.Lock()


async def _create_embed_engine() -> EmbedSearch:
    return await asyncio.to_thread(
        EmbedSearch,
        DEFAULT_EMBED_MODEL_NAME,
        DEFAULT_EMBED_TENSOR_PARALLEL,
        DEFAULT_EMBED_INSTANCES,
        worker_batch_size=DEFAULT_WORKER_BATCH_SIZE,
        device_groups=DEFAULT_DEVICE_GROUPS,
    )


async def get_embed_engine() -> EmbedSearch:
    global embed_engine
    if embed_engine is not None:
        return embed_engine
    async with _embed_lock:
        if embed_engine is None:
            embed_engine = await _create_embed_engine()
    return embed_engine


async def shutdown_embed_engine() -> None:
    global embed_engine
    if embed_engine is None:
        return
    engine = embed_engine
    embed_engine = None
    await asyncio.to_thread(engine.close)


@app.on_event("startup")
async def _startup() -> None:
    await get_embed_engine()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await shutdown_embed_engine()


# ── Endpoint ─────────────────────────────────────────────────────────────────
@app.post("/embed", response_model=List[QueryOut])
async def embed_endpoint(req: SearchRequest) -> List[QueryOut]:
    keys = req.keys or DEFAULT_KEYS
    engine = await get_embed_engine()
    output = await engine(
        req.queries,
        keys,
        k=req.k,
        batch_size=req.batch_size,
        query_batch_size=req.query_batch_size,
    )

    response: List[QueryOut] = []
    for row in output:
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


__all__ = ["app", "EmbedSearch"]
