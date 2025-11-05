"""
RMSearch FastAPI service exposing reward-model ranking over a REST endpoint.

Run the server after installing RMSearch with:

    uvicorn rmsearch:app --host 0.0.0.0 --port 8000


# Example usage:
# ---------------------------------------------------------------------------
# # 1) Direct async usage inside your application
# import asyncio
# from rmsearch import Search
#
# async def main():
#     search = Search(
#         model_name="/workspace/llama3b-rm",
#         tensor_parallel_size=1,
#         pipeline_parallel_size=1,
#     )
#     queries = ["Summarise retrieval augmented generation."]
#     keys = [
#         "Retrieval augmented generation (RAG) combines external documents with LLMs.",
#         "An unrelated sentence about cooking pasta.",
#     ]
#     results = await search(queries, keys, k=1)
#     search.close()
#     print(results[0]["keys"][0])
#
# asyncio.run(main())
#
# # 2) Provide chat-form queries
# async def with_chat_queries():
#     search = Search(
#         model_name="/workspace/llama3b-rm",
#         tensor_parallel_size=1,
#         pipeline_parallel_size=1,
#         query_batch_size=32,
#     )
#     queries = [
#         {
#             "message": [
#                 {"role": "user", "content": "Suggest a healthy hiking snack."},
#                 {"role": "assistant", "content": "Trail mix and jerky are good."},
#                 {"role": "user", "content": "Rank these options."},
#             ]
#         }
#     ]
#     keys = ["Trail mix is nutrient dense.", "Pack extra batteries."]
#     ranked = await search(queries, keys, k=2, batch_size=2)
#     search.close()
#     return ranked
#
# asyncio.run(with_chat_queries())
#
# # 3) Call the HTTP API once uvicorn is running
curl -X POST http://localhost:8000/rmsearch \
  -H "Content-Type: application/json" \
  -d '{"queries": ["How to tune a reward model?"], "keys": ["Reward models score sequences."]}'
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rmsearch.utils.vllm_reward import build_llm, search as reward_search


# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI()


# ── Configuration defaults (overridable via environment variables) ───────────
DEFAULT_MODEL_NAME = os.getenv("RMSEARCH_MODEL_NAME", "/workspace/llama3b-rm-converted-model")
DEFAULT_TENSOR_PARALLEL = int(os.getenv("RMSEARCH_TENSOR_PARALLEL", "1"))
DEFAULT_PIPELINE_PARALLEL = int(os.getenv("RMSEARCH_PIPELINE_PARALLEL", "1"))
DEFAULT_QUERY_BATCH_SIZE = int(os.getenv("RMSEARCH_QUERY_BATCH_SIZE", "128"))


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


# ── Search helper built on the vLLM reward worker ────────────────────────────
class Search:
    """Thin wrapper that formats requests for `utils.vllm_reward.search`."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        *,
        query_batch_size: int = DEFAULT_QUERY_BATCH_SIZE,
        **llm_kwargs: Any,
    ) -> None:
        self.model = build_llm(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            num_instances=pipeline_parallel_size,
            **llm_kwargs,
        )
        self.tokenizer = self.model.tokenizer
        self.query_batch_size = query_batch_size

    def _format_chat_history(self, query: ChatQuery) -> str:
        """Flatten chat history into a deterministic text block."""
        segments = [f"{turn.role}: {turn.content}" for turn in query.message]
        return "\n".join(segments)

    def _normalise_query(self, query: QueryInput) -> str:
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
        queries: Sequence[QueryInput],
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


# ── Search engine lifecycle helpers ──────────────────────────────────────────
search_engine: Optional[Search] = None
_search_lock = asyncio.Lock()


async def _create_search_engine() -> Search:
    return await asyncio.to_thread(
        Search,
        DEFAULT_MODEL_NAME,
        DEFAULT_TENSOR_PARALLEL,
        DEFAULT_PIPELINE_PARALLEL,
    )


async def get_search_engine() -> Search:
    global search_engine
    if search_engine is not None:
        return search_engine
    async with _search_lock:
        if search_engine is None:
            search_engine = await _create_search_engine()
    return search_engine


async def shutdown_search_engine() -> None:
    global search_engine
    if search_engine is None:
        return

    engine = search_engine
    search_engine = None
    await asyncio.to_thread(engine.close)


@app.on_event("startup")
async def _startup() -> None:
    await get_search_engine()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await shutdown_search_engine()


# ── Endpoint ─────────────────────────────────────────────────────────────────
@app.post("/rmsearch", response_model=List[QueryOut])
async def rmsearch(req: SearchRequest) -> List[QueryOut]:
    """
    Request bodies can provide either plain string queries or chat histories.

    Examples:

    1) Provide only queries (server uses default keys):
        {"queries": ["How to make LLM?", "What's the capital of Japan?"]}

    2) Provide queries and custom keys:
        {
          "queries": ["How to make LLM?"],
          "keys":    ["LLM is Large Language Model ..."]
        }

    3) Chat-style queries (each item is a message list):
        {
          "queries": [
            {"message": [{"role": "user", "content": "How to make LLM?"}]},
            {"message": [
              {"role": "user", "content": "What's the capital of Japan?"},
              {"role": "assistant", "content": "The capital of Japan is ..."},
              {"role": "user", "content": "What's the historical reason behind it?"}
            ]}
          ]
        }
    """

    keys = req.keys or DEFAULT_KEYS
    engine = await get_search_engine()
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


__all__ = ["app", "Search"]
