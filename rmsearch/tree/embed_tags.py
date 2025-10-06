"""Embedding utilities that wrap the vLLM embedding worker pool."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:  # Optional dependency: only available when sentence-transformers is installed
    from sentence_transformers.quantization import quantize_embeddings  # type: ignore
except Exception:  # pragma: no cover - optional import
    quantize_embeddings = None  # type: ignore

from rmsearch.utils.vllm_embed import (
    EmbeddingWorkerModel,
    build_embedding_model,
    embed as _embed_with_model,
)

__all__ = ["embed_tags", "embed_pool_context"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@contextmanager
def embed_pool_context(
    model_name: str,
    *,
    tensor_parallel_size: int,
    num_instances: int,
    device_groups: Optional[List[List[int]]] = None,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    output_to_cpu: bool = False,
) -> Iterable[EmbeddingWorkerModel]:
    pool = build_embedding_model(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=device_groups,
        output_to_cpu=output_to_cpu,
        **(llm_kwargs or {}),
    )
    try:
        yield pool
    finally:
        pool.close()


@torch.no_grad()
def embed_tags(
    tag_records: Iterable[Dict[str, Any]],
    *,
    embed_model_name: str,
    pool: Optional[EmbeddingWorkerModel] = None,
    pool_settings: Optional[Dict[str, Any]] = None,
    save_path_tags: Optional[str] = None,
    save_path_tagmeta: Optional[str] = None,
    quantize_precision: Optional[str] = None,
    reduce_to_dim: Optional[int] = None,
    worker_batch_size: int = 32,
    timeout_s: Optional[float] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Embed every tag string from ``tag_records`` using a vLLM embedding pool.

    ``tag_meta`` structure -> ``[(key_id, tag_index)]`` mapping each embedding row
    back to ``tag_records``.
    """

    tag_list = list(tag_records)
    if not tag_list:
        empty = torch.empty(0, 0)
        if save_path_tags:
            torch.save(empty, save_path_tags)
        if save_path_tagmeta:
            import json

            with open(save_path_tagmeta, "w", encoding="utf-8") as handle:
                json.dump([], handle)
        return empty, []

    flat_tags: List[str] = []
    tag_meta: List[Tuple[int, int]] = []
    for rec in tag_list:
        tags = list(rec.get("tags", []))
        key_id = int(rec.get("key_id", 0))
        for tag_idx, tag in enumerate(tags):
            flat_tags.append(tag)
            tag_meta.append((key_id, tag_idx))

    owns_pool = False
    if pool is None:
        if pool_settings is None:
            raise ValueError("Either an embedding pool or pool_settings must be provided.")
        pool = build_embedding_model(
            model_name=embed_model_name,
            tensor_parallel_size=pool_settings["tensor_parallel_size"],
            num_instances=pool_settings["num_instances"],
            device_groups=pool_settings.get("device_groups"),
            output_to_cpu=pool_settings.get("output_to_cpu", False),
            **(pool_settings.get("llm_kwargs") or {}),
        )
        owns_pool = True

    try:
        vectors = _embed_with_model(
            pool,
            flat_tags,
            batch_size=worker_batch_size,
            timeout_s=timeout_s,
        )
    finally:
        if owns_pool and pool is not None:
            pool.close()

    if not vectors:
        embeddings = torch.empty(0, 0, dtype=torch.float32, device=_device())
    else:
        embeddings = torch.tensor(vectors, dtype=torch.float32, device=_device())

    if quantize_precision:
        if quantize_embeddings is None:
            raise RuntimeError("sentence-transformers quantization is not available")
        quantised = quantize_embeddings(embeddings.cpu(), precision=quantize_precision)
        embeddings = torch.tensor(quantised, device=_device())

    if reduce_to_dim is not None and 0 < reduce_to_dim < embeddings.shape[-1]:
        mu = embeddings.mean(dim=0, keepdim=True)
        centred = embeddings - mu
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        projection = vh[:reduce_to_dim, :].T
        embeddings = _normalize(centred @ projection)

    if save_path_tags:
        torch.save(embeddings, save_path_tags)
    if save_path_tagmeta:
        import json

        with open(save_path_tagmeta, "w", encoding="utf-8") as handle:
            json.dump(tag_meta, handle)

    # tag_meta structure -> [(key_id, tag_index)] pairs aligning with embeddings rows
    return embeddings, tag_meta


if __name__ == "__main__":
    class DummyEmbedPool:
        def embed(self, inputs, batch_size=None, timeout_s=None):
            del batch_size, timeout_s
            return [[float(idx + offset) for offset in range(4)] for idx, _ in enumerate(inputs)]

        def close(self):
            pass

    dummy_pool = DummyEmbedPool()
    sample_records = [
        {"key": "Example key", "key_id": 0, "tags": ["alpha", "beta"]},
        {"key": "Another key", "key_id": 1, "tags": ["gamma"]},
    ]

    embeddings, meta = embed_tags(sample_records, embed_model_name="dummy", pool=dummy_pool)
    print("Embeddings shape:", tuple(embeddings.shape))
    print("Tag meta:", meta)
