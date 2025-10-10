"""Embedding utilities that wrap the vLLM embedding worker pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
        with open(save_path_tagmeta, "w", encoding="utf-8") as handle:
            json.dump(tag_meta, handle)

    # embeddings: torch.Tensor shaped [num_tags, hidden_dim] in model dtype/device.
    # tag_meta (list): [(key_id, tag_index)] so row `i` of `embeddings` corresponds to
    #   tag_records[tag_meta[i][0]]["tags"][tag_meta[i][1]].
    return embeddings, tag_meta


def _parse_device_groups(spec: Optional[str], tensor_parallel_size: int, num_instances: int) -> Optional[List[List[int]]]:
    if not spec:
        return None
    groups: List[List[int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        group = [int(token) for token in chunk.split(",") if token.strip()]
        if not group:
            continue
        groups.append(group)
    if not groups:
        return None
    if len(groups) != num_instances:
        raise ValueError(f"Expected {num_instances} device groups, got {len(groups)}")
    for group in groups:
        if len(group) != tensor_parallel_size:
            raise ValueError(
                "Each device group must contain exactly "
                f"{tensor_parallel_size} devices (got {group})"
            )
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed tag strings using a vLLM embedding worker pool.")
    parser.add_argument("--tag-records", type=Path, required=True, help="Path to tag records JSON (from generate_tag).")
    parser.add_argument("--embeddings-out", type=Path, required=True, help="Destination path for the embeddings tensor (.pt).")
    parser.add_argument("--tag-meta-out", type=Path, required=True, help="Destination path for the tag metadata JSON.")
    parser.add_argument("--model-name", type=str, required=True, help="Embedding model name or path.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor_parallel_size per embedding instance.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of embedding worker instances.")
    parser.add_argument(
        "--device-groups",
        type=str,
        help="Explicit GPU mapping, e.g. '0,1;2,3' for two workers with tensor_parallel_size=2.",
    )
    parser.add_argument("--worker-batch-size", type=int, default=32, help="Inputs processed per worker batch.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional timeout (s) for a worker batch.")
    parser.add_argument("--quantize", type=str, default=None, help="Quantization precision (requires sentence-transformers).")
    parser.add_argument("--reduce-dim", type=int, default=None, help="Optional dimensionality reduction via truncated SVD.")
    args = parser.parse_args()

    if not args.tag_records.exists():
        raise FileNotFoundError(f"Tag records not found: {args.tag_records}")

    tag_records = json.loads(args.tag_records.read_text())

    device_groups = _parse_device_groups(
        args.device_groups,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
    )

    pool_settings = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "num_instances": args.num_instances,
        "device_groups": device_groups,
        "output_to_cpu": True,
        "llm_kwargs": {},
    }

    embeddings, meta = embed_tags(
        tag_records,
        embed_model_name=args.model_name,
        pool_settings=pool_settings,
        worker_batch_size=args.worker_batch_size,
        timeout_s=args.timeout,
        quantize_precision=args.quantize,
        reduce_to_dim=args.reduce_dim,
    )

    args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.embeddings_out)
    args.tag_meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.tag_meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved embeddings to {args.embeddings_out} and metadata to {args.tag_meta_out}")
