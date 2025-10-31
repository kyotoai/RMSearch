"""Generate embedding-based relevance rankings for evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from rmsearch.utils import vllm_embed

__all__ = ["build_relevance_dict"]

logger = logging.getLogger(__name__)


def _load_strings_from_json(path: Path, *, field: str | None = None) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = list(data.values())
    else:
        raise TypeError(f"Unsupported payload in {path}: expected list or dict, got {type(data)!r}")

    strings: List[str] = []
    for item in values:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            if field and field in item:
                strings.append(str(item[field]))
            elif "text" in item:
                strings.append(str(item["text"]))
            elif "query" in item:
                strings.append(str(item["query"]))
        else:
            raise TypeError(f"Unsupported entry in {path}: {item!r}")

    if not strings:
        raise ValueError(f"No textual entries found in {path}")
    return strings


def _load_strings_from_csv(path: Path, *, text_column: str, id_column: str | None) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not present in {path}")
    df = df[df[text_column].notna()].copy()
    if df.empty:
        raise ValueError(f"Column '{text_column}' in {path} is empty")

    if id_column and id_column in df.columns:
        df[id_column] = df[id_column].astype(int)
        df = df.sort_values(id_column)
        ids = df[id_column].astype(int).tolist()
    else:
        ids = list(range(len(df)))
    texts = df[text_column].astype(str).tolist()
    return texts, ids


def _load_queries(
    query_json: Path | None,
    query_csv: Path | None,
    *,
    text_column: str,
    id_column: str,
) -> Tuple[List[str], List[int]]:
    if query_csv and query_csv.is_file():
        logger.info("Loading queries from CSV: %s", query_csv)
        return _load_strings_from_csv(query_csv, text_column=text_column, id_column=id_column)
    if query_json and query_json.is_file():
        logger.info("Loading queries from JSON: %s", query_json)
        strings = _load_strings_from_json(query_json, field="query")
        return strings, list(range(len(strings)))
    raise FileNotFoundError("Provide --query-csv or --query-json (no query file found).")


def _load_keys(
    key_json: Path | None,
    key_csv: Path | None,
    *,
    text_column: str,
    id_column: str,
) -> Tuple[List[str], List[int]]:
    if key_csv and key_csv.is_file():
        logger.info("Loading keys from CSV: %s", key_csv)
        return _load_strings_from_csv(key_csv, text_column=text_column, id_column=id_column)
    if key_json and key_json.is_file():
        logger.info("Loading keys from JSON: %s", key_json)
        strings = _load_strings_from_json(key_json, field="text")
        return strings, list(range(len(strings)))
    raise FileNotFoundError("Provide --key-csv or --key-json (no key file found).")


def _load_positive_pairs(
    pair_csv: Path | None,
    *,
    query_column: str,
    key_column: str,
) -> Dict[int, List[int]]:
    if pair_csv is None:
        return {}
    if not pair_csv.is_file():
        raise FileNotFoundError(f"Pair CSV not found: {pair_csv}")
    df = pd.read_csv(pair_csv)
    for column in (query_column, key_column):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not present in {pair_csv}")
    df = df[[query_column, key_column]].dropna()
    df[query_column] = df[query_column].astype(int)
    df[key_column] = df[key_column].astype(int)
    grouped = df.groupby(query_column)[key_column].apply(list)
    return {int(qid): [int(k) for k in keys] for qid, keys in grouped.items()}


def _embed_texts(
    embedder: vllm_embed.EmbeddingWorkerModel,
    texts: Sequence[str],
    *,
    batch_size: int,
    checkpoint: Path | None = None,
) -> torch.Tensor:
    vectors = vllm_embed.embed(
        embedder,
        list(texts),
        batch_size=batch_size,
        checkpoint_path=str(checkpoint) if checkpoint else None,
    )
    if not vectors:
        raise ValueError("Embedding backend returned no vectors.")
    tensor = torch.tensor(vectors, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(tensor.shape)}")
    return tensor


def _maybe_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, p=2.0, dim=-1)


def build_relevance_dict(
    queries: Sequence[str],
    keys: Sequence[str],
    *,
    model_name: str,
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    max_model_len: int = 4000,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.90,
    query_batch_size: int = 256,
    key_batch_size: int = 128,
    top_k: int = 100,
    normalize: bool = True,
    similarity_device: str = "cpu",
    query_checkpoint: Path | None = None,
    key_checkpoint: Path | None = None,
) -> List[List[int]]:
    if not queries:
        raise ValueError("Query list is empty.")
    if not keys:
        raise ValueError("Key list is empty.")
    logger.info("Embedding %d queries and %d keys with model %s", len(queries), len(keys), model_name)

    embedder = vllm_embed.build_embedding_model(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        output_to_cpu=True,
    )
    try:
        query_tensor = _embed_texts(embedder, queries, batch_size=query_batch_size, checkpoint=query_checkpoint)
        key_tensor = _embed_texts(embedder, keys, batch_size=key_batch_size, checkpoint=key_checkpoint)
    finally:
        embedder.close()

    if query_tensor.shape[1] != key_tensor.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: queries {query_tensor.shape[1]} vs keys {key_tensor.shape[1]}"
        )

    if normalize:
        query_tensor = _maybe_normalize(query_tensor)
        key_tensor = _maybe_normalize(key_tensor)

    device_choice = (similarity_device or "cpu").strip().lower()
    if device_choice == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = similarity_device
    device = torch.device(resolved_device)
    query_tensor = query_tensor.to(device)
    key_tensor = key_tensor.to(device)

    logger.info("Computing similarity matrix on %s", device)
    relevance = query_tensor @ key_tensor.T

    k = min(top_k, key_tensor.shape[0])
    if k <= 0:
        raise ValueError("top_k must be positive.")
    scores, indices = torch.topk(relevance, k=k, dim=1)
    del relevance, scores

    return indices.cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embedding-based candidate rankings for evaluation.")
    parser.add_argument("--query-csv", type=Path, default=Path("query.csv"), help="CSV file containing queries (from beir_to_pairs).")
    parser.add_argument("--query-json", type=Path, default=Path("query.json"), help="Fallback JSON file containing queries.")
    parser.add_argument("--query-text-column", type=str, default="text", help="Column holding query text in --query-csv.")
    parser.add_argument("--query-id-column", type=str, default="id", help="Column holding query ids in --query-csv.")
    parser.add_argument("--key-csv", type=Path, default=Path("key.csv"), help="CSV file containing keys (from beir_to_pairs).")
    parser.add_argument("--key-json", type=Path, default=Path("key.json"), help="Fallback JSON file containing keys.")
    parser.add_argument("--key-text-column", type=str, default="text", help="Column holding key text in --key-csv.")
    parser.add_argument("--key-id-column", type=str, default="id", help="Column holding key ids in --key-csv.")
    parser.add_argument("--pair-csv", type=Path, help="Optional CSV (query_id,key_id) providing positive pairs.")
    parser.add_argument("--pair-query-column", type=str, default="query_id", help="Query id column inside --pair-csv.")
    parser.add_argument("--pair-key-column", type=str, default="key_id", help="Key id column inside --pair-csv.")
    parser.add_argument("--output", type=Path, default=Path("relevance_dict_embed.json"), help="Destination for rankings.")
    parser.add_argument("--model-name", type=str, required=True, help="Embedding model identifier or checkpoint.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for embedding workers.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of embedding worker instances.")
    parser.add_argument("--max-model-len", type=int, default=4000, help="Maximum sequence length for the model.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory allocated to embedding workers.",
    )
    parser.add_argument("--query-batch-size", type=int, default=256, help="Batch size for query embedding calls.")
    parser.add_argument("--key-batch-size", type=int, default=128, help="Batch size for key embedding calls.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of keys to keep per query.")
    parser.add_argument("--no-normalize", action="store_true", help="Skip L2 normalisation before scoring.")
    parser.add_argument("--similarity-device", type=str, default="cpu", help="Device used for similarity (e.g. cpu or cuda).")
    parser.add_argument("--query-checkpoint", type=Path, help="Optional query embedding checkpoint (JSONL).")
    parser.add_argument("--key-checkpoint", type=Path, help="Optional key embedding checkpoint (JSONL).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    queries, query_ids = _load_queries(
        args.query_json,
        args.query_csv,
        text_column=args.query_text_column,
        id_column=args.query_id_column,
    )
    keys, key_ids = _load_keys(
        args.key_json,
        args.key_csv,
        text_column=args.key_text_column,
        id_column=args.key_id_column,
    )
    positive_pairs = _load_positive_pairs(
        args.pair_csv,
        query_column=args.pair_query_column,
        key_column=args.pair_key_column,
    ) if args.pair_csv else {}

    indices = build_relevance_dict(
        queries,
        keys,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        query_batch_size=args.query_batch_size,
        key_batch_size=args.key_batch_size,
        top_k=args.top_k,
        normalize=not args.no_normalize,
        similarity_device=args.similarity_device,
        query_checkpoint=args.query_checkpoint,
        key_checkpoint=args.key_checkpoint,
    )

    if len(indices) != len(query_ids):
        raise RuntimeError("Mismatch between embedded queries and loaded query ids.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for idx, key_idx_list in enumerate(indices):
        query_id = int(query_ids[idx])
        resolved_key_ids = [int(key_ids[key_idx]) for key_idx in key_idx_list]
        entry: Dict[str, object] = {
            "query_id": query_id,
            "key_ids": resolved_key_ids,
        }
        if positive_pairs:
            entry["positive_key_ids"] = positive_pairs.get(query_id, [])
        payload.append(entry)

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved embedding rankings for %d queries to %s", len(payload), args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    main()
