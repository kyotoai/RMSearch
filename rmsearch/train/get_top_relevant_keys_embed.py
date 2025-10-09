"""Select top-N relevant keys per query using vLLM embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch

from rmsearch.utils import vllm_embed


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_query_entries_from_json(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    entries: List[Dict[str, Any]] = []

    def _coerce_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
        if "query" not in obj:
            raise ValueError(f"JSON object in {path} is missing required 'query' field: {obj}")
        entry = dict(obj)
        entry["query"] = str(entry["query"])
        return entry

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                entries.append(_coerce_entry(item))
            elif isinstance(item, str):
                entries.append({"query": item})
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, dict):
                entries.append(_coerce_entry(value))
            elif isinstance(value, str):
                entries.append({"query": value})
    else:
        raise TypeError(f"Unsupported JSON payload in {path}: {type(data)}")

    if not entries:
        raise ValueError(f"No query entries found in {path}")
    return entries


def _load_strings_from_json(path: Path, *, value_key: str) -> List[str]:
    data = _read_json(path)
    if isinstance(data, dict):
        values: Iterable[Any] = data.values()
    elif isinstance(data, list):
        values = data
    else:
        raise TypeError(f"Unsupported JSON payload in {path}: {type(data)}")

    items: List[str] = []
    for value in values:
        if isinstance(value, str):
            items.append(value)
        elif isinstance(value, dict) and value_key in value:
            items.append(str(value[value_key]))
    if not items:
        raise ValueError(f"No '{value_key}' entries found in {path}")
    return items


def _load_strings_from_csv(path: Path, *, column: str) -> List[str]:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not present in {path}")
    series = df[column].dropna()
    values = series.astype(str).tolist()
    if not values:
        raise ValueError(f"Column '{column}' in {path} is empty")
    return values


def _load_queries(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.queries_json:
        return _load_query_entries_from_json(Path(args.queries_json))
    if args.queries_csv:
        return [{"query": text} for text in _load_strings_from_csv(Path(args.queries_csv), column=args.query_column)]
    raise ValueError("Provide --queries-json or --queries-csv")


def _load_keys(args: argparse.Namespace) -> List[str]:
    if args.keys_json:
        return _load_strings_from_json(Path(args.keys_json), value_key=args.key_json_field)
    if args.keys_csv:
        return _load_strings_from_csv(Path(args.keys_csv), column=args.key_column)
    raise ValueError("Provide --keys-json or --keys-csv")


def _load_correct_ids(path: Optional[str], expected: int) -> Optional[List[int]]:
    if not path:
        return None
    data = _read_json(Path(path))
    if isinstance(data, dict):
        values: Iterable[Any] = data.values()
    elif isinstance(data, list):
        values = data
    else:
        raise TypeError(f"Unsupported JSON payload in {path}: {type(data)}")
    numbers = [int(item) for item in values]
    if len(numbers) != expected:
        raise ValueError(f"Expected {expected} correct ids, got {len(numbers)} in {path}")
    return numbers


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=True)


def _build_records(
    query_entries: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    indices: torch.Tensor,
    scores: torch.Tensor,
    *,
    correct_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    indices_cpu = indices.cpu().tolist()
    scores_cpu = scores.cpu().tolist()
    records: List[Dict[str, Any]] = []
    for query_id, (key_ids, score_row) in enumerate(zip(indices_cpu, scores_cpu)):
        query_meta = query_entries[query_id]
        entry: Dict[str, Any] = {
            "query": query_meta["query"],
            "query_id": query_id,
            "keys": [],
        }
        if "df_id" in query_meta:
            entry["df_id"] = query_meta["df_id"]
        query_type = query_meta.get("query-type") or query_meta.get("query_type")
        if query_type:
            entry["query_type"] = query_type
        if correct_ids is not None:
            entry["correct_id"] = int(correct_ids[query_id])
        for kid, score in zip(key_ids, score_row):
            kid_int = int(kid)
            item = {
                "key_id": kid_int,
                "key": keys[kid_int] if 0 <= kid_int < len(keys) else "",
                "similarity": float(score),
            }
            entry["keys"].append(item)
        records.append(entry)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-N keys per query using vLLM embeddings."
    )
    parser.add_argument(
        "--queries-json",
        type=str,
        help="JSON file with query objects (e.g. filtered_query_recs.json) containing at least a 'query' field.",
    )
    parser.add_argument("--queries-csv", type=str, help="CSV file with query strings.")
    parser.add_argument("--query-column", type=str, default="query", help="CSV column containing query text.")
    parser.add_argument("--keys-json", type=str, help="JSON file with key strings or objects.")
    parser.add_argument("--key-json-field", type=str, default="text", help="Field to read from JSON objects in --keys-json.")
    parser.add_argument("--keys-csv", type=str, help="CSV file with key strings.")
    parser.add_argument("--key-column", type=str, default="text", help="CSV column containing key text.")
    parser.add_argument("--correct-ids-json", type=str, help="Optional JSON list/dict with ground-truth key ids per query.")
    parser.add_argument("--output", type=Path, default=Path("relevance_records_embed.json"), help="Where to write the relevance records JSON.")
    parser.add_argument("--model-name", type=str, required=True, help="Embedding model checkpoint or identifier.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for the embedding model.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of embedding model worker instances.")
    parser.add_argument("--max-model-len", type=int, default=4000, help="Maximum token length per request.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to allocate to the embedding workers.",
    )
    parser.add_argument("--query-batch-size", type=int, default=512, help="Batch size for query embedding calls.")
    parser.add_argument("--key-batch-size", type=int, default=128, help="Batch size for key embedding calls.")
    parser.add_argument("--query-checkpoint", type=Path, help="Optional JSONL checkpoint file for query embeddings.")
    parser.add_argument("--key-checkpoint", type=Path, help="Optional JSONL checkpoint file for key embeddings.")
    parser.add_argument("--similarity-device", type=str, default="cpu", help="Device used to compute similarity scores (e.g. 'cpu' or 'cuda').")
    parser.add_argument("--k-key", type=int, default=50, help="Number of keys returned per query.")
    return parser.parse_args()


def main() -> None:
    logging.getLogger("vllm").setLevel(logging.ERROR)
    args = parse_args()

    query_entries = _load_queries(args)
    queries = [entry["query"] for entry in query_entries]
    keys = _load_keys(args)
    correct_ids = _load_correct_ids(args.correct_ids_json, len(queries)) if args.correct_ids_json else None

    if not keys:
        raise ValueError("Key list is empty; provide non-empty keys input.")
    if not queries:
        raise ValueError("Query list is empty; provide non-empty queries input.")

    embedder = None
    try:
        embedder = vllm_embed.build_embedding_model(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            num_instances=args.num_instances,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            output_to_cpu=True,
        )

        query_vectors = vllm_embed.embed(
            embedder,
            queries,
            batch_size=args.query_batch_size,
            checkpoint_path=str(args.query_checkpoint) if args.query_checkpoint else None,
        )
        key_vectors = vllm_embed.embed(
            embedder,
            keys,
            batch_size=args.key_batch_size,
            checkpoint_path=str(args.key_checkpoint) if args.key_checkpoint else None,
        )
    finally:
        if embedder is not None:
            embedder.close()

    query_tensor = torch.tensor(query_vectors, dtype=torch.float32)
    key_tensor = torch.tensor(key_vectors, dtype=torch.float32)

    if query_tensor.ndim != 2 or key_tensor.ndim != 2:
        raise ValueError("Embeddings must be rank-2 tensors.")
    if query_tensor.shape[1] != key_tensor.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ: queries {query_tensor.shape[1]} vs keys {key_tensor.shape[1]}"
        )

    device_choice = (args.similarity_device or "cpu").strip().lower()
    if device_choice == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.similarity_device
    device = torch.device(resolved_device)
    query_tensor = _to_device(query_tensor, device)
    key_tensor = _to_device(key_tensor, device)

    relevance = query_tensor @ key_tensor.T

    topk = min(args.k_key, key_tensor.shape[0])
    if topk <= 0:
        raise ValueError("k-key must be positive and less than or equal to number of keys.")
    scores, indices = torch.topk(relevance, k=topk, dim=1)
    scores = scores.detach()
    indices = indices.detach()
    del relevance, query_tensor, key_tensor

    records = _build_records(query_entries, keys, indices, scores, correct_ids=correct_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved relevance records to {args.output}")


if __name__ == "__main__":
    main()
