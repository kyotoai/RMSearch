"""Rerank embedding candidates with a reward model, mirroring the notebook workflow."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Sequence

from rmsearch.utils.vllm_reward import build_llm, search

__all__ = ["rerank_candidates"]

logger = logging.getLogger(__name__)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_string_list(path: Path) -> List[str]:
    data = _load_json(path)
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = list(data.values())
    else:
        raise TypeError(f"Unsupported payload in {path} (expected list or dict).")
    strings: List[str] = []
    for item in values:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict) and "text" in item:
            strings.append(str(item["text"]))
        else:
            raise TypeError(f"Unsupported entry in {path}: {item!r}")
    if not strings:
        raise ValueError(f"No textual entries found in {path}")
    return strings


def _load_embed_records(path: Path) -> List[Dict[str, List[int]]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Embedding rankings in {path} must be a list.")
    records: List[Dict[str, List[int]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError(f"Embedding ranking entry must be an object, got {type(item)!r}")
        query_id = int(item.get("query_id", -1))
        key_ids = item.get("key_ids")
        if not isinstance(key_ids, list):
            raise TypeError(f"'key_ids' must be a list in {item!r}")
        records.append({"query_id": query_id, "key_ids": [int(k) for k in key_ids]})
    return records


def _llm_template(tokenizer):
    def format_row(row: Dict[str, str]) -> str:
        query = row["query"]
        key = row["key"]
        message = [
            {
                "role": "user",
                "content": (
                    "Give me relevance score between\n\n"
                    f"Query:{query}\n\n"
                    f"Sentence:{key}"
                ),
            }
        ]
        if len(message[0]["content"]) > 4000:
            message[0]["content"] = message[0]["content"][:4000] + "..."
        return tokenizer.apply_chat_template(message, tokenize=False)

    return format_row


def _parse_device_groups(spec: str | None, tensor_parallel_size: int, num_instances: int):
    if spec is None:
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
            raise ValueError(f"Each device group must contain {tensor_parallel_size} devices (got {group})")
    return groups


def rerank_candidates(
    queries: Sequence[str],
    keys: Sequence[str],
    embed_records: Sequence[Dict[str, List[int]]],
    *,
    model_name: str,
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    device_groups: List[List[int]] | None = None,
    max_model_len: int = 2500,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.90,
    request_batch_size: int = 128,
    timeout_s: float = 10_000.0,
    top_k: int | None = None,
) -> List[Dict[str, object]]:
    if not queries:
        raise ValueError("Query list is empty.")
    if not keys:
        raise ValueError("Key list is empty.")
    if not embed_records:
        raise ValueError("Embedding rankings list is empty.")

    logger.info("Launching reward model %s for reranking", model_name)
    rm = build_llm(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=device_groups,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        runner="pooling",
    )
    tokenizer = rm.tokenizer
    template_fn = _llm_template(tokenizer)

    requests: List[Dict[str, object]] = []
    id_maps: List[List[int]] = []
    request_query_ids: List[int] = []

    for record in embed_records:
        query_id = record["query_id"]
        if query_id < 0 or query_id >= len(queries):
            raise IndexError(f"query_id {query_id} is out of range for {len(queries)} queries.")
        candidate_ids = record["key_ids"]
        if not candidate_ids:
            continue
        if top_k is not None:
            candidate_ids = candidate_ids[:top_k]
        key_texts = []
        for kid in candidate_ids:
            if kid < 0 or kid >= len(keys):
                raise IndexError(f"key_id {kid} is out of range for {len(keys)} keys.")
            key_texts.append(keys[kid])
        requests.append({"query": queries[query_id], "keys": key_texts, "return_relevance": True})
        id_maps.append(candidate_ids)
        request_query_ids.append(query_id)

    if not requests:
        rm.close()
        raise ValueError("No requests were constructed from embedding rankings.")

    logger.info("Scoring %d requests with reward model", len(requests))
    try:
        outputs = search(
            rm,
            requests,
            template_fn,
            topk=max(len(ids) for ids in id_maps),
            query_batch_size=request_batch_size,
            timeout_s=timeout_s,
        )
    finally:
        rm.close()

    reranked: List[Dict[str, object]] = []
    for req_query_id, result, original_ids in zip(request_query_ids, outputs, id_maps):
        ordered_ids: List[int] = []
        scores: List[float] = []
        for item in result.get("keys", []):
            local_id = int(item["key_id"])
            if local_id < 0 or local_id >= len(original_ids):
                continue
            ordered_ids.append(original_ids[local_id])
            scores.append(float(item.get("relevance", 0.0)))
        reranked.append(
            {
                "query_id": req_query_id,
                "key_ids": ordered_ids,
                "relevance": scores,
            }
        )
    return reranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank embedding candidates with a reward model.")
    parser.add_argument("--query-json", type=Path, default=Path("query.json"), help="Query JSON produced by process_data.")
    parser.add_argument("--key-json", type=Path, default=Path("key.json"), help="Key JSON produced by process_data.")
    parser.add_argument(
        "--embed-json",
        type=Path,
        default=Path("relevance_dict_embed.json"),
        help="Embedding-based ranking JSON to rerank.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("relevance_dict_rerank.json"),
        help="Destination for reranked results.",
    )
    parser.add_argument("--model-name", type=str, required=True, help="Reward model identifier or checkpoint.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for reward model.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of reward model worker instances.")
    parser.add_argument(
        "--device-groups",
        type=str,
        help="Explicit GPU mapping, e.g. '0,1;2,3' for tensor_parallel_size=2 and num_instances=2.",
    )
    parser.add_argument("--max-model-len", type=int, default=2500, help="Maximum sequence length for the reward model.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory assigned to the reward model.",
    )
    parser.add_argument("--request-batch-size", type=int, default=128, help="Number of (query,key) pairs per scoring batch.")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout in seconds for reward scoring batches.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional limit on candidates per query before reranking.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    mp.set_start_method("spawn", force=True)

    queries = _load_string_list(args.query_json)
    keys = _load_string_list(args.key_json)
    embed_records = _load_embed_records(args.embed_json)
    device_groups = _parse_device_groups(args.device_groups, args.tensor_parallel_size, args.num_instances)

    reranked = rerank_candidates(
        queries,
        keys,
        embed_records,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        device_groups=device_groups,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        request_batch_size=args.request_batch_size,
        timeout_s=args.timeout,
        top_k=args.top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved reranked results for %d queries to %s", len(reranked), args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    main()
