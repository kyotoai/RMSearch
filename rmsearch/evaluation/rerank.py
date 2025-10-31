"""Rerank embedding candidates with a reward model, mirroring the notebook workflow."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from rmsearch.utils.vllm_reward import build_llm, search

__all__ = ["rerank_candidates"]

logger = logging.getLogger(__name__)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_strings_from_json(path: Path, *, field: str | None = None) -> List[str]:
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


def _load_strings_from_csv(path: Path, *, text_column: str, id_column: str | None) -> Dict[int, str]:
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not present in {path}")
    df = df[df[text_column].notna()].copy()
    if df.empty:
        raise ValueError(f"Column '{text_column}' in {path} is empty")
    if id_column and id_column in df.columns:
        df[id_column] = df[id_column].astype(int)
        df = df.sort_values(id_column)
        mapping = {int(row[id_column]): str(row[text_column]) for _, row in df.iterrows()}
    else:
        mapping = {idx: str(val) for idx, val in enumerate(df[text_column].astype(str).tolist())}
    return mapping


def _load_queries(
    query_json: Path | None,
    query_csv: Path | None,
    *,
    text_column: str,
    id_column: str,
) -> Dict[int, str]:
    if query_csv and query_csv.is_file():
        logger.info("Loading queries from CSV: %s", query_csv)
        return _load_strings_from_csv(query_csv, text_column=text_column, id_column=id_column)
    if query_json and query_json.is_file():
        logger.info("Loading queries from JSON: %s", query_json)
        strings = _load_strings_from_json(query_json, field="query")
        return {idx: text for idx, text in enumerate(strings)}
    raise FileNotFoundError("Provide --query-csv or --query-json (no query file found).")


def _load_keys(
    key_json: Path | None,
    key_csv: Path | None,
    *,
    text_column: str,
    id_column: str,
) -> Dict[int, str]:
    if key_csv and key_csv.is_file():
        logger.info("Loading keys from CSV: %s", key_csv)
        return _load_strings_from_csv(key_csv, text_column=text_column, id_column=id_column)
    if key_json and key_json.is_file():
        logger.info("Loading keys from JSON: %s", key_json)
        strings = _load_strings_from_json(key_json, field="text")
        return {idx: text for idx, text in enumerate(strings)}
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


def _load_embed_records(path: Path) -> List[Dict[str, object]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Embedding rankings in {path} must be a list.")
    records: List[Dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError(f"Embedding ranking entry must be an object, got {type(item)!r}")
        query_id = int(item.get("query_id", -1))
        key_ids = item.get("key_ids")
        if not isinstance(key_ids, list):
            raise TypeError(f"'key_ids' must be a list in {item!r}")
        record: Dict[str, object] = {"query_id": query_id, "key_ids": [int(k) for k in key_ids]}
        if "positive_key_ids" in item and isinstance(item["positive_key_ids"], list):
            record["positive_key_ids"] = [int(k) for k in item["positive_key_ids"]]
        records.append(record)
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
    query_text_by_id: Mapping[int, str],
    key_text_by_id: Mapping[int, str],
    embed_records: Sequence[Dict[str, object]],
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
    positive_pairs: Mapping[int, List[int]] | None = None,
) -> List[Dict[str, object]]:
    if not query_text_by_id:
        raise ValueError("Query mapping is empty.")
    if not key_text_by_id:
        raise ValueError("Key mapping is empty.")
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
        query_id = int(record["query_id"])
        query_text = query_text_by_id.get(query_id)
        if query_text is None:
            raise KeyError(f"Query id {query_id} missing from query mapping.")
        candidate_ids = [int(k) for k in record["key_ids"]]  # type: ignore[index]
        if not candidate_ids:
            continue
        key_texts: List[str] = []
        for kid in candidate_ids:
            key_text = key_text_by_id.get(kid)
            if key_text is None:
                raise KeyError(f"Key id {kid} missing from key mapping.")
            key_texts.append(key_text)
        requests.append({"query": query_text, "keys": key_texts, "return_relevance": True})
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
            batch_size=request_batch_size,
            timeout_s=timeout_s,
        )
    finally:
        rm.close()

    positive_lookup = positive_pairs or {}
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
        pre_key_ids = list(original_ids)
        limit = top_k if top_k is not None else len(ordered_ids)
        limit = min(limit, len(ordered_ids))
        entry: Dict[str, object] = {
            "query_id": req_query_id,
            "pre_key_ids": pre_key_ids,
            "key_ids": ordered_ids[:limit],
            "relevance": scores[:limit],
            "positive_key_ids": positive_lookup.get(req_query_id, []),
        }
        reranked.append(entry)
    return reranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank embedding candidates with a reward model.")
    parser.add_argument("--query-csv", type=Path, default=Path("query.csv"), help="CSV file containing queries.")
    parser.add_argument("--query-json", type=Path, default=Path("query.json"), help="Fallback JSON file containing queries.")
    parser.add_argument("--query-text-column", type=str, default="text", help="Column holding query text in --query-csv.")
    parser.add_argument("--query-id-column", type=str, default="id", help="Column holding query ids in --query-csv.")
    parser.add_argument("--key-csv", type=Path, default=Path("key.csv"), help="CSV file containing keys.")
    parser.add_argument("--key-json", type=Path, default=Path("key.json"), help="Fallback JSON file containing keys.")
    parser.add_argument("--key-text-column", type=str, default="text", help="Column holding key text in --key-csv.")
    parser.add_argument("--key-id-column", type=str, default="id", help="Column holding key ids in --key-csv.")
    parser.add_argument("--pair-csv", type=Path, help="Optional CSV of positive pairs (query_id,key_id).")
    parser.add_argument("--pair-query-column", type=str, default="query_id", help="Query id column inside --pair-csv.")
    parser.add_argument("--pair-key-column", type=str, default="key_id", help="Key id column inside --pair-csv.")
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
    parser.add_argument("--top-k", type=int, default=10, help="Number of most relevant keys to keep per query in the output.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    mp.set_start_method("spawn", force=True)

    query_text_by_id = _load_queries(
        args.query_json,
        args.query_csv,
        text_column=args.query_text_column,
        id_column=args.query_id_column,
    )
    key_text_by_id = _load_keys(
        args.key_json,
        args.key_csv,
        text_column=args.key_text_column,
        id_column=args.key_id_column,
    )
    embed_records = _load_embed_records(args.embed_json)

    positive_pairs: Dict[int, List[int]] = _load_positive_pairs(
        args.pair_csv,
        query_column=args.pair_query_column,
        key_column=args.pair_key_column,
    ) if args.pair_csv else {}

    for record in embed_records:
        if "positive_key_ids" in record and isinstance(record["positive_key_ids"], list):
            positive_pairs.setdefault(int(record["query_id"]), list(record["positive_key_ids"]))  # type: ignore[arg-type]

    device_groups = _parse_device_groups(args.device_groups, args.tensor_parallel_size, args.num_instances)

    reranked = rerank_candidates(
        query_text_by_id,
        key_text_by_id,
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
        positive_pairs=positive_pairs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved reranked results for %d queries to %s", len(reranked), args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    main()
