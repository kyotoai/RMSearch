"""Select top-N relevant keys per query using the reward-model search stack."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from rmsearch.tree.assign_key import assign_key_to_tag_tree
from rmsearch.tree.search_key import build_search_backend, search_key as tree_search_key


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


def _rename_query_ids(tree: Sequence[Dict[str, Any]]) -> None:
    for node in tree:
        if "query_ids" in node:
            node["key_ids"] = [int(idx) for idx in node.pop("query_ids")]
        children = node.get("children")
        if isinstance(children, list):
            _rename_query_ids(children)


def _make_cached_runner(base_fn, cache_dir: Optional[Path], prefix: str):
    if cache_dir is None:
        return base_fn
    cache_dir.mkdir(parents=True, exist_ok=True)
    counter = count()

    def runner(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        label = next(counter)
        path = cache_dir / f"{prefix}-{label}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        results = base_fn(requests)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return results

    return runner


def _build_records(
    query_entries: Sequence[Dict[str, Any]],
    outputs: Sequence[Dict[str, Any]],
    *,
    keys: Sequence[str],
    correct_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for query_id, result in enumerate(outputs):
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
        for key_record in result.get("keys", []):
            raw_idx = key_record.get("relevant_id", key_record.get("key_id", -1))
            try:
                key_idx = int(raw_idx)
            except (TypeError, ValueError):
                key_idx = -1
            item = {
                "key_id": key_idx,
                "key": keys[key_idx] if 0 <= key_idx < len(keys) else key_record.get("key", ""),
            }
            if "relevance" in key_record:
                item["relevance"] = key_record["relevance"]
            entry["keys"].append(item)
        records.append(entry)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-N keys per query using the reward-model guided search."
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
    parser.add_argument("--tag-tree", type=Path, required=True, help="Tag tree JSON used during training.")
    parser.add_argument("--tag2key-out", type=Path, help="Optional path to store the generated tag2key JSON.")
    parser.add_argument("--correct-ids-json", type=str, help="Optional JSON list/dict with ground-truth key ids per query.")
    parser.add_argument("--output", type=Path, default=Path("relevance_records_rm.json"), help="Where to write the relevance records JSON.")
    parser.add_argument("--checkpoint", type=Path, help="Directory for caching intermediate reward-model search outputs.")
    parser.add_argument("--k-tag", type=int, default=2, help="Top branches explored per depth.")
    parser.add_argument("--k-key", type=int, default=10, help="Number of keys returned per query.")
    parser.add_argument("--model-name", type=str, required=True, help="Reward model checkpoint or identifier.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for the reward model.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of reward model worker instances.")
    parser.add_argument("--max-model-len", type=int, default=4000, help="Maximum token length per request.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to allocate to the reward model workers.",
    )
    return parser.parse_args()


def main() -> None:
    logging.getLogger("vllm").setLevel(logging.ERROR)
    args = parse_args()

    query_entries = _load_queries(args)
    queries = [entry["query"] for entry in query_entries]
    keys = _load_keys(args)
    tag_tree = _read_json(args.tag_tree)
    correct_ids = _load_correct_ids(args.correct_ids_json, len(queries)) if args.correct_ids_json else None

    rm = None
    try:
        rm, run_search = build_search_backend(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            num_instances=args.num_instances,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        assign_cache = args.checkpoint / "assign_key" if args.checkpoint else None
        assign_search_fn = _make_cached_runner(run_search, assign_cache, "assign")
        key_records = [{"query": key} for key in keys]
        _, tag2key = assign_key_to_tag_tree(key_records, tag_tree, search_fn=assign_search_fn, k_tag=args.k_tag)
        _rename_query_ids(tag2key)

        if args.tag2key_out:
            args.tag2key_out.parent.mkdir(parents=True, exist_ok=True)
            args.tag2key_out.write_text(json.dumps(tag2key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        outputs = tree_search_key(
            queries,
            keys,
            tag2key,
            search_fn=run_search,
            k_tag=args.k_tag,
            k_key=args.k_key,
            checkpoint=args.checkpoint,
        )
    finally:
        if rm is not None:
            rm.close()

    records = _build_records(query_entries, outputs, keys=keys, correct_ids=correct_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved relevance records to {args.output}")


if __name__ == "__main__":
    main()
