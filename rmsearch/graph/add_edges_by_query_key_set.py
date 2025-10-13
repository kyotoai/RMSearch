"""Augment tag graph key assignments by scoring queries against keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from rmsearch.graph._graph_utils import flatten_tree, iter_nodes, load_tag_graph
from rmsearch.tree.assign_key import assign_key_to_tag_tree
from rmsearch.utils.vllm_reward import build_llm, search


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


def _load_keys(path: Path, column: str) -> List[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {path}")
        series = df[column].dropna()
        if series.empty:
            raise ValueError(f"Column '{column}' in {path} is empty")
        return [str(value) for value in series.tolist()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            values = list(data.values())
        else:
            values = data
        keys: List[str] = []
        for item in values:
            if isinstance(item, str):
                keys.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    keys.append(str(item["text"]))
                elif "key" in item:
                    keys.append(str(item["key"]))
        if not keys:
            raise ValueError(f"No textual entries found in {path}")
        return keys
    raise ValueError(f"Unsupported keys file format: {path}")


def _load_queries(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = list(data.values())
    else:
        values = data
    queries: List[str] = []
    for item in values:
        if isinstance(item, str):
            queries.append(item)
        elif isinstance(item, dict) and "query" in item:
            queries.append(str(item["query"]))
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def _build_tag_search_fn(args: argparse.Namespace, tokenizer, rm):
    def llm_template(row: Dict[str, Any]) -> str:
        message = [
            {
                "role": "user",
                "content": (
                    "Choose the best tag for the following sentence.\n\n"
                    f"Sentence: '''{row['query']}'''"
                ),
            },
            {"role": "assistant", "content": str(row["key"])},
        ]
        prompt = tokenizer.apply_chat_template(message, tokenize=False)
        return prompt

    def run_search(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not requests:
            return []
        topk = max((req.get("k", args.k_tag) for req in requests), default=args.k_tag)
        return search(
            rm,
            requests,
            llm_template,
            topk=topk,
            query_batch_size=args.batch_size,
            batch_size=args.batch_size,
            timeout_s=args.timeout,
        )

    return run_search


def _score_keys(
    args: argparse.Namespace,
    rm,
    tokenizer,
    queries: Sequence[str],
    keys: Sequence[str],
) -> List[List[int]]:
    def llm_template(row: Dict[str, Any]) -> str:
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
        prompt = tokenizer.apply_chat_template(message, tokenize=False)
        return prompt

    requests = [{"query": text, "keys": list(keys), "k": args.k_key} for text in queries]
    scored = search(
        rm,
        requests,
        llm_template,
        topk=args.k_key,
        query_batch_size=args.batch_size,
        batch_size=args.batch_size,
        timeout_s=args.timeout,
    )
    top_indices: List[List[int]] = []
    for result in scored:
        indices: List[int] = []
        for item in result.get("keys", []):
            local_idx = int(item.get("key_id", 0))
            if 0 <= local_idx < len(keys):
                indices.append(local_idx)
        top_indices.append(indices[: args.k_key])
    return top_indices


def _nodes_along_path(tree: Sequence[Dict[str, Any]], path: Sequence[int]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    node_list: Sequence[Dict[str, Any]] = tree
    for idx in path:
        if idx < 0 or idx >= len(node_list):
            return []
        node = node_list[idx]
        nodes.append(node)
        node_list = node.get("children", [])
    return nodes


def _initialise_key_sets(tree: Sequence[Dict[str, Any]]) -> None:
    for node in iter_nodes(tree):
        node["key_ids"] = {int(k) for k in node.get("key_ids", []) if k is not None}


def _finalise_key_sets(tree: Sequence[Dict[str, Any]]) -> None:
    for node in iter_nodes(tree):
        node["key_ids"] = sorted(int(k) for k in node.get("key_ids", []))


def augment_graph(args: argparse.Namespace) -> None:
    keys = _load_keys(args.keys_file, args.key_column)
    queries = _load_queries(args.queries)
    tag_tree = load_tag_graph(args.tag2key)
    _initialise_key_sets(tag_tree)

    device_groups = _parse_device_groups(
        args.device_groups,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
    )

    rm = build_llm(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size if device_groups is None else len(device_groups[0]),
        num_instances=args.num_instances if device_groups is None else len(device_groups),
        device_groups=device_groups,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        runner="pooling",
    )

    try:
        tag_search_fn = _build_tag_search_fn(args, rm.tokenizer, rm)
        query_records = [{"query": text} for text in queries]
        query_assignments, _ = assign_key_to_tag_tree(
            query_records,
            tag_tree,
            search_fn=tag_search_fn,
            k_tag=args.k_tag,
        )

        top_key_indices = _score_keys(args, rm, rm.tokenizer, queries, keys)
    finally:
        rm.close()

    for assignment, key_indices in zip(query_assignments, top_key_indices):
        if not key_indices:
            continue
        for path in assignment.get("tag_ids", []):
            for node in _nodes_along_path(tag_tree, path):
                key_set: set[int] = node.setdefault("key_ids", set())  # type: ignore[assignment]
                key_set.update(int(idx) for idx in key_indices if 0 <= idx < len(keys))

    _finalise_key_sets(tag_tree)
    records = flatten_tree(tag_tree)
    df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved augmented tag graph to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand tag graph key assignments using query-key reward scoring.")
    parser.add_argument("--keys-file", type=Path, required=True, help="CSV or JSON file containing key texts.")
    parser.add_argument("--key-column", type=str, default="text", help="Column name when reading keys from CSV.")
    parser.add_argument("--queries", type=Path, required=True, help="JSON file containing queries (strings or objects with 'query').")
    parser.add_argument("--tag2key", type=Path, required=True, help="Input tag2key parquet file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination parquet file for the updated graph.")
    parser.add_argument("--model-name", type=str, required=True, help="Reward model name or path.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor_parallel_size per reward model instance.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of reward model worker instances.")
    parser.add_argument("--device-groups", type=str, help="Explicit GPU mapping, e.g. '0,1;2,3' for two workers.")
    parser.add_argument("--batch-size", type=int, default=128, help="Prompts processed per worker batch.")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout (s) for reward model batches.")
    parser.add_argument("--k-tag", type=int, default=2, help="Number of tag branches explored per depth.")
    parser.add_argument("--k-key", type=int, default=5, help="Number of keys associated with each query.")
    parser.add_argument("--max-model-len", type=int, default=2500, help="Maximum model context length.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilisation passed to vLLM.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    augment_graph(args)


if __name__ == "__main__":
    main()
