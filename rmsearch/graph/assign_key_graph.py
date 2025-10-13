"""Assign keys to the tag graph and persist the augmented parquet artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from rmsearch.graph._graph_utils import (
    flatten_tree,
    index_path_to_tag_ids,
    load_tag_graph,
)
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


def _load_keys_from_json(path: Path) -> List[str]:
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


def _load_keys(args: argparse.Namespace) -> List[str]:
    if args.keys_json and args.keys_json.exists():
        return _load_keys_from_json(args.keys_json)
    if args.keys_csv and args.keys_csv.exists():
        df = pd.read_csv(args.keys_csv)
        if args.key_column not in df.columns:
            raise ValueError(f"Column '{args.key_column}' not found in {args.keys_csv}")
        series = df[args.key_column].dropna()
        if series.empty:
            raise ValueError(f"Column '{args.key_column}' in {args.keys_csv} is empty")
        return [str(value) for value in series.tolist()]
    raise ValueError("Provide --keys-json or --keys-csv")


def _build_tag_search_fn(args: argparse.Namespace, tokenizer, rm):
    def llm_template(row: Dict[str, Any]) -> str:
        message = [
            {
                "role": "user",
                "content": (
                    "Generate the most appropriate tag for the following sentence.\n\n"
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


def _build_key2tag_records(tree: Sequence[Dict[str, Any]], assignments: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for key_id, entry in enumerate(assignments):
        mapped_paths: List[List[int]] = []
        for path in entry.get("tag_ids", []):
            converted = index_path_to_tag_ids(tree, path)
            if converted:
                mapped_paths.append(converted)
        records.append({"key_id": key_id, "tag_ids": mapped_paths})
    return pd.DataFrame.from_records(records)


def assign_keys(args: argparse.Namespace) -> None:
    keys = _load_keys(args)
    tag_tree = load_tag_graph(args.tag_graph)

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
        search_fn = _build_tag_search_fn(args, rm.tokenizer, rm)
        key_records = [{"query": text} for text in keys]
        key2tag_assignments, tag2key_tree = assign_key_to_tag_tree(
            key_records,
            tag_tree,
            search_fn=search_fn,
            k_tag=args.k_tag,
        )
    finally:
        rm.close()

    key2tag_df = _build_key2tag_records(tag2key_tree, key2tag_assignments)
    tag2key_df = pd.DataFrame(flatten_tree(tag2key_tree))

    args.key2tag_out.parent.mkdir(parents=True, exist_ok=True)
    args.tag2key_out.parent.mkdir(parents=True, exist_ok=True)
    key2tag_df.to_parquet(args.key2tag_out, index=False)
    tag2key_df.to_parquet(args.tag2key_out, index=False)

    print(f"Saved key→tag assignments to {args.key2tag_out}")
    print(f"Saved augmented tag graph to {args.tag2key_out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign free-form keys to the tag graph using a reward model.")
    parser.add_argument("--tag-graph", type=Path, required=True, help="Input tag graph parquet file.")
    parser.add_argument("--keys-json", type=Path, help="JSON file containing key strings or objects with 'text'/'key'.")
    parser.add_argument("--keys-csv", type=Path, help="CSV file containing key strings.")
    parser.add_argument("--key-column", type=str, default="text", help="Column name to read keys from when using CSV input.")
    parser.add_argument("--key2tag-out", type=Path, required=True, help="Destination parquet file for key-to-tag paths.")
    parser.add_argument("--tag2key-out", type=Path, required=True, help="Destination parquet file for augmented tag graph.")
    parser.add_argument("--model-name", type=str, required=True, help="Reward model name or path for scoring.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor_parallel_size per reward model instance.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of reward model worker instances.")
    parser.add_argument("--device-groups", type=str, help="Explicit GPU mapping, e.g. '0,1;2,3' for two workers with tensor_parallel_size=2.")
    parser.add_argument("--batch-size", type=int, default=128, help="Prompts processed per worker batch.")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout (s) for reward model batches.")
    parser.add_argument("--k-tag", type=int, default=2, help="Number of child tags explored per depth.")
    parser.add_argument("--max-model-len", type=int, default=2500, help="Maximum model context length.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilisation passed to vLLM.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    assign_keys(args)


if __name__ == "__main__":
    main()
