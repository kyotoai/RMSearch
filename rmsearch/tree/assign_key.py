"""Assign queries to tag-tree paths using iterative LLM scoring."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import heapq
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from rmsearch.utils.vllm_reward2 import build_llm, search

__all__ = ["assign_key_to_tag_tree"]

SearchFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
AsyncSearchFn = Callable[[List[Dict[str, Any]]], Awaitable[List[Dict[str, Any]]]]


def _get_tag_dict(tag_ids: List[int], tree: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not tag_ids:
        return None
    node: Any = tree
    for tag_id in tag_ids[:-1]:
        if not isinstance(node, list) or tag_id >= len(node):
            return None
        node = node[tag_id].get("children", [])
    leaf = node[tag_ids[-1]] if isinstance(node, list) and tag_ids[-1] < len(node) else None
    return leaf


def _set_query_id(tag2query: List[Dict[str, Any]], tag_ids: List[int], query_id: int) -> None:
    subtree: List[Dict[str, Any]] = tag2query
    for tag_id in tag_ids:
        node = subtree[tag_id]
        node.setdefault("query_ids", []).append(query_id)
        subtree = node.get("children", [])


def _run_search(search_fn: SearchFn | AsyncSearchFn, requests: List[Dict[str, Any]]):
    maybe_result = search_fn(requests)
    if asyncio.iscoroutine(maybe_result):
        return asyncio.run(maybe_result)
    return maybe_result


def assign_key_to_tag_tree(
    queries: Sequence[Dict[str, Any]],
    tag_tree: List[Dict[str, Any]],
    *,
    search_fn: SearchFn | AsyncSearchFn,
    k_tag: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Iteratively expand tag assignments by querying an LLM ranking backend.

    ``tag_tree`` structure -> ``[{"tag": str, "children": [...]}]``. Each node may also
    contain metadata fields such as ``tag_ids``.
    """

    tag2query = copy.deepcopy(tag_tree)
    query2tag_ids = [{"tag_ids": [[] for _ in range(k_tag)]} for _ in range(len(queries))]

    root_tags = [node.get("tag", "") for node in tag_tree]
    tags_request = [{"tags": [root_tags]} for _ in range(len(queries))]

    while_end = False
    depth = 0

    while not while_end:
        depth += 1
        requests: List[Dict[str, Any]] = []
        query_and_slot: List[Tuple[int, int]] = []

        for query_id, record in enumerate(tags_request):
            for nth_tag, tag_list in enumerate(record["tags"]):
                query_and_slot.append((query_id, nth_tag))
                requests.append(
                    {
                        "query": queries[query_id]["query"],
                        "keys": tag_list,
                        "k": k_tag,
                        "return_relevance": True,
                    }
                )

        if not requests:
            break

        outputs = _run_search(search_fn, requests)
        tags_request = [{"tags": []} for _ in range(len(queries))]
        results: Dict[int, Dict[str, List[Any]]] = {
            query_id: {"tag_ids_list": [], "relevance_list": []} for query_id in range(len(queries))
        }

        for request_idx, output_dict in enumerate(outputs):
            query_id, slot = query_and_slot[request_idx]
            prior_tag_ids = query2tag_ids[query_id]["tag_ids"][slot]
            for top_idx in range(k_tag):
                try:
                    new_tag_id = output_dict["keys"][top_idx]["key_id"]
                    relevance = output_dict["keys"][top_idx].get("relevance", 0.0)
                except Exception:
                    continue
                results[query_id]["tag_ids_list"].append(prior_tag_ids + [int(new_tag_id)])
                results[query_id]["relevance_list"].append(float(relevance))

        while_end = True
        for query_id, holder in results.items():
            relevance = holder["relevance_list"]
            tag_ids_list = holder["tag_ids_list"]
            if not relevance:
                top_paths: List[List[int]] = []
            else:
                top_indices = heapq.nlargest(
                    min(k_tag, len(relevance)),
                    range(len(relevance)),
                    key=lambda idx: relevance[idx],
                )
                top_paths = [tag_ids_list[idx] for idx in top_indices]

            query2tag_ids[query_id]["tag_ids"] = top_paths

            for tag_ids in top_paths:
                tag_info = _get_tag_dict(tag_ids, tag2query)
                if not tag_info or not tag_info.get("children"):
                    continue
                child_tags = [child.get("tag", "") for child in tag_info["children"]]
                if child_tags:
                    while_end = False
                    tags_request[query_id]["tags"].append(child_tags)

    for query_id, record in enumerate(query2tag_ids):
        for tag_ids in record["tag_ids"]:
            _set_query_id(tag2query, tag_ids, query_id)

    # query2tag_ids (list): each element is
    #   {"tag_ids": [[<path indices to best leaf>, ...]]}
    #   where each inner list represents a path from root to a leaf in tag_tree.
    # tag2query (list of dicts): clone of the input tag tree with additional
    #   "query_ids" lists appended to nodes indicating which queries traverse them.
    return query2tag_ids, tag2query


def _load_queries(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.queries_json:
        data = json.loads(Path(args.queries_json).read_text())
        if isinstance(data, dict):
            # Accept mapping like {"0": {...}}
            values = list(data.values())
        else:
            values = data
        queries = []
        for item in values:
            if isinstance(item, str):
                queries.append({"query": item})
            elif isinstance(item, dict) and "query" in item:
                queries.append({"query": str(item["query"])})
        if not queries:
            raise ValueError("queries_json must contain strings or objects with a 'query' field")
        return queries

    if args.queries_csv:
        df = pd.read_csv(args.queries_csv)
        if args.query_column not in df.columns:
            raise ValueError(f"Column '{args.query_column}' not found in {args.queries_csv}")
        return [{"query": str(value)} for value in df[args.query_column].dropna().tolist()]

    raise ValueError("Provide --queries-json or --queries-csv")


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
    parser = argparse.ArgumentParser(description="Assign queries to a tag tree using a reward model search backend.")
    parser.add_argument("--tag-tree", type=Path, required=True, help="Path to the tag tree JSON file.")
    parser.add_argument("--queries-json", type=Path, help="JSON file containing either a list of strings or objects with 'query'.")
    parser.add_argument("--queries-csv", type=Path, help="CSV file containing queries.")
    parser.add_argument("--query-column", type=str, default="query", help="Column name to read queries from when using CSV input.")
    parser.add_argument("--query2tag-out", type=Path, required=True, help="Destination JSON file for query-to-tag assignments.")
    parser.add_argument("--tag2query-out", type=Path, required=True, help="Destination JSON file for augmented tag tree.")
    parser.add_argument("--model-name", type=str, required=True, help="Reward model name or path for scoring.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor_parallel_size per reward model instance.")
    parser.add_argument("--num-instances", type=int, default=4, help="Number of reward model worker instances.")
    parser.add_argument(
        "--device-groups",
        type=str,
        help="Explicit GPU mapping, e.g. '0,1;2,3' for two workers with tensor_parallel_size=2.",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Prompts processed per worker batch.")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout (s) for reward model batches.")
    parser.add_argument("--k-tag", type=int, default=2, help="Number of child tags explored per depth.")
    args = parser.parse_args()

    if not args.tag_tree.exists():
        raise FileNotFoundError(f"Tag tree file not found: {args.tag_tree}")

    queries = _load_queries(args)
    tag_tree = json.loads(args.tag_tree.read_text())

    device_groups = _parse_device_groups(
        args.device_groups,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
    )

    rm = build_llm(
        model_name=args.model_name,
        tensor_parallel_size=len(device_groups[0]) if device_groups else args.tensor_parallel_size,
        num_instances=len(device_groups) if device_groups else args.num_instances,
        device_groups=device_groups if device_groups else None,
        max_model_len=2500,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
        runner="pooling",
    )
    tokenizer = rm.tokenizer

    def llm_template_func(row: Dict[str, Any]) -> str:
        message = [
            {
                "role": "user",
                "content": (
                    "Generate tag for the sentence\n\n"
                    f"Sentence:'''{row['query']}'''"
                ),
            },
            {"role": "assistant", "content": str(row["key"])}
        ]
        if len(message[0]["content"]) > 4000:
            message[0]["content"] = message[0]["content"][:4000] + "..."
        return tokenizer.apply_chat_template(message, tokenize=False)

    def run_search(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not requests:
            return []
        topk = max((req.get("k", args.k_tag) for req in requests), default=args.k_tag)
        return search(
            rm,
            requests,
            llm_template_func,
            topk=topk,
            batch_size=args.batch_size,
            timeout_s=args.timeout,
        )

    try:
        query2tag, tag2query = assign_key_to_tag_tree(
            queries,
            tag_tree,
            search_fn=run_search,
            k_tag=args.k_tag,
        )
    finally:
        rm.close()

    args.query2tag_out.parent.mkdir(parents=True, exist_ok=True)
    args.query2tag_out.write_text(json.dumps(query2tag, ensure_ascii=False, indent=2))
    args.tag2query_out.parent.mkdir(parents=True, exist_ok=True)
    args.tag2query_out.write_text(json.dumps(tag2query, ensure_ascii=False, indent=2))
    print(f"Saved query2tag assignments to {args.query2tag_out}")
    print(f"Saved augmented tag tree to {args.tag2query_out}")
