"""Rank keys for each query by traversing a tag graph with a reward model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rmsearch.graph._graph_utils import iter_nodes, load_tag_graph
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


def _load_sequence(path: Path, field: str) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = list(data.values())
    else:
        values = data
    items: List[str] = []
    for value in values:
        if isinstance(value, str):
            items.append(value)
        elif isinstance(value, dict) and field in value:
            items.append(str(value[field]))
    if not items:
        raise ValueError(f"No textual entries with '{field}' found in {path}")
    return items


def _contains_key_ids(tree: Sequence[Dict[str, Any]]) -> bool:
    for node in iter_nodes(tree):
        if node.get("key_ids"):
            return True
    return False


def _build_tag_search_fn(args: argparse.Namespace, tokenizer, rm):
    def llm_template(row: Dict[str, Any]) -> str:
        message = [
            {
                "role": "user",
                "content": (
                    "Generate the most suitable tag for the following sentence.\n\n"
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
    requests: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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

    return search(
        rm,
        requests,
        llm_template,
        topk=args.k_key,
        query_batch_size=args.batch_size,
        batch_size=args.batch_size,
        timeout_s=args.timeout,
    )


def _collect_candidate_keys(
    tree: Sequence[Dict[str, Any]],
    path: Sequence[int],
) -> List[int]:
    node_list: Sequence[Dict[str, Any]] = tree
    collected: List[int] = []
    terminal: Optional[Dict[str, Any]] = None
    for idx in path:
        if idx < 0 or idx >= len(node_list):
            return []
        node = node_list[idx]
        node_list = node.get("children", [])
        terminal = node
    if terminal:
        collected.extend(int(k) for k in terminal.get("key_ids", []) if k is not None)
    if collected:
        seen = set()
        deduped: List[int] = []
        for key_id in collected:
            if key_id in seen:
                continue
            seen.add(key_id)
            deduped.append(key_id)
        return deduped
    # fall back to collecting along the path
    node_list = tree
    seen: set[int] = set()
    fallback: List[int] = []
    for idx in path:
        node = node_list[idx]
        for key_id in node.get("key_ids", []):
            key_int = int(key_id)
            if key_int in seen:
                continue
            seen.add(key_int)
            fallback.append(key_int)
        node_list = node.get("children", [])
    return fallback


def _accumulate_candidate_keys(
    tree: Sequence[Dict[str, Any]],
    paths: Sequence[Sequence[int]],
) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for path in paths:
        for key_id in _collect_candidate_keys(tree, path):
            if key_id in seen:
                continue
            seen.add(key_id)
            ordered.append(key_id)
    return ordered


def _prepare_key_requests(
    args: argparse.Namespace,
    queries: Sequence[str],
    keys: Sequence[str],
    tree: Sequence[Dict[str, Any]],
    assignments: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    requests: List[Dict[str, Any]] = []
    metas: List[List[int]] = []
    for query_text, entry in zip(queries, assignments):
        index_paths = entry.get("tag_ids", [])
        candidate_ids = _accumulate_candidate_keys(tree, index_paths)
        if not candidate_ids:
            limit = min(len(keys), max(args.k_key * 10, args.fallback_key_sample))
            candidate_ids = list(range(limit))
        selected_keys = [keys[idx] for idx in candidate_ids if 0 <= idx < len(keys)]
        requests.append({"query": query_text, "keys": selected_keys, "k": args.k_key})
        metas.append(candidate_ids)
    return requests, metas


def _build_outputs(
    queries: Sequence[str],
    keys: Sequence[str],
    metas: Sequence[List[int]],
    scored: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for query_text, candidate_ids, result in zip(queries, metas, scored):
        ranked: List[Dict[str, Any]] = []
        for key_info in result.get("keys", []):
            local_idx = int(key_info.get("key_id", 0))
            if not (0 <= local_idx < len(candidate_ids)):
                continue
            global_id = candidate_ids[local_idx]
            if not (0 <= global_id < len(keys)):
                continue
            ranked.append(
                {
                    "key_id": int(global_id),
                    "key": keys[global_id],
                    "relevance": float(key_info.get("relevance", 0.0)),
                }
            )
        outputs.append({"query": query_text, "keys": ranked})
    return outputs


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.queries and not args.keys and not args.tag2key:
        raise ValueError("Provide --queries, --keys, and --tag2key")

    queries = _load_sequence(args.queries, "query") if args.queries else []
    keys = _load_sequence(args.keys, "text") if args.keys else []
    tag_tree = load_tag_graph(args.tag2key)

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

        if not _contains_key_ids(tag_tree) and keys:
            key_records = [{"query": text} for text in keys]
            _, tag_tree = assign_key_to_tag_tree(
                key_records,
                tag_tree,
                search_fn=tag_search_fn,
                k_tag=args.k_tag,
            )

        query_records = [{"query": text} for text in queries]
        query_assignments, _ = assign_key_to_tag_tree(
            query_records,
            tag_tree,
            search_fn=tag_search_fn,
            k_tag=args.k_tag,
        )

        requests, metas = _prepare_key_requests(args, queries, keys, tag_tree, query_assignments)
        scored = _score_keys(args, rm, rm.tokenizer, requests)
        outputs = _build_outputs(queries, keys, metas, scored)
    finally:
        rm.close()

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search keys via the tag graph using a reward model.")
    parser.add_argument("--queries", type=Path, help="JSON file containing query strings or objects with a 'query' field.")
    parser.add_argument("--keys", type=Path, help="JSON file containing key strings or objects with a 'text' field.")
    parser.add_argument("--tag2key", type=Path, required=True, help="Parquet file with the tag graph annotated with key_ids.")
    parser.add_argument("--k-tag", type=int, default=2, help="Number of top tags explored per layer.")
    parser.add_argument("--k-key", type=int, default=5, help="Number of keys retrieved per query.")
    parser.add_argument("--model-name", type=str, required=True, help="Path or identifier of the reward model.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for the reward model.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of reward model worker processes.")
    parser.add_argument("--device-groups", type=str, help="Explicit GPU mapping, e.g. '0,1;2,3' for two workers.")
    parser.add_argument("--batch-size", type=int, default=128, help="Prompts processed per worker batch.")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout (s) for reward model batches.")
    parser.add_argument("--max-model-len", type=int, default=2500, help="Maximum sequence length for the reward model.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Fraction of GPU memory made available to workers.")
    parser.add_argument("--output", type=Path, help="Optional path to store ranked results in JSON format.")
    parser.add_argument("--fallback-key-sample", type=int, default=256, help="Number of keys scored when no candidates are attached to a tag path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved search results to {args.output}")
    else:
        for record in outputs:
            print(f"Query: {record['query']}")
            for item in record["keys"]:
                print(f"  • ({item['relevance']:.4f}) {item['key_id']}: {item['key']}")


if __name__ == "__main__":
    main()
