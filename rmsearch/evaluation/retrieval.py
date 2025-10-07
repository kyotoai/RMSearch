"""Retrieval evaluation utilities."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from rmsearch.tree.assign_key import assign_key_to_tag_tree

__all__ = ["retrieval_evaluation"]

SearchFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
AsyncSearchFn = Callable[[List[Dict[str, Any]]], Awaitable[List[Dict[str, Any]]]]


def _get_tag_dict(tag_ids: List[int], tag_tree: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not tag_ids:
        return None
    node: Any = tag_tree
    for tag_id in tag_ids[:-1]:
        if not isinstance(node, list) or tag_id >= len(node):
            return None
        node = node[tag_id].get("children", [])
    if not isinstance(node, list) or tag_ids[-1] >= len(node):
        return None
    return node[tag_ids[-1]]


def _run_search(search_fn: SearchFn | AsyncSearchFn, requests: List[Dict[str, Any]]):
    maybe_result = search_fn(requests)
    if asyncio.iscoroutine(maybe_result):
        return asyncio.run(maybe_result)
    return maybe_result


def retrieval_evaluation(
    queries: Sequence[str],
    sentences: Sequence[str],
    tag_tree: List[Dict[str, Any]],
    *,
    search_fn: SearchFn | AsyncSearchFn,
    k_tag: int = 2,
    k_key: int = 10,
    correct_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Score sentence relevance by navigating the tag tree before ranking keys."""

    query_records = [{"query": query} for query in queries]
    query2tag_ids, tag2query = assign_key_to_tag_tree(query_records, tag_tree, search_fn=search_fn, k_tag=k_tag)

    requests: List[Dict[str, Any]] = []
    for query_id, record in enumerate(query2tag_ids):
        combined_key_ids: List[int] = []
        for tag_ids in record["tag_ids"]:
            tag_info = _get_tag_dict(tag_ids, tag2query)
            if tag_info and "query_ids" in tag_info:
                combined_key_ids.extend(int(idx) for idx in tag_info["query_ids"])
        unique_keys = []
        seen = set()
        for idx in combined_key_ids:
            if 0 <= idx < len(sentences) and idx not in seen:
                seen.add(idx)
                unique_keys.append(idx)
        selected_sentences = [sentences[idx] for idx in unique_keys]
        requests.append(
            {
                "query": queries[query_id],
                "keys": selected_sentences,
                "k": k_key,
                "return_relevance": True,
                "key_ids": unique_keys,
            }
        )

    outputs = _run_search(search_fn, requests)

    for idx, record in enumerate(outputs):
        record["correct_id"] = int(correct_ids[idx]) if correct_ids and idx < len(correct_ids) else None
        for key_entry, key_id in zip(record.get("keys", []), requests[idx]["key_ids"]):
            key_entry["relevant_id"] = key_id

    return outputs


if __name__ == "__main__":
    import argparse
    import json
    import logging
    import multiprocessing as mp

    import pandas as pd

    from rmsearch.utils.vllm_reward2 import build_llm, search

    parser = argparse.ArgumentParser(description="Run retrieval evaluation using a vLLM reward model.")
    parser.add_argument("--working-dir", type=Path, default=Path("/workspace/RMS_exp"), help="Root working directory used during training.")
    parser.add_argument("--data-name", type=str, default="smollm-corpus", help="Dataset identifier under the working directory.")
    parser.add_argument("--model-name", type=str, default="/workspace/llama3b-rm-converted-model", help="Path to the converted reward model.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for the reward model workers.")
    parser.add_argument("--num-instances", type=int, default=4, help="Number of reward model worker instances.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size per inference call.")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout in seconds for reward model requests.")
    parser.add_argument("--k-tag", type=int, default=2, help="Branching factor when traversing the tag tree.")
    parser.add_argument("--k-key", type=int, default=10, help="Number of keys retrieved per query in the final stage.")
    parser.add_argument("--output", type=Path, default=Path("relevance_dict.json"), help="Where to save the evaluation results.")
    args = parser.parse_args()

    working_dir = args.working_dir
    data_dir = working_dir / "data" / args.data_name

    df = pd.read_csv(data_dir / "df_small.csv")
    with (data_dir / "query_dict.json").open() as handle:
        query_dict = json.load(handle)
    with (data_dir / "tag2query-tag_tree.json").open() as handle:
        tag_tree = json.load(handle)

    sentences = [df.iloc[i]["text"] for i in range(len(df))]
    queries: List[str] = []
    correct_ids: List[int] = []
    for idx in range(len(df)):
        questions = query_dict[str(idx)]["questions"]
        queries.extend(questions)
        correct_ids.extend([idx for _ in range(len(questions))])

    logging.getLogger("vllm").setLevel(logging.ERROR)
    mp.set_start_method("spawn", force=True)

    device_groups: List[List[int]] = []
    device_id = 0
    for _ in range(args.num_instances):
        group = []
        for _ in range(args.tensor_parallel_size):
            group.append(device_id)
            device_id += 1
        device_groups.append(group)

    rm = build_llm(
        model_name=args.model_name,
        tensor_parallel_size=len(device_groups[0]) if device_groups else args.tensor_parallel_size,
        num_instances=len(device_groups) or args.num_instances,
        device_groups=device_groups if device_groups else None,
        max_model_len=2500,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
        runner="pooling",
    )
    tokenizer = rm.tokenizer

    def llm_template_func(row: Dict[str, Any]) -> str:
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
        outputs = retrieval_evaluation(
            queries,
            sentences,
            tag_tree,
            search_fn=run_search,
            k_tag=args.k_tag,
            k_key=args.k_key,
            correct_ids=correct_ids,
        )
    finally:
        rm.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, ensure_ascii=False, indent=2)

    print(f"Saved retrieval evaluation results to {args.output}")
