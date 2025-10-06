"""Retrieval evaluation utilities."""

from __future__ import annotations

import asyncio
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
    def fake_search(requests: List[Dict[str, Any]]):
        result = []
        for req in requests:
            keys = req["keys"]
            result.append(
                {
                    "keys": [
                        {"key_id": i, "relevance": 1.0 - 0.1 * i, "text": text}
                        for i, text in enumerate(keys)
                    ]
                }
            )
        return result

    queries = ["Explain retrieval"]
    sentences = ["Retrieval augments generation.", "Cooking is fun."]
    tag_tree = [
        {"tag": "AI", "children": [{"tag": "IR", "children": [], "query_ids": [0]}, {"tag": "Cooking", "children": [], "query_ids": [1]}]},
    ]

    outputs = retrieval_evaluation(queries, sentences, tag_tree, search_fn=fake_search)
    print(outputs)
