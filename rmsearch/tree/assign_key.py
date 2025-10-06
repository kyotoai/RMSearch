"""Assign queries to tag-tree paths using iterative LLM scoring."""

from __future__ import annotations

import asyncio
import copy
import heapq
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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

    # query2tag_ids structure -> [{"tag_ids": [[int, ...], ...]}]
    # tag2query structure -> tag_tree copy augmented with "query_ids" lists
    return query2tag_ids, tag2query


if __name__ == "__main__":
    def fake_search(requests: List[Dict[str, Any]]):
        outputs = []
        for req in requests:
            outputs.append(
                {
                    "keys": [
                        {"key_id": min(idx, 1), "relevance": 1.0 - 0.1 * idx}
                        for idx in range(len(req["keys"]))
                    ]
                }
            )
        return outputs

    sample_queries = [{"query": "Explain retrieval"}]
    sample_tree = [
        {"tag": "AI", "children": [{"tag": "IR", "children": []}, {"tag": "Cooking", "children": []}]},
        {"tag": "General", "children": []},
    ]

    q2tag, t2query = assign_key_to_tag_tree(sample_queries, sample_tree, search_fn=fake_search, k_tag=2)
    print(q2tag)
    print(t2query)
