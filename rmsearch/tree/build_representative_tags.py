"""Async helper for assigning representative tags to a hierarchical tag tree."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "build_representative_tags",
    "get_representative_tag_request",
    "set_representative_tag",
    "get_node_by_path",
    "is_leaf",
    "extract_text",
]

RequestFn = Callable[[List[str]], Awaitable[List[str]]]


def extract_text(text_c: str, tag_name: str) -> Optional[str]:
    import re

    pattern = rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>"
    match = re.search(pattern, text_c, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def get_node_by_path(tag_tree_recs: List[Dict[str, Any]], tag_ids: Sequence[int]) -> Dict[str, Any]:
    node: Dict[str, Any] = {"children": tag_tree_recs}
    for idx in tag_ids:
        node = node["children"][idx]
    return node


def set_representative_tag(
    tag_tree_recs: List[Dict[str, Any]],
    tag_ids: Sequence[int],
    representative_tag: str,
) -> List[Dict[str, Any]]:
    if not tag_ids:
        raise ValueError("tag_ids must contain at least one index")
    node = get_node_by_path(tag_tree_recs, tag_ids)
    node["tag"] = representative_tag
    return tag_tree_recs


def is_leaf(node: Dict[str, Any]) -> bool:
    return not node.get("children")


async def get_representative_tag_request(
    tag_tree_recs: List[Dict[str, Any]],
    *,
    request_func: RequestFn,
    n_tag_sample: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Propose representative tags for all parents whose children already have tags."""
    pending: List[Dict[str, Any]] = []
    progress_made = False

    def ensure_leaf_tags(node: Dict[str, Any]) -> None:
        nonlocal progress_made
        if is_leaf(node):
            if "tag" not in node:
                tags = node.get("tags", [])
                node["tag"] = tags[0] if tags else "general"
                progress_made = True
        else:
            for child in node.get("children", []):
                ensure_leaf_tags(child)

    def enqueue(node_list: List[Dict[str, Any]], path: List[int]) -> None:
        for idx, child in enumerate(node_list):
            if not is_leaf(child):
                enqueue(child["children"], path + [idx])

        if not path:
            return

        parent = get_node_by_path(tag_tree_recs, path)
        if "tag" in parent:
            return

        child_tags = [child.get("tag") for child in node_list if "tag" in child]
        if len(child_tags) != len(node_list):
            return

        sample = child_tags if len(child_tags) <= n_tag_sample else random.sample(child_tags, n_tag_sample)
        if not sample:
            sample = ["general"]
        lines = "\n".join(f"- {tag}" for tag in sample)
        prompt = (
            "You are a taxonomy expert. Given the following sample tags from one cluster,\n"
            "produce ONE concise representative tag (≤ 6 words) that best describes them all.\n"
            "Do NOT include punctuation at the end. Output ONLY the tag text, nothing else.\n\n"
            f"Sample tags:\n{lines}\n\nRepresentative tag:"
        )
        pending.append({"path": path, "prompt": prompt})

    synthetic_root = {"children": tag_tree_recs}
    ensure_leaf_tags(synthetic_root)
    enqueue(tag_tree_recs, [])

    if not pending and not progress_made:
        return tag_tree_recs, True

    prompts = [entry["prompt"] for entry in pending]
    outputs = await request_func(prompts)

    if len(outputs) != len(pending):
        raise ValueError("request_func returned mismatched number of outputs")

    for entry, output in zip(pending, outputs):
        tag_text = extract_text(output, "tag") or output.strip()
        tag_text = tag_text or "general"
        set_representative_tag(tag_tree_recs, entry["path"], tag_text)

    return tag_tree_recs, False


async def build_representative_tags(
    tag_tree_recs: List[Dict[str, Any]],
    *,
    request_func: Optional[RequestFn] = None,
    n_tag_sample: int = 6,
    save_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Iteratively populate parent ``tag`` fields by calling ``request_func``.

    ``tag_tree_recs`` structure -> ``[{"tag": str, "tags": [str], "children": [...]}]``.
    """

    # tag_tree_recs structure -> [{"tag": str, "tags": [str], "children": [...]}]
    if request_func is None:
        async def _missing(prompts: List[str]) -> List[str]:
            raise RuntimeError("request_func must be provided to build_representative_tags")

        request_func = _missing

    converged = False
    while not converged:
        tag_tree_recs, converged = await get_representative_tag_request(
            tag_tree_recs,
            request_func=request_func,
            n_tag_sample=n_tag_sample,
        )

    if save_path is not None:
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(tag_tree_recs, handle, ensure_ascii=False, indent=2)

    return tag_tree_recs


if __name__ == "__main__":
    async def _demo_request(prompts: List[str]) -> List[str]:
        return [f"<tag>Representative {idx}</tag>" for idx, _ in enumerate(prompts)]

    sample_tree = [
        {"tags": ["graph", "retrieval"], "children": []},
        {"tags": ["reward", "policy"], "children": []},
        {
            "children": [
                {"tags": ["planning", "agents"], "children": []},
                {"tags": ["coordination", "workflow"], "children": []},
            ]
        },
    ]

    result = asyncio.run(
        build_representative_tags(
            sample_tree,
            request_func=_demo_request,
            n_tag_sample=2,
        )
    )
    print(json.dumps(result, indent=2))
