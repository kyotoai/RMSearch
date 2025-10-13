"""Utility helpers for converting between tag graph parquet files and tree structures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _ensure_list(value: Any) -> List[Any]:
    if _is_null(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _ensure_int_list(value: Any) -> List[int]:
    items = []
    for item in _ensure_list(value):
        try:
            items.append(int(item))
        except Exception:
            continue
    return items


def _strip_private_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if not k.startswith("_")}


def load_tag_graph(path: Path) -> List[Dict[str, Any]]:
    """Load tag_graph.parquet into a tree-of-dicts structure."""
    df = pd.read_parquet(path)
    raw_records = {int(row["tag_id"]): row for row in df.to_dict("records")}

    children_map = {
        tag_id: _ensure_int_list(record.get("children_tag_ids"))
        for tag_id, record in raw_records.items()
    }
    parent_map: Dict[int, Optional[int]] = {}
    for tag_id, record in raw_records.items():
        parent = record.get("parent_tag_id")
        parent_map[tag_id] = None if _is_null(parent) else int(parent)

    # Sort roots by recorded tree_path to preserve original ordering
    def _path_key(tag_id: int) -> Sequence[int]:
        record = raw_records[tag_id]
        tree_path = _ensure_list(record.get("tree_path"))
        return [int(x) for x in tree_path] if tree_path else [tag_id]

    root_ids = sorted([tag_id for tag_id, parent in parent_map.items() if parent is None], key=_path_key)

    def build_node(tag_id: int) -> Dict[str, Any]:
        record = raw_records[tag_id]
        node: Dict[str, Any] = {
            "tag_id": tag_id,
            "tag": record.get("tag"),
            "tags": _ensure_list(record.get("tags")),
            "key_ids": _ensure_int_list(record.get("key_ids")),
            "children": [],
            "child_tag_ids": children_map[tag_id],
        }
        existing_query_ids = _ensure_int_list(record.get("query_ids"))
        if existing_query_ids:
            node["query_ids"] = existing_query_ids
        extras = {
            k: v
            for k, v in record.items()
            if k
            not in {
                "tag_id",
                "tag",
                "tags",
                "key_ids",
                "children_tag_ids",
                "query_ids",
                "parent_tag_id",
                "tree_path",
                "depth",
            }
        }
        if extras:
            node["extra"] = _strip_private_fields(extras)
        for child_id in children_map[tag_id]:
            node["children"].append(build_node(child_id))
        return node

    return [build_node(tag_id) for tag_id in root_ids]


def index_tree_by_id(tree: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}

    def visit(node: Dict[str, Any]) -> None:
        lookup[node["tag_id"]] = node
        for child in node.get("children", []):
            visit(child)

    for root in tree:
        visit(root)
    return lookup


def flatten_tree(tree: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten a tree-of-dicts back into records suitable for parquet."""
    records: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any], parent_tag_id: Optional[int], path: List[int]) -> None:
        children = node.get("children", [])
        child_ids = [child["tag_id"] for child in children]
        key_ids = {int(idx) for idx in _ensure_int_list(node.get("key_ids"))}
        for alias in ("query_ids",):
            key_ids.update(int(idx) for idx in _ensure_int_list(node.get(alias)))

        record: Dict[str, Any] = {
            "tag_id": int(node["tag_id"]),
            "parent_tag_id": parent_tag_id,
            "children_tag_ids": child_ids,
            "tag": node.get("tag"),
            "tags": _ensure_list(node.get("tags")),
            "key_ids": sorted(key_ids),
            "tree_path": path.copy(),
            "depth": len(path),
        }
        if "extra" in node:
            record.update(_strip_private_fields(dict(node["extra"])))
        records.append(record)

        for idx, child in enumerate(children):
            visit(child, int(node["tag_id"]), path + [idx])

    for idx, root in enumerate(tree):
        visit(root, None, [idx])
    return records


def index_path_to_tag_ids(tree: Sequence[Dict[str, Any]], index_path: Sequence[int]) -> List[int]:
    node_list: Sequence[Dict[str, Any]] = tree
    tag_ids: List[int] = []
    for idx in index_path:
        if idx < 0 or idx >= len(node_list):
            return []
        node = node_list[idx]
        tag_ids.append(int(node["tag_id"]))
        node_list = node.get("children", [])
    return tag_ids


def follow_index_path(
    tree: Sequence[Dict[str, Any]], index_path: Sequence[int]
) -> Optional[Dict[str, Any]]:
    node_list: Sequence[Dict[str, Any]] = tree
    node: Optional[Dict[str, Any]] = None
    for idx in index_path:
        if idx < 0 or idx >= len(node_list):
            return None
        node = node_list[idx]
        node_list = node.get("children", [])
    return node


def iter_nodes(tree: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for root in tree:
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.get("children", [])))
