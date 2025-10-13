"""Convert a hierarchical tag tree JSON into a flat parquet graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

__all__ = ["tree_to_graph_records", "convert_tree_to_graph"]


def _normalise_tree(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "children" in data and isinstance(data["children"], list):
            return data["children"]
    raise TypeError("tag tree JSON must be a list or a mapping with a 'children' list")


def tree_to_graph_records(tag_tree: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    next_id = 0

    def visit(node: Dict[str, Any], path: List[int], parent_id: Optional[int]) -> int:
        nonlocal next_id
        tag_id = next_id
        next_id += 1

        children = node.get("children") or []
        record: Dict[str, Any] = {
            "tag_id": tag_id,
            "parent_tag_id": parent_id,
            "tree_path": path.copy(),
            "depth": len(path),
            "tag": node.get("tag"),
            "tags": node.get("tags"),
            "children_tag_ids": [],
        }
        if "key_ids" in node:
            record["key_ids"] = [int(idx) for idx in node.get("key_ids", [])]
        if "query_ids" in node:
            record["query_ids"] = [int(idx) for idx in node.get("query_ids", [])]
        records.append(record)
        record_idx = len(records) - 1

        for child_idx, child in enumerate(children):
            child_id = visit(child, path + [child_idx], tag_id)
            records[record_idx]["children_tag_ids"].append(child_id)

        return tag_id

    for root_idx, root in enumerate(tag_tree):
        visit(root, [root_idx], None)

    return records


def convert_tree_to_graph(tag_tree_path: Path, output_path: Path) -> None:
    data = json.loads(tag_tree_path.read_text(encoding="utf-8"))
    tag_tree = _normalise_tree(data)
    records = tree_to_graph_records(tag_tree)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert tag_tree_recs.json into tag_graph.parquet")
    parser.add_argument("--tag-tree", type=Path, required=True, help="Input tag tree JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination parquet path (defaults to <tag-tree-dir>/tag_graph.parquet).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.tag_tree.exists():
        raise FileNotFoundError(f"Tag tree file not found: {args.tag_tree}")

    output_path = args.output
    if output_path is None:
        output_path = args.tag_tree.with_name("tag_graph.parquet")

    convert_tree_to_graph(args.tag_tree, output_path)
    print(f"Wrote tag graph with {output_path}")


if __name__ == "__main__":
    main()
