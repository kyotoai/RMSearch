"""Construct a hierarchical tag tree from precomputed tag embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from rmsearch.tree.hierarchical_kmeans import HierarchicalKMeans

__all__ = ["build_tag_tree"]


def _load_embeddings(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(f"Embeddings tensor must be 2D, found shape {tuple(tensor.shape)}")
    return tensor.detach().cpu()


def _load_tag_meta(path: Path) -> List[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tag metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("tag_meta.json must contain a JSON list.")
    return data


def build_tag_tree(
    embeddings: torch.Tensor,
    *,
    n_clusters: int,
    max_leaf_size: int,
    random_state: int | None,
) -> List[Dict[str, Any]]:
    model = HierarchicalKMeans(
        n_clusters=n_clusters,
        max_leaf_size=max_leaf_size,
        random_state=random_state,
    )
    model.fit(embeddings)
    leaf_members = model.leaf_members()
    return HierarchicalKMeans.convert_tree_dict_to_json(leaf_members)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hierarchical tag tree from tag embeddings.")
    parser.add_argument("--working-dir", type=Path, default=Path("."), help="Project root that contains the data directory.")
    parser.add_argument("--data-name", type=str, required=True, help="Dataset name inside the working directory's data folder.")
    parser.add_argument("--embeddings", type=Path, default=None, help="Optional direct path to tag_embeddings.pt.")
    parser.add_argument("--tag-meta", type=Path, default=None, help="Optional direct path to tag_meta.json.")
    parser.add_argument("--output", type=Path, default=None, help="Destination for tag_tree_recs.json.")
    parser.add_argument("--branching-factor", type=int, default=10, help="Maximum number of child clusters per internal node.")
    parser.add_argument("--max-leaf-size", type=int, default=60, help="Maximum number of items per leaf cluster.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for k-means initialisation.")
    args = parser.parse_args()

    data_dir = args.working_dir / "data" / args.data_name

    embeddings_path = args.embeddings or (data_dir / "tag_embeddings.pt")
    tag_meta_path = args.tag_meta or (data_dir / "tag_meta.json")
    output_path = args.output or (data_dir / "tag_tree_recs.json")

    embeddings = _load_embeddings(embeddings_path)
    tag_meta = _load_tag_meta(tag_meta_path)
    if embeddings.shape[0] != len(tag_meta):
        raise ValueError(
            "tag_embeddings.pt row count does not match tag_meta.json length "
            f"({embeddings.shape[0]} vs {len(tag_meta)})."
        )

    tree = build_tag_tree(
        embeddings,
        n_clusters=args.branching_factor,
        max_leaf_size=args.max_leaf_size,
        random_state=args.random_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(tree, handle, ensure_ascii=False, indent=2)

    print(f"Saved tag tree with {len(tree)} root nodes to {output_path}")


if __name__ == "__main__":
    main()

