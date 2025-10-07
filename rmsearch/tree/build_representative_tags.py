"""Helper for assigning representative tags to a hierarchical tag tree."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import argparse

from vllm import SamplingParams

from rmsearch.utils.vllm_generate import LLMWorkerModel, build_llm

__all__ = [
    "build_representative_tags",
    "set_representative_tag",
    "get_node_by_path",
    "is_leaf",
    "extract_text",
]


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


def _prepare_requests(
    tag_tree_recs: List[Dict[str, Any]],
    *,
    n_tag_sample: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect prompts for parents whose children already have tags."""

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

    return pending, progress_made


def _run_iteration(
    tag_tree_recs: List[Dict[str, Any]],
    *,
    model: LLMWorkerModel,
    sampling_params: SamplingParams,
    worker_batch_size: int,
    timeout_s: Optional[float],
    n_tag_sample: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    pending, progress_made = _prepare_requests(tag_tree_recs, n_tag_sample=n_tag_sample)

    if not pending and not progress_made:
        return tag_tree_recs, True

    if pending:
        prompts = [entry["prompt"] for entry in pending]
        outputs = model.generate(
            prompts,
            sampling_params=sampling_params,
            batch_size=worker_batch_size,
            timeout_s=timeout_s,
        )

        if len(outputs) != len(pending):
            raise RuntimeError("vLLM worker returned mismatched number of outputs")

        for entry, output in zip(pending, outputs):
            tag_text = extract_text(output, "tag") or output.strip()
            tag_text = tag_text or "general"
            set_representative_tag(tag_tree_recs, entry["path"], tag_text)

    return tag_tree_recs, False


def build_representative_tags(
    tag_tree_recs: List[Dict[str, Any]],
    *,
    model_name: str,
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    device_groups: Optional[List[List[int]]] = None,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[SamplingParams] = None,
    worker_batch_size: int = 8,
    timeout_s: Optional[float] = None,
    n_tag_sample: int = 6,
    save_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Iteratively populate parent ``tag`` fields using a vLLM worker pool.

    ``tag_tree_recs`` structure -> ``[{"tag": str, "tags": [str], "children": [...]}]``.
    """

    sampling = sampling_params or SamplingParams(temperature=0.0, top_p=0.9, max_tokens=32)
    llm_kwargs = llm_kwargs or {}

    model = build_llm(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=device_groups,
        **llm_kwargs,
    )

    try:
        converged = False
        while not converged:
            tag_tree_recs, converged = _run_iteration(
                tag_tree_recs,
                model=model,
                sampling_params=sampling,
                worker_batch_size=worker_batch_size,
                timeout_s=timeout_s,
                n_tag_sample=n_tag_sample,
            )
    finally:
        model.close()

    if save_path is not None:
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(tag_tree_recs, handle, ensure_ascii=False, indent=2)

    # tag_tree_recs (list): hierarchical structure where each node resembles
    #   {"tag": "<representative label>",
    #    "tags": ["<leaf tag>", ...] (only present for leaves),
    #    "children": [<child nodes>] (omitted on leaves)}.
    return tag_tree_recs


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
    parser = argparse.ArgumentParser(description="Populate parent tags in a tag tree using a vLLM model.")
    parser.add_argument("--tag-tree", type=Path, required=True, help="Input tag tree JSON file.")
    parser.add_argument("--output", type=Path, help="Destination path for the updated tag tree JSON.")
    parser.add_argument("--model-name", type=str, required=True, help="Generation model name or path.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor_parallel_size per worker instance.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of worker instances to launch.")
    parser.add_argument(
        "--device-groups",
        type=str,
        help="Explicit GPU mapping, e.g. '0,1;2,3' for two workers with tensor_parallel_size=2.",
    )
    parser.add_argument("--worker-batch-size", type=int, default=8, help="Prompts processed per worker batch.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional timeout (s) for each worker batch.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for prompt generation.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top_p value.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens generated per prompt.")
    parser.add_argument("--n-tag-sample", type=int, default=6, help="Number of child tags to include in each prompt.")
    args = parser.parse_args()

    if not args.tag_tree.exists():
        raise FileNotFoundError(f"Tag tree file not found: {args.tag_tree}")

    tag_tree = json.loads(args.tag_tree.read_text())

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    device_groups = _parse_device_groups(
        args.device_groups,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
    )

    output_path = args.output or args.tag_tree

    updated_tree = build_representative_tags(
        tag_tree,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        device_groups=device_groups,
        llm_kwargs={},
        sampling_params=sampling,
        worker_batch_size=args.worker_batch_size,
        timeout_s=args.timeout,
        n_tag_sample=args.n_tag_sample,
        save_path=str(output_path),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saved updated tag tree to {output_path}")
