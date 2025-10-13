"""Helper for assigning representative tags to a hierarchical tag tree."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vllm import SamplingParams

from rmsearch.utils.vllm_generate import LLMWorkerModel, build_llm

__all__ = [
    "build_representative_tags_v2",
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
    max_sample_children: int,
    max_sample_other: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect prompts for parents whose children already have tags."""

    pending: List[Dict[str, Any]] = []
    progress_made = False

    def _dedupe(items: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for item in items:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

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

    def collect_tags(node: Dict[str, Any]) -> List[str]:
        tags: List[str] = []
        tag_val = node.get("tag")
        if tag_val:
            tags.append(tag_val)
        for child in node.get("children", []):
            tags.extend(collect_tags(child))
        return tags

    def collect_tags_from_list(nodes: Sequence[Dict[str, Any]]) -> List[str]:
        collected: List[str] = []
        for node in nodes:
            collected.extend(collect_tags(node))
        return collected

    synthetic_root = {"children": tag_tree_recs}
    ensure_leaf_tags(synthetic_root)

    all_tags = _dedupe(collect_tags_from_list(tag_tree_recs))

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

        child_tags = _dedupe(child_tags)
        if not child_tags:
            child_tags = ["general"]
        if len(child_tags) > max_sample_children:
            sampled_children = random.sample(child_tags, max_sample_children)
        else:
            sampled_children = child_tags

        subtree_tags = _dedupe(collect_tags(parent))
        contrast_pool = [tag for tag in all_tags if tag not in subtree_tags]
        if contrast_pool:
            if len(contrast_pool) > max_sample_other:
                sampled_contrast = random.sample(contrast_pool, max_sample_other)
            else:
                sampled_contrast = contrast_pool
        else:
            sampled_contrast = ["general"]

        child_lines = "\n".join(f"- {tag}" for tag in sampled_children)
        contrast_lines = "\n".join(f"- {tag}" for tag in sampled_contrast)
        prompt = (
            "You are curating a hierarchical taxonomy. Review the child tags belonging to a single cluster\n"
            "and craft ONE descriptive representative tag (7-12 words) that captures their shared theme.\n"
            "Use the contrast tags to avoid overly generic wording. Return only the representative tag text,\n"
            "without punctuation or additional commentary.\n\n"
            f"Child sample tags:\n{child_lines}\n\n"
            f"Contrast tags from other clusters:\n{contrast_lines}\n\n"
            "Representative tag:"
        )
        pending.append({"path": path, "prompt": prompt})

    enqueue(tag_tree_recs, [])
    return pending, progress_made


def _run_iteration(
    tag_tree_recs: List[Dict[str, Any]],
    *,
    model: LLMWorkerModel,
    sampling_params: SamplingParams,
    worker_batch_size: int,
    timeout_s: Optional[float],
    max_sample_children: int,
    max_sample_other: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    pending, progress_made = _prepare_requests(
        tag_tree_recs,
        max_sample_children=max_sample_children,
        max_sample_other=max_sample_other,
    )

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

        for entry, output_text in zip(pending, outputs):
            tag_text = extract_text(output_text, "tag") or output_text.strip()
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
    max_sample_children: int = 20,
    max_sample_other: int = 20,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return build_representative_tags_v2(
        tag_tree_recs,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=device_groups,
        llm_kwargs=llm_kwargs,
        sampling_params=sampling_params,
        worker_batch_size=worker_batch_size,
        timeout_s=timeout_s,
        max_sample_children=max_sample_children,
        max_sample_other=max_sample_other,
        save_path=save_path,
    )


def build_representative_tags_v2(
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
    max_sample_children: int = 20,
    max_sample_other: int = 20,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if llm_kwargs is None:
        llm_kwargs = {}

    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=32)

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
                sampling_params=sampling_params,
                worker_batch_size=worker_batch_size,
                timeout_s=timeout_s,
                max_sample_children=max_sample_children,
                max_sample_other=max_sample_other,
            )
    finally:
        model.close()

    if save_path is not None:
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(tag_tree_recs, handle, ensure_ascii=False, indent=2)

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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilisation passed to vLLM.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional maximum model context length.")
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype override for the vLLM engine.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading from HF Hub.")
    parser.add_argument("--worker-batch-size", type=int, default=8, help="Prompts processed per worker batch.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional timeout (s) for each worker batch.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for prompt generation.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top_p value.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens generated per prompt.")
    parser.add_argument(
        "--max-sample-children",
        type=int,
        default=20,
        help="Maximum number of child tags to include in each prompt.",
    )
    parser.add_argument(
        "--max-sample-other",
        type=int,
        default=20,
        help="Maximum number of contrast tags sampled from other branches.",
    )
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

    llm_kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True

    output_path = args.output or args.tag_tree

    updated_tree = build_representative_tags_v2(
        tag_tree,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        device_groups=device_groups,
        llm_kwargs=llm_kwargs,
        sampling_params=sampling,
        worker_batch_size=args.worker_batch_size,
        timeout_s=args.timeout,
        max_sample_children=args.max_sample_children,
        max_sample_other=args.max_sample_other,
        save_path=str(output_path),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saved updated tag tree to {output_path}")
