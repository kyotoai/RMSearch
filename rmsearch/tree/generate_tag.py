"""Utilities for generating candidate tags with vLLM worker models.

This module extracts the tag-generation portion of
``examples/train_en.ipynb`` so it can be reused from library code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from vllm import SamplingParams

from rmsearch.utils.vllm_generate import LLMWorkerModel, build_llm

__all__ = ["generate_tag", "build_model_from_settings", "build_pool_from_settings"]


def _json_list_from_text(text: str) -> List[str]:
    """Parse a model response into a clean list of tag strings."""
    text = text.strip()
    if not text:
        return []

    def _maybe_list(candidate: str) -> Optional[List[str]]:
        try:
            obj = json.loads(candidate)
        except Exception:
            return None
        if isinstance(obj, list) and all(isinstance(item, str) for item in obj):
            return [item.strip() for item in obj if item and item.strip()]
        return None

    parsed = _maybe_list(text)
    if parsed is not None:
        return parsed

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if fenced:
        parsed = _maybe_list(fenced.group(1))
        if parsed is not None:
            return parsed

    quoted = re.findall(r'"([^\"]{1,80})"', text)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]

    lines = [re.sub(r"^[-*•]\s*", "", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) <= 80]
    if lines:
        return lines[:5]

    return [text.splitlines()[0][:50]] if text else []


def _ensure_sampling(
    sampling: Optional[SamplingParams],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> SamplingParams:
    """Return the provided sampling params or a safe default."""
    if sampling is not None:
        return sampling
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def build_model_from_settings(model_name: str, settings: Dict[str, Any]) -> LLMWorkerModel:
    """Instantiate an ``LLMWorkerModel`` from the notebook-style settings dict."""
    tensor_parallel = settings["tensor_parallel_size"]
    num_instances = settings["num_instances"]
    device_groups = settings.get("device_groups")
    llm_kwargs = dict(settings.get("llm_kwargs") or {})
    return build_llm(
        model_name,
        tensor_parallel_size=tensor_parallel,
        num_instances=num_instances,
        device_groups=device_groups,
        **llm_kwargs,
    )


def build_pool_from_settings(model_name: str, settings: Dict[str, Any]) -> LLMWorkerModel:
    """Backward-compatible alias for older imports."""
    return build_model_from_settings(model_name, settings)


def generate_tag(
    keys: Iterable[str],
    *,
    model_name: str,
    model: Optional[LLMWorkerModel] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[SamplingParams] = None,
    worker_batch_size: int = 8,
    timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Generate tags for arbitrary key strings using a vLLM worker model.

    ``tag_recs`` structure -> ``[{"key": str, "key_id": int, "tags": [str, ...]}]``
    """

    key_list = list(keys)
    if not key_list:
        return []

    prompts = [
        (
            "You are a tagging assistant.\n"
            "Task: Create 3–6 short, specific tags (1–3 words each) "
            "that describe the following key/phrase.\n"
            "Output ONLY a JSON array of strings. No commentary.\n\n"
            f"Key: \"{key}\"\n\n"
            'Example Output: ["LLM Inference", "Vector Search", "RAG"]'
        )
        for key in key_list
    ]

    sampling = _ensure_sampling(
        sampling_params,
        temperature=0.3,
        top_p=0.9,
        max_tokens=128,
    )

    owns_model = False
    if model is None:
        if model_settings is None:
            raise ValueError("Either an LLM model or model_settings must be provided.")
        model = build_model_from_settings(model_name, model_settings)
        owns_model = True

    try:
        outputs = model.generate(
            prompts,
            sampling_params=sampling,
            batch_size=worker_batch_size,
            timeout_s=timeout_s,
        )
    finally:
        if owns_model and model is not None:
            model.close()

    cleaned_batches: List[List[str]] = []
    for response in outputs:
        tags = _json_list_from_text(response)
        seen, unique_tags = set(), []
        for tag in tags:
            tag = re.sub(r"[^\w\-&/ +]", "", tag).strip()
            tag = re.sub(r"\s+", " ", tag)
            lower = tag.lower()
            if not tag or lower in seen:
                continue
            seen.add(lower)
            unique_tags.append(tag)
            if len(unique_tags) >= 6:
                break
        if not unique_tags:
            unique_tags = ["general"]
        cleaned_batches.append(unique_tags)

    # tag_recs (list): each element summarises a source key and looks like
    #   {"key": "<original key string>",
    #    "key_id": <index of the key in the provided iterable>,
    #    "tags": ["<generated tag 1>", "<generated tag 2>", ...]}
    return [
        {"key": key, "key_id": idx, "tags": tags}
        for idx, (key, tags) in enumerate(zip(key_list, cleaned_batches))
    ]


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
    parser = argparse.ArgumentParser(description="Generate tags for keys using a vLLM worker pool.")
    parser.add_argument("--keys-file", type=Path, required=True, help="Text file containing one key per line.")
    parser.add_argument("--key-column", type=str, default=None, help="Text file containing one key per line.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for tag records.")
    parser.add_argument("--model-name", type=str, required=True, help="Generation model name or path.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor_parallel_size per instance.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of worker instances to launch.")
    parser.add_argument(
        "--device-groups",
        type=str,
        help="Explicit GPU mapping, e.g. '0,1;2,3' for two workers with tensor_parallel_size=2.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilisation passed to vLLM.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional maximum model context length.")
    parser.add_argument("--worker-batch-size", type=int, default=8, help="Prompts per batch dispatched to each worker.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional timeout (s) for each worker batch.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top_p value.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens generated per prompt.")
    args = parser.parse_args()

    if not args.keys_file.exists():
        raise FileNotFoundError(f"Keys file not found: {args.keys_file}")

    if args.keys_file.suffix == ".csv":
        if args.key_column == None:
            raise Exception("When key_file is csv file, --key-column must be provided.")
        import pandas as pd
        df = pd.read_csv(args.keys_file)
        keys = df["text"].to_list()

    else:
        try:
            with open(args.keys_file) as f:
                keys = json.load(f)
        except:
            raise Exception("keys-file must be .csv or .json")

    #keys = [line.strip() for line in args.keys_file.read_text().splitlines() if line.strip()]
    if not keys:
        raise ValueError("No keys found in the provided file.")

    device_groups = _parse_device_groups(
        args.device_groups,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
    )

    llm_kwargs = {}
    llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    
    model_settings = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "num_instances": args.num_instances,
        "device_groups": device_groups,
        "llm_kwargs": llm_kwargs,
    }

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    tag_recs = generate_tag(
        keys,
        model_name=args.model_name,
        model_settings=model_settings,
        sampling_params=sampling,
        worker_batch_size=args.worker_batch_size,
        timeout_s=args.timeout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tag_recs, ensure_ascii=False, indent=2))
    print(f"Saved tag records to {args.output}")
