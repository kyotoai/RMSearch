"""Utilities for generating candidate tags with vLLM worker models.

This module extracts the tag-generation portion of
``examples/train_en.ipynb`` so it can be reused from library code.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from vllm import SamplingParams

from rmsearch.utils.vllm_generate import LLMWorkerModel, build_llm

__all__ = ["generate_tag", "build_model_from_settings"]


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

    # tag_recs structure -> [{"key": str, "key_id": int, "tags": [str, ...]}]
    return [
        {"key": key, "key_id": idx, "tags": tags}
        for idx, (key, tags) in enumerate(zip(key_list, cleaned_batches))
    ]


if __name__ == "__main__":
    class DummyModel:
        """Minimal stand-in that mimics the ``LLMWorkerModel`` API."""

        def __init__(self, responses: List[str]):
            self._responses = responses

        def generate(self, prompts, sampling_params=None, batch_size=None, timeout_s=None):
            del prompts, sampling_params, batch_size, timeout_s
            return list(self._responses)

        def close(self):
            pass

    dummy_outputs = [
        '["Graph Theory", "Retrieval", "Knowledge Base"]',
        '["Reinforcement Learning", "Reward Model"]',
    ]
    demo_model = DummyModel(dummy_outputs)
    demo_keys = ["Graph-based retrieval augmentation", "Optimising reward models"]
    records = generate_tag(demo_keys, model_name="dummy", model=demo_model)
    for rec in records:
        print(rec)
