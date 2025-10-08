"""
Utilities for interacting with a vLLM ``serve`` instance (for example the
``openai/gpt-oss-20b`` checkpoint) using the OpenAI-compatible REST API.

The public surface mirrors ``rmsearch.utils.vllm_generate`` so callers can
switch between local in-process generation and a long-lived HTTP server
without changing import sites.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from requests import Response

try:
    from vllm import SamplingParams  # type: ignore
except Exception:  # pragma: no cover - keep runtime optional
    SamplingParams = None  # type: ignore


_DEFAULT_ENDPOINT = os.environ.get("VLLM_SERVE_ENDPOINT", "http://127.0.0.1:8000/v1")
_DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")


def _batched(seq: Sequence[Tuple[int, str]], size: int) -> Iterable[Sequence[Tuple[int, str]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _sampling_to_payload(
    sampling: Optional["SamplingParams"],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate ``SamplingParams`` into the OpenAI-compatible request payload."""
    payload: Dict[str, Any] = dict(defaults)
    if sampling is None:
        return payload

    def maybe(name: str, value: Any):
        if value is not None:
            payload[name] = value

    attr_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "min_p": "min_p",
        "top_k": "top_k",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "best_of": "best_of",
        "n": "n",
        "use_beam_search": "use_beam_search",
        "beam_width": "beam_width",
        "length_penalty": "length_penalty",
        "repetition_penalty": "repetition_penalty",
        "logprobs": "logprobs",
        "max_tokens": "max_tokens",
    }
    for attr, key in attr_map.items():
        maybe(key, getattr(sampling, attr, None))

    stop_list = getattr(sampling, "stop", None)
    if stop_list:
        if isinstance(stop_list, str):
            payload["stop"] = [stop_list]
        else:
            payload["stop"] = list(stop_list)
    if getattr(sampling, "stop_token_ids", None):
        payload["stop_token_ids"] = list(sampling.stop_token_ids)
    if getattr(sampling, "ignore_eos", None) is not None:
        payload["ignore_eos"] = getattr(sampling, "ignore_eos")
    return payload


@dataclass
class _RequestConfig:
    endpoint_url: str
    api_key: str
    organization: Optional[str]
    request_timeout: float
    max_retries: int
    extra_headers: Dict[str, str]
    healthcheck: bool
    default_sampling: Dict[str, Any]


class LLMServeModel:
    """Thin client that batches requests to a running ``vllm serve`` endpoint."""

    def __init__(self, model_name: str, config: _RequestConfig):
        if SamplingParams is None:
            raise ImportError("vllm is required for SamplingParams; install vllm before use.")

        self.model_name = model_name
        self.config = config
        self._session = requests.Session()
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        if config.organization:
            headers["OpenAI-Organization"] = config.organization
        headers.update(config.extra_headers)
        self._session.headers.update(headers)

        if config.healthcheck:
            self._healthcheck()

    # ---------------------------- setup helpers ----------------------------
    def _healthcheck(self) -> None:
        """Confirm that the model is reachable via ``GET /v1/models/{id}``."""
        url = f"{self.config.endpoint_url.rstrip('/')}/models/{self.model_name}"
        timeout = self.config.request_timeout
        try:
            resp = self._session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to reach vLLM server at {self.config.endpoint_url!r}: {exc}"
            ) from exc

        if resp.status_code == 404:
            raise RuntimeError(
                f"vLLM server reachable but model {self.model_name!r} is not loaded. "
                "Ensure you launched `vllm serve` with this model identifier."
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"vLLM server returned HTTP {resp.status_code}: {resp.text}"
            )

    # ----------------------------- main API -----------------------------
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional["SamplingParams"] = None,
        batch_size: int = 8,
        timeout_s: Optional[float] = None,
    ) -> List[str]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not prompts:
            return []

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=32, temperature=0.7, top_p=0.95)

        indexed = list(enumerate(prompts))
        outputs: List[Optional[str]] = [None] * len(prompts)

        start = time.time()
        deadline = start + timeout_s if timeout_s is not None else None

        for chunk in _batched(indexed, batch_size):
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for vLLM server response.")
            else:
                remaining = None

            chunk_prompts = [p for _, p in chunk]
            chunk_texts = self._request(chunk_prompts, sampling_params, remaining)

            for (idx, _), text in zip(chunk, chunk_texts):
                outputs[idx] = text

        return [text or "" for text in outputs]

    def close(self, kill: bool = False) -> None:  # noqa: ARG002 - match original signature
        self._session.close()

    # --------------------------- internal helpers ---------------------------
    def _request(
        self,
        prompts: List[str],
        sampling_params: "SamplingParams",
        time_budget: Optional[float],
    ) -> List[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompts if len(prompts) > 1 else prompts[0],
        }
        payload.update(_sampling_to_payload(sampling_params, self.config.default_sampling))

        url = f"{self.config.endpoint_url.rstrip('/')}/completions"

        last_exc: Optional[Exception] = None
        remaining_budget = time_budget
        for attempt in range(1, self.config.max_retries + 1):
            timeout = self._resolve_timeout(remaining_budget)
            attempt_start = time.time()
            try:
                response = self._session.post(url, json=payload, timeout=timeout)
                self._raise_for_status(response)
                choice_texts = self._extract_texts(response.json())
                if len(prompts) == 1:
                    return choice_texts[:1]
                return choice_texts
            except Exception as exc:  # broad catch to retry on decoding errors too
                last_exc = exc
                if attempt == self.config.max_retries:
                    break
                if remaining_budget is not None:
                    elapsed = time.time() - attempt_start
                    remaining_budget = max(0.0, remaining_budget - elapsed)
                    if remaining_budget <= 0:
                        break
                time.sleep(min(2 ** (attempt - 1), 10))
        assert last_exc is not None
        raise RuntimeError(f"vLLM request failed after {self.config.max_retries} attempts") from last_exc

    def _raise_for_status(self, response: Response) -> None:
        if response.status_code < 400:
            return
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(f"vLLM server error {response.status_code}: {detail}")

    @staticmethod
    def _extract_texts(payload: Dict[str, Any]) -> List[str]:
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("Unexpected response format: 'choices' missing or not a list.")
        # Some servers may return multiple choices per prompt when n>1. We keep order by index.
        choices_sorted = sorted(
            choices,
            key=lambda item: item.get("index", 0),
        )
        texts = []
        for choice in choices_sorted:
            text = ""
            if isinstance(choice, dict):
                if "text" in choice:
                    text = choice.get("text") or ""
                elif "message" in choice:
                    # ChatCompletion-compatible payload
                    message = choice.get("message") or {}
                    text = message.get("content") or ""
            texts.append(text)
        return texts

    def _resolve_timeout(self, time_budget: Optional[float]) -> float:
        base = self.config.request_timeout
        if time_budget is None:
            return base
        if time_budget <= 0:
            raise TimeoutError("No time left for request.")
        return min(base, time_budget)


def build_llm(
    model_name: str,
    tensor_parallel_size: int,
    num_instances: int,
    device_groups: Optional[List[List[int]]] = None,
    **llm_kwargs: Any,
) -> LLMServeModel:
    """Create an ``LLMServeModel`` with a signature compatible with the local helper."""
    endpoint = llm_kwargs.pop("endpoint_url", llm_kwargs.pop("base_url", _DEFAULT_ENDPOINT))
    api_key = llm_kwargs.pop("api_key", _DEFAULT_API_KEY)
    organization = llm_kwargs.pop("organization", os.environ.get("OPENAI_ORG"))
    request_timeout = float(llm_kwargs.pop("request_timeout", 120.0))
    max_retries = int(llm_kwargs.pop("max_retries", 3))
    extra_headers = llm_kwargs.pop("extra_headers", None) or {}
    healthcheck = bool(llm_kwargs.pop("healthcheck", True))

    sampling_defaults: Dict[str, Any] = {}
    recognised_sampling_keys = {
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "best_of",
        "n",
        "use_beam_search",
        "beam_width",
        "length_penalty",
        "repetition_penalty",
        "logprobs",
        "max_tokens",
        "stop",
        "stop_token_ids",
        "ignore_eos",
    }
    for key in list(llm_kwargs.keys()):
        if key in recognised_sampling_keys:
            sampling_defaults[key] = llm_kwargs.pop(key)

    if llm_kwargs:
        warnings.warn(
            f"Ignoring unsupported serve kwargs: {sorted(llm_kwargs.keys())}",
            RuntimeWarning,
            stacklevel=2,
        )

    cfg = _RequestConfig(
        endpoint_url=endpoint,
        api_key=api_key,
        organization=organization,
        request_timeout=request_timeout,
        max_retries=max_retries,
        extra_headers=dict(extra_headers),
        healthcheck=healthcheck,
        default_sampling=sampling_defaults,
    )
    return LLMServeModel(model_name=model_name, config=cfg)


def generate(model: LLMServeModel, prompts: List[str], **gen_kwargs: Any) -> List[str]:
    return model.generate(prompts, **gen_kwargs)


__all__ = ["LLMServeModel", "build_llm", "generate"]
