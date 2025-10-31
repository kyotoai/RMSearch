"""Utility submodules powering RMSearch's vLLM integrations.

This package re-exports the worker pool helpers (e.g. ``vllm_generate``)
and falls back to the legacy ``rmsearch/utils.py`` implementations for
``convert_model`` / ``revert_model`` so existing imports keep working.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, List

from . import vllm_embed, vllm_generate, vllm_generate_gptoss, vllm_reward, vllm_serve_generate

__all__: List[str] = [
    "vllm_generate",
    "vllm_generate_gptoss",
    "vllm_serve_generate",
    "vllm_reward",
    "vllm_embed",
    "convert_model",
    "revert_model",
]

_LEGACY_MODULE_NAME = "rmsearch._legacy_utils"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "utils.py"


def _load_legacy_module() -> ModuleType:
    module = sys.modules.get(_LEGACY_MODULE_NAME)
    if module is not None:
        return module

    if not _LEGACY_PATH.exists():
        raise ModuleNotFoundError(f"Legacy utils module not found at {_LEGACY_PATH}")

    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load legacy utils module from {_LEGACY_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def __getattr__(name: str) -> Any:
    if name in {"convert_model", "revert_model"}:
        legacy = _load_legacy_module()
        try:
            return getattr(legacy, name)
        except AttributeError as exc:  # pragma: no cover - legacy module contract
            raise AttributeError(name) from exc
    raise AttributeError(name)


def __dir__() -> Iterable[str]:  # pragma: no cover - convenience only
    return sorted(set(__all__))

