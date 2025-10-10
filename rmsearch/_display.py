"""Helpers for coordinating notebook vs. terminal display behaviour."""

from __future__ import annotations

import os
from typing import Callable, Optional


def _coerce_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _detect_notebook() -> bool:
    override = _coerce_bool(os.environ.get("RMSEARCH_NOTEBOOK_UI"))
    if override is not None:
        return override

    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False

    shell = get_ipython()
    if shell is None:
        return False

    name = shell.__class__.__name__
    module = getattr(shell, "__module__", "")

    if name == "ZMQInteractiveShell":
        return True
    if "google.colab" in module.lower():
        return True
    return False


_NOTEBOOK = _detect_notebook()


def in_notebook() -> bool:
    return _NOTEBOOK


def should_use_tqdm() -> bool:
    override = _coerce_bool(os.environ.get("RMSEARCH_USE_TQDM"))
    if override is not None:
        return override
    return not _NOTEBOOK


def should_enable_board() -> bool:
    override = _coerce_bool(os.environ.get("RMSEARCH_ENABLE_WORKER_BOARD"))
    if override is not None:
        return override
    return _NOTEBOOK


def resolve_clear_output() -> Callable[..., None]:
    override = _coerce_bool(os.environ.get("RMSEARCH_ENABLE_CLEAR_OUTPUT"))
    enabled = override if override is not None else should_enable_board()
    if not enabled:
        return lambda *_, **__: None

    try:
        from IPython.display import clear_output as ipy_clear  # type: ignore
    except Exception:
        return lambda *_, **__: None

    def _wrapped(*args, **kwargs):
        try:
            ipy_clear(*args, **kwargs)
        except Exception:
            pass

    return _wrapped


__all__ = [
    "in_notebook",
    "resolve_clear_output",
    "should_enable_board",
    "should_use_tqdm",
]
