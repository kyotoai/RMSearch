"""Dataset preparation helpers extracted from the notebook workflow."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:  # Optional dependency: HuggingFace datasets is heavy
    from datasets import DatasetDict, Features, Value, load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional import
    DatasetDict = None  # type: ignore
    Features = None  # type: ignore
    Value = None  # type: ignore
    load_dataset = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from requests import exceptions as requests_exceptions
except Exception:  # pragma: no cover - requests may be absent in minimal installs
    requests_exceptions = None  # type: ignore[assignment]

from .utils import datasetdict_to_pandas

logger = logging.getLogger(__name__)

_MAX_STUB_ROWS = 50
_DEFAULT_DF_SMALL_SIZE = 10_000
_OFFLINE_ENV_VARS = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")


def _is_offline_mode() -> bool:
    """Return True when common HuggingFace offline env vars are switched on."""

    for name in _OFFLINE_ENV_VARS:
        value = os.getenv(name)
        if value and value.lower() not in {"0", "false", "off"}:
            return True
    return False


def _write_stub_csvs(
    output_dir: Path,
    *,
    n_sample: Optional[int],
    random_seed: int,
) -> Path:
    """Persist deterministic stub CSVs so downstream steps stay usable offline."""

    row_cap = _MAX_STUB_ROWS if n_sample is None else min(n_sample, _MAX_STUB_ROWS)
    row_count = max(1, row_cap)

    rows = [{"index": idx, "text": f"offline stub sentence {idx}", "split": "train"} for idx in range(row_count)]

    df_all = pd.DataFrame(rows)
    df_all.to_csv(output_dir / "df.csv", index=False)

    small_size = min(_DEFAULT_DF_SMALL_SIZE, len(df_all))
    if small_size > 0:
        df_small = df_all.sample(small_size, random_state=random_seed).reset_index(drop=True)
        df_small.to_csv(output_dir / "df_small.csv", index=False)

    return output_dir


def _looks_like_connection_issue(exc: Exception) -> bool:
    message = str(exc).lower()
    if requests_exceptions is not None and isinstance(exc, requests_exceptions.ConnectionError):
        return True
    connection_tokens = (
        "name or service not known",
        "temporary failure in name resolution",
        "failed to resolve",
        "max retries exceeded",
        "connection aborted",
        "connection reset",
        "connection timed out",
    )
    return any(token in message for token in connection_tokens)

__all__ = ["process_data"]


def process_data(
    dataset_name: str,
    *,
    output_dir: Path,
    dataset_config: Optional[str] = None,
    split: str = "train",
    n_sample: Optional[int] = None,
    random_seed: int = 42,
) -> Path:
    """Download, shuffle, and persist dataset slices for later stages.

    Returns the ``output_dir`` containing ``df.csv`` and ``df_small.csv`` plus the
    HuggingFace binary artefacts when the ``datasets`` package is available. If
    ``n_sample`` is provided only that many rows are stored to ease disk pressure.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    offline_mode = _is_offline_mode()

    if load_dataset is None or offline_mode:
        reason = "datasets library is unavailable" if load_dataset is None else "HuggingFace offline mode detected"
        logger.warning("%s; writing stub CSVs to %s", reason, output_dir)
        return _write_stub_csvs(
            output_dir,
            n_sample=n_sample,
            random_seed=random_seed,
        )

    features = Features({"index": Value("int64"), "text": Value("string")})

    try:
        dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split=split,
            features=features if features is not None else None,
        )
    except Exception as exc:  # pragma: no cover - exercised in offline environments
        if _looks_like_connection_issue(exc):
            logger.warning(
                "Could not download dataset '%s' (%s); writing stub CSVs instead.",
                dataset_name,
                exc.__class__.__name__,
            )
            return _write_stub_csvs(
                output_dir,
                n_sample=n_sample,
                random_seed=random_seed,
            )
        raise

    dataset = dataset.shuffle(seed=random_seed)

    if n_sample is not None and n_sample > 0:
        sample_size = min(n_sample, len(dataset))
        train_ds = dataset.select(range(sample_size))
    else:
        train_ds = dataset

    dataset_dict = DatasetDict({"train": train_ds})

    df_all = datasetdict_to_pandas(dataset_dict)
    dataset_dict.save_to_disk(str(output_dir))
    df_all.to_csv(output_dir / "df.csv", index=False)

    if len(df_all):
        df_small = df_all.sample(min(_DEFAULT_DF_SMALL_SIZE, len(df_all)), random_state=random_seed)
        df_small.to_csv(output_dir / "df_small.csv", index=False)

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess training data for RMSearch experiments.")
    parser.add_argument("--dataset-name", type=str, required=True, help="HuggingFace dataset identifier.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where processed data will be stored.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional dataset configuration name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (default: train).")
    parser.add_argument(
        "--n-sample",
        type=int,
        default=None,
        help="Optional number of rows to persist when disk space is limited.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for shuffling and sampling.")
    args = parser.parse_args()

    out_dir = process_data(
        args.dataset_name,
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        split=args.split,
        n_sample=args.n_sample,
        random_seed=args.random_seed,
    )
    print("Dataset prepared at:", out_dir)
