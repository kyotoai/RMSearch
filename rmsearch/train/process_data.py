"""Dataset preparation helpers extracted from the notebook workflow."""

from __future__ import annotations

import json
import argparse
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

from .utils import datasetdict_to_pandas

__all__ = ["process_data"]


def process_data(
    dataset_name: str,
    *,
    output_dir: Path,
    dataset_config: Optional[str] = None,
    split: str = "train",
    n_sample_train: int = 100_000,
    n_sample_test: int = 8_000,
    n_small_sample: int = 10_000,
    random_seed: int = 42,
) -> Path:
    """Download, shuffle, and persist dataset slices for later stages.

    Returns the ``output_dir`` containing ``df.csv`` and ``df_small.csv`` plus the
    HuggingFace binary artefacts when the ``datasets`` package is available.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    if load_dataset is None:
        # Minimal fallback: create dummy CSVs so downstream steps can run in docs/tests.
        sample_count = max(1, min(n_small_sample, n_sample_train, 5))
        sample = pd.DataFrame({"text": ["dummy example sentence" for _ in range(sample_count)]})
        sample.to_csv(output_dir / "df.csv", index=False)
        sample.to_csv(output_dir / "df_small.csv", index=False)
        return output_dir

    features = Features({"index": Value("int64"), "text": Value("string")})

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        features=features if features is not None else None,
    )

    dataset = dataset.shuffle(seed=random_seed)
    train_ds = dataset.select(range(min(n_sample_train, len(dataset))))

    dataset_dict = DatasetDict({"train": train_ds})
    if n_sample_test > 0:
        test_ds = dataset.select(range(min(n_sample_test, len(dataset))))
        dataset_dict["test"] = test_ds

    df_all = datasetdict_to_pandas(dataset_dict)
    dataset_dict.save_to_disk(str(output_dir))
    df_all.to_csv(output_dir / "df.csv", index=False)

    if n_small_sample > 0:
        df_small = df_all.sample(min(n_small_sample, len(df_all)), random_state=random_seed)
        df_small.to_csv(output_dir / "df_small.csv", index=False)

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess training data for RMSearch experiments.")
    parser.add_argument("--dataset-name", type=str, required=True, help="HuggingFace dataset identifier.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where processed data will be stored.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional dataset configuration name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (default: train).")
    parser.add_argument("--n-sample-train", type=int, default=100_000, help="Number of training samples to retain.")
    parser.add_argument("--n-sample-test", type=int, default=8_000, help="Number of samples reserved for evaluation.")
    parser.add_argument("--n-small-sample", type=int, default=10_000, help="Size of df_small.csv for quick iterations.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for shuffling and sampling.")
    args = parser.parse_args()

    out_dir = process_data(
        args.dataset_name,
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        split=args.split,
        n_sample_train=args.n_sample_train,
        n_sample_test=args.n_sample_test,
        n_small_sample=args.n_small_sample,
        random_seed=args.random_seed,
    )
    print("Dataset prepared at:", out_dir)
