"""Dataset preparation helpers extracted from the notebook workflow."""

from __future__ import annotations

import json
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
    demo_dir = Path("./demo_dataset")
    out = process_data("dummy-dataset", output_dir=demo_dir, n_sample_train=5, n_sample_test=0, n_small_sample=3)
    print("Dataset prepared at:", out)
