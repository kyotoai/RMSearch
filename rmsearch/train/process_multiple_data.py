from __future__ import annotations
"""Dataset preparation helpers (multi-dataset edition)."""

# set up to download hugging face datasets
#  pip install -U "huggingface_hub[cli]"
#  pip install hf_transfer
# hf version    # sanity check
# # Dowload datasets e.g. ag_news imdb yelp_polarity
# for d in ag_news imdb yelp_polarity; do
#   hf download --repo-type dataset "$d" --local-dir "./hf_datasets/$d" --local-dir-use-symlinks False
# done
# ALWAYS DO: pip install -e RMSearch/.
# python process_multiple_data.py  \
#  --datasets-names HuggingFaceTB/smollm-corpus,ag_news,imdb  \
#  --dataset-configs cosmopedia-v2,,  \
#  --partitions 0.6,0.3,0.1   \
#  --n-sample 2000  \
#  --stream   \
#  --output-dir ./data_test/mix_1k

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:  # Optional dependency: HuggingFace datasets is heavy
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional import
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore
    load_dataset = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from requests import exceptions as requests_exceptions
except Exception:  # pragma: no cover
    requests_exceptions = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MAX_STUB_ROWS = 50
_DEFAULT_DF_SMALL_SIZE = 10_000
_OFFLINE_ENV_VARS = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
_DEFAULT_STREAM_BUFFER_SIZE = 10_000


def _is_offline_mode() -> bool:
    """Return True when common HuggingFace offline env vars are switched on."""
    for name in _OFFLINE_ENV_VARS:
        value = os.getenv(name)
        if value and value.lower() not in {"0", "false", "off"}:
            return True
    return False


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


@dataclass
class DatasetSpec:
    name: str
    config: Optional[str]
    split: str


@dataclass
class LoadedInfo:
    name: str
    config: Optional[str]
    split: str
    requested_rows: Optional[int]  # n_i requested (None => full)
    actual_rows: int               # rows materialized
    streaming: bool                # whether streaming path used
    source_column_value: str       # value used in df['source']


@dataclass
class Manifest:
    created_at_utc: str
    offline_mode: bool
    random_seed: int
    n_sample_total_requested: Optional[int]
    n_sample_total_actual: int
    partitions_requested: Optional[List[float]]
    partitions_effective: Optional[List[float]]
    streaming: bool
    datasets: List[LoadedInfo]


def _write_stub_csvs_multi(
    output_dir: Path,
    *,
    dataset_specs: Sequence[DatasetSpec],
    n_sample_total: Optional[int],
    random_seed: int,
) -> Path:
    """Persist deterministic stub CSVs (multi-source) for offline use."""
    row_cap = _MAX_STUB_ROWS if n_sample_total is None else min(n_sample_total, _MAX_STUB_ROWS)
    row_count = max(1, row_cap)

    rows = []
    # Spread rows evenly across sources just to keep things deterministic
    k = max(1, len(dataset_specs))
    for i in range(row_count):
        spec = dataset_specs[i % k]
        source = _source_tag(spec.name, spec.config, spec.split)
        rows.append({"index": i, "text": f"offline stub sentence {i}", "split": "train", "source": source})

    df_all = pd.DataFrame(rows)
    df_all.to_csv(output_dir / "df.csv", index=False)

    small_size = min(_DEFAULT_DF_SMALL_SIZE, len(df_all))
    if small_size > 0:
        df_small = df_all.sample(small_size, random_state=random_seed).reset_index(drop=True)
        df_small.to_csv(output_dir / "df_small.csv", index=False)

    # minimal manifest
    manifest = Manifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        offline_mode=True,
        random_seed=random_seed,
        n_sample_total_requested=n_sample_total,
        n_sample_total_actual=len(df_all),
        partitions_requested=None,
        partitions_effective=None,
        streaming=False,
        datasets=[
            LoadedInfo(
                name=sp.name,
                config=sp.config,
                split=sp.split,
                requested_rows=None if n_sample_total is None else None,  # not meaningful offline
                actual_rows=len(df_all) // k,
                streaming=False,
                source_column_value=_source_tag(sp.name, sp.config, sp.split),
            )
            for sp in dataset_specs
        ],
    )
    (output_dir / "loaded_info.json").write_text(json.dumps(asdict(manifest), indent=2))
    return output_dir


def _parse_list_arg(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or None


def _normalize_partitions(k: int, parts: Optional[Sequence[float]]) -> List[float]:
    """Validate and normalize partitions to length k; default = uniform."""
    if parts is None:
        return [1.0 / k] * k
    if len(parts) != k:
        raise ValueError(f"--partitions must have length {k}, got {len(parts)}")
    for p in parts:
        if not (p > 0.0 and p < 1.0):
            raise ValueError("--partitions values must satisfy 0 < p_i < 1")
    s = sum(parts)
    if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-9):
        # normalize but warn
        logger.warning("Sum(partitions)=%.6f != 1; normalizing.", s)
        parts = [p / s for p in parts]
    return list(parts)


def _allocate_counts(n_total: int, parts: Sequence[float]) -> List[int]:
    """Round n_total * parts to integers that sum exactly to n_total."""
    raw = [p * n_total for p in parts]
    floors = [int(math.floor(x)) for x in raw]
    remainder = n_total - sum(floors)
    # Distribute the remainder by largest fractional parts
    frac_order = sorted(range(len(raw)), key=lambda i: (raw[i] - floors[i]), reverse=True)
    for i in range(remainder):
        floors[frac_order[i]] += 1
    return floors


def _source_tag(name: str, config: Optional[str], split: str) -> str:
    return f"{name}" + (f":{config}" if config else "") + f"/{split}"


def _normalize_df_for_hf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize df so Dataset.from_pandas/save_to_disk won't choke:
    - Ensure column names are strings
    - Convert datetime-like values to ISO strings
    - Convert lists/dicts to JSON strings
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    # Convert pandas datetime64 columns to ISO strings (UTC if tz-aware, naive as-is)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            s = df[col]
            try:
                # Try to get UTC ISO; if naive, just isoformat without timezone
                if getattr(s.dt, "tz", None) is not None:
                    df[col] = s.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                else:
                    df[col] = s.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
            except Exception:
                # Fallback: map elementwise
                df[col] = s.apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

    # Object columns that may contain datetime.date/datetime objs, lists, dicts
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(50)
            if any(hasattr(x, "isoformat") for x in sample):
                df[col] = df[col].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
            # lists/dicts → JSON strings
            sample = df[col].dropna().head(50)  # re-sample after possible datetime fix
            if any(isinstance(x, (list, dict)) for x in sample):
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

    return df



def _load_one_dataset(
    spec: DatasetSpec,
    *,
    n_rows: Optional[int], # how many rows to keep (None = keep all)
    random_seed: int,
    stream: bool,          # use HF streaming API or not
) -> Tuple[pd.DataFrame, LoadedInfo]:
    """Load a single dataset into a DataFrame, sampling n_rows if provided."""
    if load_dataset is None:
        raise RuntimeError("The 'datasets' library is not available in this environment.")

    load_kwargs = {"split": spec.split, "streaming": stream}
    if spec.config is not None:
        load_kwargs["name"] = spec.config

    try:
        ds = load_dataset(spec.name, **load_kwargs)
    except Exception as exc:
        if _looks_like_connection_issue(exc):
            raise RuntimeError(
                f"Connection error while loading '{spec.name}': {exc.__class__.__name__}"
            ) from exc
        raise

    source_value = _source_tag(spec.name, spec.config, spec.split)

    if stream:
        if Dataset is None:
            raise RuntimeError("Streaming mode requires the HuggingFace datasets package.")
        buffer_size = _DEFAULT_STREAM_BUFFER_SIZE
        if n_rows is not None and n_rows > 0:
            buffer_size = max(buffer_size, n_rows)
        stream_ds = ds.shuffle(seed=random_seed, buffer_size=buffer_size)

        iterable = stream_ds if (n_rows is None or n_rows <= 0) else stream_ds.take(n_rows)
        rows = [dict(sample) for sample in iterable]
        df = pd.DataFrame(rows)
    else:
        ds = ds.shuffle(seed=random_seed)
        if n_rows is not None and n_rows > 0:
            take = min(n_rows, len(ds))
            ds = ds.select(range(take))
        # materialize to pandas
        try:
            df = ds.to_pandas()
        except Exception:
            # fallback: manual copy
            df = pd.DataFrame([dict(ex) for ex in ds])

    if df.empty:
        actual = 0
    else:
        df = df.copy()
        df["source"] = source_value
        actual = len(df)

    info = LoadedInfo(
        name=spec.name,
        config=spec.config,
        split=spec.split,
        requested_rows=n_rows,
        actual_rows=actual,
        streaming=stream,
        source_column_value=source_value,
    )
    return df, info


def process_data_multi(
    dataset_names: Sequence[str],
    *,
    output_dir: Path,
    dataset_configs: Optional[Sequence[Optional[str]]] = None,
    split: str = "train",
    n_sample_total: Optional[int] = None,
    partitions: Optional[Sequence[float]] = None,
    random_seed: int = 42,
    stream: bool = False,
) -> Path:
    """
    Download, shuffle, sample, and persist concatenated slices from multiple datasets.

    - Adds a 'source' column to identify origin dataset/config/split.
    - Writes: df.csv, df_small.csv, dataset_dict.json
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    offline_mode = _is_offline_mode()

    # Build aligned specs
    k = len(dataset_names)
    if k == 0:
        raise ValueError("At least one dataset name must be provided.")

    if dataset_configs is None:
        dataset_configs = [None] * k
    elif len(dataset_configs) == 1 and k > 1:
        dataset_configs = [dataset_configs[0]] * k
    elif len(dataset_configs) != k:
        raise ValueError(f"--dataset-configs must have length 1 or {k}, got {len(dataset_configs)}")

    specs = [DatasetSpec(nm, cfg, split) for nm, cfg in zip(dataset_names, dataset_configs)]

    # Offline or datasets unavailable => stubs
    if load_dataset is None or offline_mode:
        reason = "datasets library is unavailable" if load_dataset is None else "HuggingFace offline mode detected"
        logger.warning("%s; writing stub CSVs to %s", reason, output_dir)
        return _write_stub_csvs_multi(
            output_dir,
            dataset_specs=specs,
            n_sample_total=n_sample_total,
            random_seed=random_seed,
        )

    # Determine per-dataset requested rows
    if n_sample_total is None:
        # full concat: no partitions needed
        n_requested = [None] * k
        eff_parts = None
        parts_req = None
    else:
        parts = _normalize_partitions(k, partitions)
        counts = _allocate_counts(n_sample_total, parts)
        n_requested = counts
        eff_parts = [c / n_sample_total if n_sample_total > 0 else 0.0 for c in counts]
        parts_req = parts

    # Load each dataset
    dfs: List[pd.DataFrame] = []
    infos: List[LoadedInfo] = []

    for spec, n_i in zip(specs, n_requested):
        try:
            df_i, info_i = _load_one_dataset(spec, n_rows=n_i, random_seed=random_seed, stream=stream)
        except Exception as exc:  # pragma: no cover (primarily offline/connection issues)
            if _looks_like_connection_issue(exc):
                logger.warning(
                    "Could not download dataset '%s' (%s); writing stub CSVs instead.",
                    spec.name,
                    exc.__class__.__name__,
                )
                return _write_stub_csvs_multi(
                    output_dir,
                    dataset_specs=specs,
                    n_sample_total=n_sample_total,
                    random_seed=random_seed,
                )
            raise
        dfs.append(df_i)
        infos.append(info_i)

    # Concatenate
    df_all = pd.concat([df for df in dfs if not df.empty], ignore_index=True) if dfs else pd.DataFrame()
    n_total_actual = len(df_all)

    # Persist CSVs
    df_all.to_csv(output_dir / "df.csv", index=False)
    if n_total_actual:
        df_small = df_all.sample(min(_DEFAULT_DF_SMALL_SIZE, n_total_actual), random_state=random_seed)
        df_small.to_csv(output_dir / "df_small.csv", index=False)

    # Manifest JSON
        # Write the custom manifest as loaded_info.json (provenance + counts)
    manifest = Manifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        offline_mode=False,
        random_seed=random_seed,
        n_sample_total_requested=n_sample_total,
        n_sample_total_actual=n_total_actual,
        partitions_requested=parts_req,
        partitions_effective=eff_parts,
        streaming=stream,
        datasets=infos,
    )
    (output_dir / "loaded_info.json").write_text(json.dumps(asdict(manifest), indent=2))

    # Build and persist a true HF DatasetDict so output_dir contains dataset_dict.json from HF
    try:
        if Dataset is None or DatasetDict is None:
            raise RuntimeError("The 'datasets' library is unavailable for saving HF DatasetDict.")

        df_norm = _normalize_df_for_hf(df_all)

        # Important: preserve_index=False to avoid an "_index" column
        hf_ds = Dataset.from_pandas(df_norm, preserve_index=False)
        hf_dict = DatasetDict({"train": hf_ds})

        # Save directly into output_dir so HF writes dataset_dict.json there
        # hf_dict.save_to_disk(str(output_dir))

        # At this point, output_dir will contain HF files including:
        # - dataset_dict.json    (HF authoritative metadata)
        # - dataset_info.json, state.json, Arrow data, etc.
    except Exception as e:
        logger.warning("Could not persist HF DatasetDict artifacts: %s", e)


    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Download and preprocess training data from multiple HF datasets."
    )
    parser.add_argument(
        "--datasets-names",
        type=str,
        required=True,
        help="Comma-separated list of HuggingFace dataset identifiers, e.g. 'imdb,ag_news'.",
    )
    parser.add_argument(
        "--dataset-configs",
        type=str,
        default=None,
        help="Comma-separated list of configs aligned 1:1 with datasets-names (or a single value applied to all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where processed data will be stored.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load for all datasets (default: train).",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=None,
        help="Total number of rows to draw across all datasets. If omitted, loads full concatenation.",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default=None,
        help="Comma-separated fractions (0<p_i<1) summing to ~1; allocates n_i = round(p_i * n_sample) per dataset.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for shuffling and sampling.")
    parser.add_argument("--stream", action="store_true", help="Load datasets through the streaming API.")
    args = parser.parse_args()

    names = _parse_list_arg(args.datasets_names) or []

    # allow empty placeholders in --dataset-configs, e.g. "cosmopedia-v2,,"
    if args.dataset_configs is None:
        configs = None
    else:
        tokens = [t.strip() for t in args.dataset_configs.split(",")]
        # convert "" -> None so length stays aligned with datasets
        configs = [tok if tok != "" else None for tok in tokens]

    # configs_list_raw = _parse_list_arg(args.dataset_configs)
    # configs = None if configs_list_raw is None else configs_list_raw

    parts_raw = _parse_list_arg(args.partitions)
    parts = None if parts_raw is None else [float(x) for x in parts_raw]

    out_dir = process_data_multi(
        names,
        output_dir=args.output_dir,
        dataset_configs=configs,
        split=args.split,
        n_sample_total=args.n_sample,
        partitions=parts,
        random_seed=args.random_seed,
        stream=args.stream,
    )
    print("Datasets prepared at:", out_dir)
