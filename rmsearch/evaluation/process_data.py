"""Utilities to materialise evaluation datasets from HuggingFace into RMSearch artefacts."""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
try:  # Optional dependency; HuggingFace datasets is heavy.
    from datasets import Dataset, IterableDataset, load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional import.
    Dataset = None  # type: ignore[misc,assignment]
    IterableDataset = None  # type: ignore[misc,assignment]
    load_dataset = None  # type: ignore[misc,assignment]
logger = logging.getLogger(__name__)
__all__ = ["process_data"]
_OFFLINE_ENV_VARS = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
_DEFAULT_DATASET = "BeIR/fiqa"
_DEFAULT_QUERY_SPLIT = "queries"
_DEFAULT_KEY_SPLIT = "corpus"
_DEFAULT_PAIR_SPLIT = "qrels"
def _is_offline_mode() -> bool:
    for name in _OFFLINE_ENV_VARS:
        value = os.getenv(name)
        if value and value.lower() not in {"0", "false", "off"}:
            return True
    return False
def _looks_like_connection_issue(exc: Exception) -> bool:
    tokens = (
        "failed to resolve",
        "temporary failure in name resolution",
        "connection timed out",
        "connection aborted",
        "connection reset",
        "name or service not known",
    )
    message = str(exc).lower()
    return any(token in message for token in tokens)
def _write_stub(output_dir: Path) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    logger.warning("Falling back to stub evaluation artefacts in %s", output_dir)
    queries = [f"stub query {idx}" for idx in range(5)]
    keys = [f"stub key sentence {idx}" for idx in range(8)]
    pairs = [(qi, qi % len(keys)) for qi in range(len(queries))]
    return queries, keys, pairs
def _ensure_columns(row: Dict[str, object], columns: Sequence[str], *, source: str) -> None:
    missing = [col for col in columns if col not in row]
    if missing:
        raise KeyError(f"Missing columns {missing} in {source} (have: {list(row.keys())})")
def _materialise_split(
    dataset_name: str,
    *,
    split: str,
    columns: Sequence[str],
    dataset_config: Optional[str],
    max_rows: Optional[int],
) -> List[Dict[str, object]]:
    if load_dataset is None:
        raise RuntimeError("The 'datasets' package is required to download evaluation data.")
    load_kwargs = {"split": split}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    ds = load_dataset(dataset_name, **load_kwargs)
    if IterableDataset is not None and isinstance(ds, IterableDataset):
        iterator: Iterable[Dict[str, object]] = ds
    elif Dataset is not None and isinstance(ds, Dataset):
        iterator = ds  # type: ignore[assignment]
    else:
        raise TypeError(f"Unsupported dataset type for split '{split}': {type(ds)!r}")
    rows: List[Dict[str, object]] = []
    for idx, row in enumerate(iterator):
        _ensure_columns(row, columns, source=f"{dataset_name}:{split}")
        rows.append({col: row[col] for col in columns})
        if max_rows is not None and idx + 1 >= max_rows:
            break
    if not rows:
        raise ValueError(f"Split '{split}' from dataset '{dataset_name}' produced no rows.")
    return rows
def _build_index_and_payload(
    records: Sequence[Dict[str, object]],
    *,
    id_column: str,
    text_column: str,
) -> Tuple[List[str], Dict[str, int]]:
    payload: List[str] = []
    index: Dict[str, int] = {}
    for row in records:
        identifier = str(row[id_column])
        text = row[text_column]
        if text is None:
            continue
        payload.append(str(text))
        index[identifier] = len(payload) - 1
    if not payload:
        raise ValueError(f"No non-empty '{text_column}' entries found while building payload.")
    return payload, index
def _build_pairs(
    records: Sequence[Dict[str, object]],
    *,
    query_id_lookup: Dict[str, int],
    key_id_lookup: Dict[str, int],
    query_column: str,
    key_column: str,
    deduplicate: bool,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for row in records:
        query_raw = str(row[query_column])
        key_raw = str(row[key_column])
        if query_raw not in query_id_lookup or key_raw not in key_id_lookup:
            continue
        pair = (query_id_lookup[query_raw], key_id_lookup[key_raw])
        if deduplicate:
            if pair in seen:
                continue
            seen.add(pair)
        pairs.append(pair)
    if not pairs:
        raise ValueError("Pair split did not match any query/key identifiers.")
    return pairs
def process_data(
    dataset_name: str = _DEFAULT_DATASET,
    *,
    output_dir: Path,
    dataset_config: Optional[str] = None,
    query_split: str = _DEFAULT_QUERY_SPLIT,
    key_split: str = _DEFAULT_KEY_SPLIT,
    pair_split: str = _DEFAULT_PAIR_SPLIT,
    query_id_column: str = "id",
    query_text_column: str = "text",
    key_id_column: str = "id",
    key_text_column: str = "text",
    pair_query_column: str = "query-id",
    pair_key_column: str = "corpus-id",
    max_queries: Optional[int] = None,
    max_keys: Optional[int] = None,
    max_pairs: Optional[int] = None,
    deduplicate_pairs: bool = True,
) -> Path:
    """Download evaluation data and export RMSearch-compatible artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if _is_offline_mode() or load_dataset is None:
        queries, keys, pairs = _write_stub(output_dir)
    else:
        try:
            query_records = _materialise_split(
                dataset_name,
                split=query_split,
                columns=[query_id_column, query_text_column],
                dataset_config=dataset_config,
                max_rows=max_queries,
            )
            key_records = _materialise_split(
                dataset_name,
                split=key_split,
                columns=[key_id_column, key_text_column],
                dataset_config=dataset_config,
                max_rows=max_keys,
            )
            pair_records = _materialise_split(
                dataset_name,
                split=pair_split,
                columns=[pair_query_column, pair_key_column],
                dataset_config=dataset_config,
                max_rows=max_pairs,
            )
        except Exception as exc:
            if isinstance(exc, (RuntimeError, ValueError)):
                raise
            if _looks_like_connection_issue(exc):
                queries, keys, pairs = _write_stub(output_dir)
            else:
                raise
        else:
            queries, query_lookup = _build_index_and_payload(
                query_records,
                id_column=query_id_column,
                text_column=query_text_column,
            )
            keys, key_lookup = _build_index_and_payload(
                key_records,
                id_column=key_id_column,
                text_column=key_text_column,
            )
            pairs = _build_pairs(
                pair_records,
                query_id_lookup=query_lookup,
                key_id_lookup=key_lookup,
                query_column=pair_query_column,
                key_column=pair_key_column,
                deduplicate=deduplicate_pairs,
            )
    query_path = output_dir / "query.json"
    key_path = output_dir / "key.json"
    pair_path = output_dir / "pair.csv"
    query_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(keys, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pair_df = pd.DataFrame(pairs, columns=["query_id", "key_id"])
    pair_df.to_csv(pair_path, index=False)
    logger.info(
        "Materialised %d queries, %d keys, %d pairs into %s",
        len(queries),
        len(keys),
        len(pairs),
        output_dir,
    )
    return output_dir
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download evaluation data and produce RMSearch artefacts.")
    parser.add_argument("--dataset-name", type=str, default=_DEFAULT_DATASET, help="HuggingFace dataset identifier.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional configuration name.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated files.")
    parser.add_argument("--query-split", type=str, default=_DEFAULT_QUERY_SPLIT, help="Split containing query rows.")
    parser.add_argument("--key-split", type=str, default=_DEFAULT_KEY_SPLIT, help="Split containing key rows.")
    parser.add_argument("--pair-split", type=str, default=_DEFAULT_PAIR_SPLIT, help="Split containing query-key relations.")
    parser.add_argument("--query-id-column", type=str, default="id", help="Identifier column for query rows.")
    parser.add_argument("--query-text-column", type=str, default="text", help="Text column for query rows.")
    parser.add_argument("--key-id-column", type=str, default="id", help="Identifier column for key rows.")
    parser.add_argument("--key-text-column", type=str, default="text", help="Text column for key rows.")
    parser.add_argument("--pair-query-column", type=str, default="query-id", help="Column pointing to query identifiers.")
    parser.add_argument("--pair-key-column", type=str, default="corpus-id", help="Column pointing to key identifiers.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional maximum number of queries to keep.")
    parser.add_argument("--max-keys", type=int, default=None, help="Optional maximum number of keys to keep.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional maximum number of query-key pairs to keep.")
    parser.add_argument("--no-deduplicate", action="store_true", help="Keep duplicate query-key pairs if provided.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    process_data(
        args.dataset_name,
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        query_split=args.query_split,
        key_split=args.key_split,
        pair_split=args.pair_split,
        query_id_column=args.query_id_column,
        query_text_column=args.query_text_column,
        key_id_column=args.key_id_column,
        key_text_column=args.key_text_column,
        pair_query_column=args.pair_query_column,
        pair_key_column=args.pair_key_column,
        max_queries=args.max_queries,
        max_keys=args.max_keys,
        max_pairs=args.max_pairs,
        deduplicate_pairs=not args.no_deduplicate,
    )
if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    main()