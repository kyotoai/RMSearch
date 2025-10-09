"""Sample preference-style batches from relevance records for DPO training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

DEFAULT_RANDOM_SEED = 42


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_relevance_records(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
    records: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "query" in item:
            records.append(item)
    if not records:
        raise ValueError(f"No valid query records found in {path}")
    return records


def _load_filtered_queries(path: Path) -> Dict[int, Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
    lookup: Dict[int, Dict[str, Any]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        df_id = entry.get("df_id")
        if df_id is None:
            continue
        try:
            lookup[int(df_id)] = entry
        except (TypeError, ValueError):
            continue
    return lookup


def _load_dataframe(path: Path, *, column: str) -> Dict[int, str]:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not present in {path}")
    values = df[column].dropna().astype(str)
    return dict(enumerate(values.tolist()))


def _sample_from_list(items: Sequence[Any], *, rng: random.Random) -> Any:
    if not items:
        raise ValueError("Cannot sample from an empty sequence")
    return rng.choice(items)


def _sample_key_from_df(df_lookup: Dict[int, str], df_id: int, *, rng: random.Random) -> Optional[Dict[str, Any]]:
    text = df_lookup.get(df_id)
    if text is None:
        return None
    return {"key": text, "key_id": df_id}


def _sample_key_from_relevance(result: Dict[str, Any], *, rng: random.Random) -> Optional[Dict[str, Any]]:
    keys = result.get("keys") or []
    if not keys:
        return None
    sampled = _sample_from_list(keys, rng=rng)
    key_id = sampled.get("key_id")
    key_text = sampled.get("key")
    if key_id is None or key_text is None:
        return None
    return {"key": key_text, "key_id": int(key_id)}


def sample_dpo_batch(
    relevance_records: Sequence[Dict[str, Any]],
    filtered_queries: Optional[Dict[int, Dict[str, Any]]],
    df_lookup: Dict[int, str],
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> List[Dict[str, Any]]:
    rng = random.Random(random_seed)
    samples: List[Dict[str, Any]] = []

    if not relevance_records:
        pool = list(df_lookup.items())
        if len(pool) < 2:
            raise ValueError("Need at least two df entries to sample without relevance records")
        sampled_pairs = rng.sample(pool, k=min(2, len(pool)))
        keys = [text for _, text in sampled_pairs]
        key_ids = [idx for idx, _ in sampled_pairs]
        samples.append({"query": "<random-query>", "query_id": -1, "keys": keys, "key_ids": key_ids})
        return samples

    for record in relevance_records:
        query = record.get("query")
        query_id = record.get("query_id")
        df_id = record.get("df_id")
        query_type = record.get("query_type")

        if query is None or query_id is None:
            continue

        top_key = _sample_key_from_relevance(record, rng=rng)
        df_key = None
        if df_id is not None:
            df_key = _sample_key_from_df(df_lookup, int(df_id), rng=rng)

        if top_key is None and df_key is None:
            continue

        chosen_keys: List[Dict[str, Any]] = []
        key_ids: List[int] = []
        for candidate in (top_key, df_key):
            if candidate is None:
                continue
            chosen_keys.append(candidate["key"])
            key_ids.append(candidate["key_id"])

        entry: Dict[str, Any] = {
            "query": query,
            "query_id": int(query_id),
            "keys": chosen_keys,
            "key_ids": key_ids,
        }

        if query_type:
            entry["query-type"] = query_type
        elif filtered_queries and df_id is not None:
            meta = filtered_queries.get(int(df_id))
            if meta and meta.get("query-type"):
                entry["query-type"] = meta["query-type"]

        samples.append(entry)

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample DPO-style query/key pairs from relevance records.")
    parser.add_argument(
        "--relevance-json",
        type=Path,
        help="Optional relevance records JSON (RM or embedding). If omitted, samples are drawn from the source CSV.",
    )
    parser.add_argument(
        "--filtered-queries-json",
        type=Path,
        help="Optional filtered_query_recs.json to retrieve df_id/query-type metadata.",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        required=True,
        help="Original dataframe CSV (e.g. df.csv) containing the source sentences.",
    )
    parser.add_argument(
        "--source-column",
        type=str,
        default="text",
        help="Column name inside --source-csv that matches df_id rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/smollm-corpus/sampled_query_key_set.json"),
        help="Where to write the sampled query/key pairs JSON.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    relevance_records = _load_relevance_records(args.relevance_json)
    filtered_lookup = _load_filtered_queries(args.filtered_queries_json) if args.filtered_queries_json else None
    df_lookup = _load_dataframe(args.source_csv, column=args.source_column)

    samples = sample_dpo_batch(
        relevance_records,
        filtered_lookup,
        df_lookup,
        random_seed=args.random_seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    print(f"Wrote {len(samples)} sampled query/key pairs to {args.output}")


if __name__ == "__main__":
    main()
