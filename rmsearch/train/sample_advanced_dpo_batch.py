"""Advanced sampling of preference-style batches from relevance records for DPO training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

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
        if isinstance(item, dict):
            records.append(item)
    if not records:
        raise ValueError(f"No valid query records found in {path}")
    return records


def _load_filtered_queries(path: Path) -> Dict[int, Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
    lookup: Dict[int, Dict[str, Any]] = {}
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        try:
            lookup[i] = entry
        except (TypeError, ValueError):
            continue
    return lookup


def _load_dataframe(path: Path, *, column: str) -> Dict[int, str]:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not present in {path}")
    values = df[column].dropna().astype(str)
    return dict(enumerate(values.tolist()))


def _sample_key_from_df(df_lookup: Dict[int, str], df_id: int, *, rng: random.Random) -> Optional[Dict[str, Any]]:
    text = df_lookup.get(df_id)
    if text is None:
        return None
    return {"key": text, "key_id": df_id}


def _collect_relevance_candidates(result: Dict[str, Any], *, blocked_ids: Set[int]) -> List[Dict[str, Any]]:
    keys = result.get("keys") or []
    candidates: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    for entry in keys:
        if not isinstance(entry, dict):
            continue
        key_id = entry.get("key_id")
        key_text = entry.get("key")
        if key_id is None or key_text is None:
            continue
        key_id_int = int(key_id)
        if key_id_int in blocked_ids or key_id_int in seen:
            continue
        seen.add(key_id_int)
        candidates.append({"key": str(key_text), "key_id": key_id_int})
    return candidates


def sample_advanced_dpo_batch(
    relevance_records: Sequence[Dict[str, Any]],
    filtered_queries: Optional[Dict[int, Dict[str, Any]]],
    df_lookup: Dict[int, str],
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_sampled_keys: int = 2,
) -> List[Dict[str, Any]]:
    rng = random.Random(random_seed)
    samples: List[Dict[str, Any]] = []

    if not relevance_records:
        pool = list(df_lookup.items())
        needed = n_sampled_keys + 1
        if len(pool) < needed:
            raise ValueError(
                f"Need at least {needed} rows in the dataframe to sample "
                "a correspond key plus the requested sampled keys when relevance records are missing."
            )
        rng.shuffle(pool)
        correspond_id, correspond_key = pool[0]
        sampled_pool = pool[1 : needed]
        samples.append(
            {
                "query": "<random-query>",
                "query_id": -1,
                "correspond_keys": [correspond_key],
                "correspond_key_ids": [correspond_id],
                "sampled_keys": [text for _, text in sampled_pool],
                "sampled_key_ids": [idx for idx, _ in sampled_pool],
            }
        )
        return samples

    for record in relevance_records:
        #query = record.get("query")
        query_id = record.get("query_id")
        query = filtered_queries[query_id]["query"]
        #record["query"] = query
        df_id = record.get("df_id")
        query_type = record.get("query_type")

        if query_id is None:
            continue

        correspond_keys: List[str] = []
        correspond_key_ids: List[int] = []
        correspond_id_set: Set[int] = set()
        if df_id is not None:
            df_key = _sample_key_from_df(df_lookup, int(df_id), rng=rng)
            if df_key is not None:
                correspond_keys.append(df_key["key"])
                correspond_key_ids.append(df_key["key_id"])
                #correspond_keys.append(df_lookup[df_key["key_id"]])
                correspond_id_set.add(df_key["key_id"])

        if not correspond_keys:
            continue

        for i in range(len(record["keys"])):
            key_id = record["keys"][i]["key_id"]
            record["keys"][i]["key"] = df_lookup[key_id]

        candidates = _collect_relevance_candidates(record, blocked_ids=correspond_id_set)

        sample_size = min(n_sampled_keys, len(candidates))
        if sample_size == 0:
            continue
        if sample_size == len(candidates):
            rng.shuffle(candidates)
            sampled = candidates
        else:
            sampled = rng.sample(candidates, k=sample_size)

        entry: Dict[str, Any] = {
            "query": query,
            "query_id": int(query_id),
            "correspond_keys": correspond_keys,
            "correspond_key_ids": correspond_key_ids,
            "sampled_keys": [item["key"] for item in sampled],
            "sampled_key_ids": [item["key_id"] for item in sampled],
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
    parser = argparse.ArgumentParser(
        description="Sample DPO-style query/key pairs with multiple relevance candidates per query."
    )
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
        "--n-sampled-keys",
        type=int,
        default=2,
        help="Number of distinct keys to draw from the relevance records per query (must be >= 1).",
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

    if args.n_sampled_keys < 1:
        raise ValueError("--n-sampled-keys must be a positive integer")

    relevance_records = _load_relevance_records(args.relevance_json)
    filtered_lookup = _load_filtered_queries(args.filtered_queries_json) if args.filtered_queries_json else None
    df_lookup = _load_dataframe(args.source_csv, column=args.source_column)

    samples = sample_advanced_dpo_batch(
        relevance_records,
        filtered_lookup,
        df_lookup,
        random_seed=args.random_seed,
        n_sampled_keys=args.n_sampled_keys,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    print(f"Wrote {len(samples)} sampled query/key pairs to {args.output}")


if __name__ == "__main__":
    main()
