"""Filter flattened query recommendations by query type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

__all__ = ["filter_query_recs"]


def filter_query_recs(
    records: Sequence[dict],
    *,
    query_type: str,
) -> List[dict]:
    """Return only records whose ``query-type`` field matches ``query_type``."""

    target = query_type.strip()
    if not target:
        raise ValueError("query_type must be a non-empty string")
    return [record for record in records if record.get("query-type") == target]


def _load_records(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")


def _write_records(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter query recommendation records by query type.")
    parser.add_argument("--input", type=Path, required=True, help="Path to query_recs.json.")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the filtered JSON.")
    parser.add_argument("--filter", type=str, default="questions", help="Value of the 'query-type' field to keep.")
    args = parser.parse_args()

    records = _load_records(args.input)
    filtered = filter_query_recs(records, query_type=args.filter)
    _write_records(args.output, filtered)

    print(f"Filtered {len(records)} records down to {len(filtered)} with query-type='{args.filter}'. Saved to {args.output}")
