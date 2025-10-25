#!/usr/bin/env python3
"""
beir_to_pairs.py

Creates:
  <out>/<dataset>/query.csv          (id, original_query_id, text)
  <out>/<dataset>/key.csv            (id, original_key_id, text)
  <out>/<dataset>/pair.csv           (query_id, original_query_id, key_id, original_key_id)

Usage:
  pip install beir pandas tqdm
  python beir_to_pairs.py --outdir ./beir_out --split test scifact nq
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

import pandas as pd
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def _find_split_by_qrels(ds_folder: Path, preferred: str) -> str:
    """
    Determine available split by looking for qrels/<split>.tsv.
    Falls back in order: preferred -> test -> dev -> train -> first *.tsv found.
    """
    qrels_dir = ds_folder / "qrels"
    if not qrels_dir.exists():
        raise FileNotFoundError(f"No 'qrels' directory under {ds_folder}")

    candidates = [preferred, "test", "dev", "train"]
    for c in candidates:
        if (qrels_dir / f"{c}.tsv").exists():
            return c

    any_qrels = sorted(qrels_dir.glob("*.tsv"))
    if any_qrels:
        return any_qrels[0].stem

    raise FileNotFoundError(f"No qrels/*.tsv found under {qrels_dir}")


def _main_content_from_doc(doc: Dict[str, Any]) -> str:
    """
    'Main content' for keys: prefer doc['text']; if missing/empty, fall back to doc['title'].
    """
    text = (doc.get("text") or "").strip()
    if text:
        return text
    return (doc.get("title") or "").strip()


def convert_one(dataset: str, outdir: Path, preferred_split: str):
    print(f"==> Downloading '{dataset}' ...")
    data_root = outdir / "_raw"
    data_root.mkdir(parents=True, exist_ok=True)

    # Download BEIR zip and unzip
    url = f"{BASE_URL}/{dataset}.zip"
    data_path = util.download_and_unzip(url, str(data_root))

    # Unzips to <data_root>/<dataset>
    ds_folder = Path(data_path) / dataset
    if not ds_folder.exists():
        ds_folder = Path(data_path)

    # Determine split from qrels/*.tsv
    split = _find_split_by_qrels(ds_folder, preferred_split)
    print(f"   Using split: {split}")

    # Load structures
    loader = GenericDataLoader(data_folder=str(ds_folder))
    corpus, queries, qrels = loader.load(split=split)

    # Sort for deterministic IDs
    sorted_queries = sorted(queries.items(), key=lambda kv: kv[0])
    sorted_docs = sorted(corpus.items(), key=lambda kv: kv[0])

    # Build query.csv rows
    query_rows = []
    for _, (qid, qtext) in enumerate(tqdm(sorted_queries, desc=f"[{dataset}] queries"), start=0):
        query_rows.append({
            # id filled later by DataFrame index or explicit range
            "original_query_id": qid,
            "text": (qtext or "").strip(),
        })
    query_df = pd.DataFrame(query_rows, columns=["original_query_id", "text"])
    query_df.insert(0, "id", range(len(query_df)))  # 0-based index as 'id'

    # Map original_query_id -> new id
    qid2new = {row.original_query_id: int(row.id) for row in query_df.itertuples(index=False)}

    # Build key.csv rows (docs)
    key_rows = []
    for _, (doc_id, doc) in enumerate(tqdm(sorted_docs, desc=f"[{dataset}] docs"), start=0):
        key_rows.append({
            "original_key_id": doc_id,
            "text": _main_content_from_doc(doc),
        })
    key_df = pd.DataFrame(key_rows, columns=["original_key_id", "text"])
    key_df.insert(0, "id", range(len(key_df)))  # 0-based index as 'id'

    # Map original_key_id -> new id
    did2new = {row.original_key_id: int(row.id) for row in key_df.itertuples(index=False)}

    # Build pair.csv (positive qrels)
    pair_rows = []
    for qid, docrels in tqdm(qrels.items(), desc=f"[{dataset}] qrels"):
        if qid not in qid2new:
            # In rare cases, qrels can reference a query not in queries
            continue
        new_q = qid2new[qid]
        for doc_id, rel in docrels.items():
            if rel and rel > 0 and doc_id in did2new:
                new_d = did2new[doc_id]
                pair_rows.append({
                    "query_id": new_q,
                    "original_query_id": qid,
                    "key_id": new_d,
                    "original_key_id": doc_id,
                })

    out_ds = outdir / dataset
    out_ds.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    query_df.to_csv(out_ds / "query.csv", index=False)
    key_df.to_csv(out_ds / "key.csv", index=False)
    pd.DataFrame(pair_rows, columns=["query_id", "original_query_id", "key_id", "original_key_id"])\
        .to_csv(out_ds / "pair.csv", index=False)

    print(f"   Wrote: {out_ds/'query.csv'}, {out_ds/'key.csv'}, {out_ds/'pair.csv'}")


def main():
    p = argparse.ArgumentParser(description="Download BEIR datasets to query/key/pair CSV format.")
    p.add_argument("datasets", nargs="+", help="e.g., scifact nq fiqa")
    p.add_argument("--outdir", default="./out", help="Output directory")
    p.add_argument("--split", default="test", help="Preferred split (falls back to dev/train/others if absent)")
    args = p.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        try:
            convert_one(ds, outdir, args.split)
        except Exception as e:
            print(f"[ERROR] Dataset '{ds}' failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
