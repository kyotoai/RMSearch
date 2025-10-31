"""Retrieval evaluation utilities."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from rmsearch.tree.assign_key import assign_key_to_tag_tree

__all__ = ["retrieval_evaluation"]

SearchFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
AsyncSearchFn = Callable[[List[Dict[str, Any]]], Awaitable[List[Dict[str, Any]]]]


def _get_tag_dict(tag_ids: List[int], tag_tree: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not tag_ids:
        return None
    node: Any = tag_tree
    for tag_id in tag_ids[:-1]:
        if not isinstance(node, list) or tag_id >= len(node):
            return None
        node = node[tag_id].get("children", [])
    if not isinstance(node, list) or tag_ids[-1] >= len(node):
        return None
    return node[tag_ids[-1]]


def _run_search(search_fn: SearchFn | AsyncSearchFn, requests: List[Dict[str, Any]]):
    maybe_result = search_fn(requests)
    if asyncio.iscoroutine(maybe_result):
        return asyncio.run(maybe_result)
    return maybe_result


def retrieval_evaluation(
    queries: Sequence[str],
    sentences: Sequence[str],
    tag_tree: List[Dict[str, Any]],
    *,
    search_fn: SearchFn | AsyncSearchFn,
    k_tag: int = 2,
    k_key: int = 10,
    correct_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Score sentence relevance by navigating the tag tree before ranking keys."""

    query_records = [{"query": query} for query in queries]
    query2tag_ids, tag2query = assign_key_to_tag_tree(query_records, tag_tree, search_fn=search_fn, k_tag=k_tag)

    requests: List[Dict[str, Any]] = []
    for query_id, record in enumerate(query2tag_ids):
        combined_key_ids: List[int] = []
        for tag_ids in record["tag_ids"]:
            tag_info = _get_tag_dict(tag_ids, tag2query)
            if tag_info and "query_ids" in tag_info:
                combined_key_ids.extend(int(idx) for idx in tag_info["query_ids"])
        unique_keys = []
        seen = set()
        for idx in combined_key_ids:
            if 0 <= idx < len(sentences) and idx not in seen:
                seen.add(idx)
                unique_keys.append(idx)
        selected_sentences = [sentences[idx] for idx in unique_keys]
        requests.append(
            {
                "query": queries[query_id],
                "keys": selected_sentences,
                "k": k_key,
                "return_relevance": True,
                "key_ids": unique_keys,
            }
        )

    outputs = _run_search(search_fn, requests)

    for idx, record in enumerate(outputs):
        record["correct_id"] = int(correct_ids[idx]) if correct_ids and idx < len(correct_ids) else None
        for key_entry, key_id in zip(record.get("keys", []), requests[idx]["key_ids"]):
            key_entry["relevant_id"] = key_id

    # outputs (list): one element per query containing
    #   {
    #     "keys": [
    #        {"key_id": <local index in request>, "relevance": <score>, "relevant_id": <global sentence idx>, ...},
    #        ...
    #     ],
    #     "correct_id": <oracle index if provided>
    #   }
    return outputs


if __name__ == "__main__":
    import argparse
    import json
    import logging
    import multiprocessing as mp
    from typing import Tuple

    import pandas as pd

    from rmsearch.utils.vllm_reward import build_llm, search

    def _load_table_csv(path: Path, text_column: str, id_column: str) -> Tuple[List[str], List[int]]:
        df = pd.read_csv(path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not present in {path}")
        df = df[df[text_column].notna()].copy()
        if df.empty:
            raise ValueError(f"Column '{text_column}' in {path} is empty")
        if id_column and id_column in df.columns:
            df[id_column] = df[id_column].astype(int)
            df = df.sort_values(id_column)
            ids = df[id_column].astype(int).tolist()
        else:
            ids = list(range(len(df)))
        texts = df[text_column].astype(str).tolist()
        return texts, ids

    def _load_positive_pairs_csv(path: Path, query_column: str, key_column: str) -> Dict[int, List[int]]:
        df = pd.read_csv(path)
        for column in (query_column, key_column):
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not present in {path}")
        df = df[[query_column, key_column]].dropna()
        df[query_column] = df[query_column].astype(int)
        df[key_column] = df[key_column].astype(int)
        grouped = df.groupby(query_column)[key_column].apply(list)
        return {int(qid): [int(k) for k in keys] for qid, keys in grouped.items()}

    def _score_without_graph(
        queries: Sequence[str],
        keys: Sequence[str],
        *,
        search_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
        k_key: int,
        query_ids: Sequence[int],
        key_ids: Sequence[int],
        positive_pairs: Optional[Dict[int, List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        if not keys:
            raise ValueError("Key list is empty.")
        if k_key <= 0:
            raise ValueError("k-key must be positive.")
        effective_k = min(k_key, len(keys))
        requests: List[Dict[str, Any]] = []
        for query_text in queries:
            requests.append(
                {
                    "query": query_text,
                    "keys": list(keys),
                    "k": effective_k,
                    "return_relevance": True,
                }
            )

        outputs = search_fn(requests)
        results: List[Dict[str, Any]] = []
        for idx, result in enumerate(outputs):
            query_id = int(query_ids[idx]) if idx < len(query_ids) else idx
            positives = positive_pairs.get(query_id, []) if positive_pairs else []
            result["query_id"] = query_id
            result["correct_id"] = positives[0] if positives else None
            if positives:
                result["positive_key_ids"] = positives
            for key_item in result.get("keys", []):
                local_id = int(key_item["key_id"])
                if 0 <= local_id < len(key_ids):
                    key_item["relevant_id"] = int(key_ids[local_id])
            results.append(result)
        return results

    parser = argparse.ArgumentParser(description="Run retrieval evaluation using a vLLM reward model.")
    parser.add_argument("--model-name", type=str, default="/workspace/llama3b-rm-converted-model", help="Path to the converted reward model.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for the reward model workers.")
    parser.add_argument("--num-instances", type=int, default=4, help="Number of reward model worker instances.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size per inference call.")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout in seconds for reward model requests.")
    parser.add_argument("--k-tag", type=int, default=2, help="Branching factor when traversing the tag tree.")
    parser.add_argument("--k-key", type=int, default=10, help="Number of keys retrieved per query in the final stage.")
    parser.add_argument("--output", type=Path, default=Path("relevance_dict.json"), help="Where to save the evaluation results.")

    parser.add_argument("--query-csv", type=Path, default=Path("query.csv"), help="BEIR-style query CSV (id,text).")
    parser.add_argument("--query-text-column", type=str, default="text", help="Column containing query text in --query-csv.")
    parser.add_argument("--query-id-column", type=str, default="id", help="Column containing query ids in --query-csv.")
    parser.add_argument("--key-csv", type=Path, default=Path("key.csv"), help="BEIR-style key CSV (id,text).")
    parser.add_argument("--key-text-column", type=str, default="text", help="Column containing key text in --key-csv.")
    parser.add_argument("--key-id-column", type=str, default="id", help="Column containing key ids in --key-csv.")
    parser.add_argument("--pair-csv", type=Path, help="Optional BEIR-style pair CSV (query_id,key_id).")
    parser.add_argument("--pair-query-column", type=str, default="query_id", help="Query id column inside --pair-csv.")
    parser.add_argument("--pair-key-column", type=str, default="key_id", help="Key id column inside --pair-csv.")

    parser.add_argument("--working-dir", type=Path, default=Path("/workspace/RMS_exp"), help="Legacy root working directory used during training.")
    parser.add_argument("--data-name", type=str, default="smollm-corpus", help="Legacy dataset identifier under the working directory.")
    args = parser.parse_args()

    logging.getLogger("vllm").setLevel(logging.ERROR)
    mp.set_start_method("spawn", force=True)

    use_beir_inputs = args.query_csv and args.key_csv and args.query_csv.is_file() and args.key_csv.is_file()

    if use_beir_inputs:
        queries, query_ids = _load_table_csv(args.query_csv, args.query_text_column, args.query_id_column)
        sentences, key_ids = _load_table_csv(args.key_csv, args.key_text_column, args.key_id_column)
        positive_pairs = (
            _load_positive_pairs_csv(args.pair_csv, args.pair_query_column, args.pair_key_column)
            if args.pair_csv
            else {}
        )
        correct_ids = None
        tag_tree = None
    else:
        working_dir = args.working_dir
        data_dir = working_dir / "data" / args.data_name

        df = pd.read_csv(data_dir / "df_small.csv")
        with (data_dir / "query_dict.json").open() as handle:
            query_dict = json.load(handle)
        with (data_dir / "tag2query-tag_tree.json").open() as handle:
            tag_tree = json.load(handle)

        sentences = [df.iloc[i]["text"] for i in range(len(df))]
        queries_list: List[str] = []
        correct_ids = []
        for idx in range(len(df)):
            questions = query_dict[str(idx)]["questions"]
            queries_list.extend(questions)
            correct_ids.extend([idx for _ in range(len(questions))])
        queries = queries_list
        query_ids = list(range(len(queries)))
        key_ids = list(range(len(sentences)))
        positive_pairs = {}

    device_groups: List[List[int]] = []
    device_id = 0
    for _ in range(args.num_instances):
        group = []
        for _ in range(args.tensor_parallel_size):
            group.append(device_id)
            device_id += 1
        device_groups.append(group)

    rm = build_llm(
        model_name=args.model_name,
        tensor_parallel_size=len(device_groups[0]) if device_groups else args.tensor_parallel_size,
        num_instances=len(device_groups) or args.num_instances,
        device_groups=device_groups if device_groups else None,
        max_model_len=2500,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
        runner="pooling",
    )
    tokenizer = rm.tokenizer

    def llm_template_func(row: Dict[str, Any]) -> str:
        query = row["query"]
        key = row["key"]
        message = [
            {
                "role": "user",
                "content": (
                    "Give me relevance score between\n\n"
                    f"Query:{query}\n\n"
                    f"Sentence:{key}"
                ),
            }
        ]
        if len(message[0]["content"]) > 4000:
            message[0]["content"] = message[0]["content"][:4000] + "..."
        return tokenizer.apply_chat_template(message, tokenize=False)

    def run_search(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not requests:
            return []
        default_topk = args.k_key if use_beir_inputs else args.k_tag
        topk = max((req.get("k", default_topk) for req in requests), default=default_topk)
        return search(
            rm,
            requests,
            llm_template_func,
            topk=topk,
            batch_size=args.batch_size,
            timeout_s=args.timeout,
        )

    try:
        if use_beir_inputs:
            outputs = _score_without_graph(
                queries,
                sentences,
                search_fn=run_search,
                k_key=args.k_key,
                query_ids=query_ids,
                key_ids=key_ids,
                positive_pairs=positive_pairs,
            )
        else:
            outputs = retrieval_evaluation(
                queries,
                sentences,
                tag_tree,  # type: ignore[arg-type]
                search_fn=run_search,
                k_tag=args.k_tag,
                k_key=args.k_key,
                correct_ids=correct_ids,
            )
    finally:
        rm.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, ensure_ascii=False, indent=2)

    print(f"Saved retrieval evaluation results to {args.output}")
