"""Score keys for each query by traversing a tag graph with a reward model."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import torch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rmsearch.tree.assign_key import assign_key_to_tag_tree
from rmsearch.utils.vllm_reward import build_llm, search


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sequence(path: Path) -> List[str]:
    data = _read_json(path)
    if isinstance(data, dict):
        values: Iterable[Any] = data.values()
    elif isinstance(data, list):
        values = data
    else:
        raise TypeError(f"Unsupported data type {type(data)} in {path}")

    items: List[str] = []
    for value in values:
        if isinstance(value, str):
            items.append(value)
        elif isinstance(value, dict) and "query" in value:
            items.append(str(value["query"]))
        elif isinstance(value, dict) and "text" in value:
            items.append(str(value["text"]))
    if not items:
        raise ValueError(f"No textual entries found in {path}")
    return items


def _get_tag_node(tag_ids: Sequence[int], tree: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not tag_ids:
        return None
    node: Any = tree
    for depth, tag_idx in enumerate(tag_ids):
        if not isinstance(node, list) or tag_idx < 0 or tag_idx >= len(node):
            return None
        selected = node[tag_idx]
        if depth == len(tag_ids) - 1:
            return selected
        node = selected.get("children", [])
    return None


def _tree_contains_field(tree: Sequence[Dict[str, Any]], field: str) -> bool:
    for node in tree:
        if field in node:
            return True
        children = node.get("children")
        if isinstance(children, list) and children and _tree_contains_field(children, field):
            return True
    return False


def _extract_node_key_ids(node: Optional[Dict[str, Any]]) -> List[int]:
    if not node:
        return []
    if "key_ids" in node and isinstance(node["key_ids"], list):
        return [int(idx) for idx in node["key_ids"]]
    if "query_ids" in node and isinstance(node["query_ids"], list):
        return [int(idx) for idx in node["query_ids"]]
    return []

'''
def search_key(
    queries: Sequence[str],
    keys: Sequence[str],
    tag_tree: Sequence[Dict[str, Any]],
    *,
    search_fn,
    k_tag: int = 2,
    k_key: int = 5,
) -> List[Dict[str, Any]]:
    key_records = [{"key": key} for key in keys]
    key2tag_ids, _ = assign_key_to_tag_tree(key_records, tag_tree, search_fn=search_fn, k_tag=k_tag)

    requests: List[Dict[str, Any]] = []
    for key_id, record in enumerate(key2tag_ids):
        candidate_key_ids: List[int] = []
        for tag_ids in record.get("tag_ids", []):
            node = _get_tag_node(tag_ids, tag_tree)
            if not node:
                continue
            candidate_key_ids.extend(int(idx) for idx in node.get("key_ids", []))

        deduped_ids: List[int] = []
        seen = set()
        for key_id in candidate_key_ids:
            if 0 <= key_id < len(keys) and key_id not in seen:
                seen.add(key_id)
                deduped_ids.append(key_id)

        selected_keys = [keys[key_id] for key_id in deduped_ids]
        requests.append(
            {
                "query": queries[query_id],
                "keys": selected_keys,
                "k": k_key,
                "return_relevance": True,
                "key_ids": deduped_ids,
            }
        )

    outputs = search_fn(requests)
    for idx, record in enumerate(outputs):
        request_key_ids = requests[idx]["key_ids"]
        for key_entry, key_id in zip(record.get("keys", []), request_key_ids):
            key_entry["relevant_id"] = key_id
    return outputs
'''


def search_key(
    queries: Sequence[str],
    keys: Sequence[str],
    tag2key: Sequence[Dict[str, Any]],
    *,
    search_fn,
    k_tag: int = 2,
    k_key: int = 5,
    checkpoint: Optional[Path] = None,
) -> List[Dict[str, Any]]:

    # queries: ["...", ...]
    # keys: ["...", ...]
    # tag2key: [{"tag":"", "key_ids":[0,2, ...], "children":[{"key_ids":[2, ...]},{"key_ids":[0, ...]}, ...]}, ...]

    checkpoint_dir = Path(checkpoint) if checkpoint else None
    raw_search_fn = search_fn

    def run_search_requests(requests: List[Dict[str, Any]], label: Optional[str]) -> List[Dict[str, Any]]:
        if not requests:
            return []
        checkpoint_path: Optional[Path] = None
        if checkpoint_dir and label:
            checkpoint_path = checkpoint_dir / f"{label}.json"
            if checkpoint_path.exists():
                return json.loads(checkpoint_path.read_text(encoding="utf-8"))

        results = raw_search_fn(requests)

        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        return results

    assign_call_idx = 0

    def assign_search(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nonlocal assign_call_idx
        label = f"assign_key-output{assign_call_idx}"
        assign_call_idx += 1
        return run_search_requests(requests, label)

    if not _tree_contains_field(tag2key, "key_ids"):
        key_records = [{"query": key} for key in keys]
        _, tag2key = assign_key_to_tag_tree(key_records, tag2key, search_fn=assign_search, k_tag=k_tag)

    query2tag_ids = [{"tag_ids":[[] for _ in range(k_tag)]} for i in range(len(queries))]   # query2tag_ids = [{"tag_ids":[[]]}, ...]   # (num_queries, "tag_ids":(k_tag, depth))
    
    tags = [tag2key_dict["tag"] for tag2key_dict in tag2key]
    tags_request = [{"tags":[tags]} for _ in range(len(queries))]    # tags_request = [{"tags":[[]]}, ...]   # (num_queries, "tags":(k_tag, num_tags_in_branch))
    while_end = False
    depth = 0

    def get_tag_dict(tag_ids, tag_dict):
        if tag_ids == []: return None
    
        for tag_id in tag_ids[:-1]:
            if "children" not in tag_dict[tag_id]:
                return None
            else:
                tag_dict = tag_dict[tag_id]["children"]
        
        return tag_dict[tag_ids[-1]]
        

    while not while_end:

        depth += 1

        requests = []
        query_and_n_top_ids = [] # [(query_id, n_top), ...] # (num_request)
        total_requests = 0

        for query_id in range(len(tags_request)):
            for nth_tag_ids, tag_list in enumerate(tags_request[query_id]["tags"]):
                query_and_n_top_ids.append((query_id, nth_tag_ids))
                requests.append({"query":queries[query_id], "keys":tag_list, "k": k_tag, "return_relevance":True})
                total_requests += len(tag_list)

        output = run_search_requests(requests, f"search_key-output{depth}")
        
        tags_request = [{"tags":[]} for _ in range(len(queries))] # tags_request = [{"tags":[[]]}, ...]   # (num_queries, "tags":(k_tag, num_tags_in_branch))
        result1 = {query_id:{"tag_ids_list":[], "relevance_list":[]} for query_id in range(len(queries))}   # {"query_id":{"tag_ids_list":[[3,1],],"relevance_list":[]}}
        for reuqest_id, output_dict in enumerate(output):
            query_id, nth_tag_ids = query_and_n_top_ids[reuqest_id]
            tags = []
            tag_relevance = []
            pre_tag_ids = query2tag_ids[query_id]["tag_ids"][nth_tag_ids]

            for top_nth in range(k_tag):
                try:  # output[query_id]["keys"][top_nth] can be index out of range. It goes to next output_dict.
                    new_tag_id = output_dict["keys"][top_nth]["key_id"]
                    relevance = output_dict["keys"][top_nth]["relevance"]
                    result1[query_id]["tag_ids_list"].append(pre_tag_ids+[new_tag_id])
                    result1[query_id]["relevance_list"].append(relevance)
                except:
                    continue

        while_end = True

        for query_id in result1:
            tag_relevance = result1[query_id]["relevance_list"]
            tag_ids_list = result1[query_id]["tag_ids_list"]
            if len(tag_relevance) == 0:
                top_tag_ids_list = []
            if len(tag_relevance) < k_tag:  # if there are not enough tag_ids
                _, indices = torch.topk(torch.tensor(tag_relevance), k=len(tag_relevance))
                top_tag_ids_list = [tag_ids_list[index.item()] for index in indices]
            else:  # normal pattern
                _, indices = torch.topk(torch.tensor(tag_relevance), k=k_tag)
                top_tag_ids_list = [tag_ids_list[index.item()] for index in indices]
                
            #_, indices = torch.topk(torch.tensor(tag_relevance), k=k_tag)
            #top_tag_ids_list = [tag_ids_list[index.item()] for index in indices]

            new_tag_ids_list = []
            for tag_ids in top_tag_ids_list:
                tag_dict = get_tag_dict(tag_ids, tag2key)
                if "children" not in tag_dict:
                    continue
                elif tag_dict["children"] == []:
                    continue

                tags = []
                for j in range(len(tag_dict["children"])):
                    tag = tag_dict["children"][j]["tag"]
                    tags.append(tag)

                while_end = False
                tags_request[query_id]["tags"].append(tags)
                new_tag_ids_list.append(tag_ids)

            query2tag_ids[query_id]["tag_ids"] = new_tag_ids_list
            
    query2key_ids = [] # [{"key_ids":[]}, ...]   # (num_queries, "key_ids":(n_total_hit_keys))
    total_requests = 0
    requests = []
    for query_id, _ in enumerate(query2tag_ids):
        combined_key_ids = []
        for tag_ids in _["tag_ids"]:
            tag_dict = get_tag_dict(tag_ids, tag2key)
            if not tag_dict:
                continue
            combined_key_ids.extend(_extract_node_key_ids(tag_dict))
            
        
        query2key_ids.append({"key_ids":combined_key_ids})
        selected_keys = []
        for key_id in combined_key_ids:
            try:
                selected_keys.append(keys[key_id])
            except:
                continue
            #selected_keys = [keys[key_id] for key_id in combined_key_ids]
            
        requests.append({"query":queries[query_id], "keys":selected_keys, "k": k_key, "return_relevance":True})
        total_requests += len(selected_keys)

    output = run_search_requests(requests, "search_key-final")

    # output : [{"query":, "query_id":0, "keys":[{"key_id":8,"key":,}, ...]}, ...]
    return output



def build_search_backend(
    *,
    model_name: str,
    tensor_parallel_size: int,
    num_instances: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
):
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    rm = build_llm(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=None,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        runner="pooling",
    )
    tokenizer = rm.tokenizer

    def llm_template_func(row: Dict[str, Any]) -> str:
        message = [
            {
                "role": "user",
                "content": (
                    "Give me relevance score between\n\n"
                    f"Query:{row['query']}\n\n"
                    f"Sentence:{row['key']}"
                ),
            }
        ]
        if len(message[0]["content"]) > 4000:
            message[0]["content"] = message[0]["content"][:4000] + "..."
        return tokenizer.apply_chat_template(message, tokenize=False)

    def run_search(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not requests:
            return []
        topk = max((req.get("k", 1) for req in requests), default=1)
        batch_size = max(1, min(len(requests), 128))
        return search(
            rm,
            requests,
            llm_template_func,
            topk=topk,
            batch_size=batch_size,
            timeout_s=10_000.0,
        )

    return rm, run_search


def _sample_inputs() -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    queries = [
        "How can I route retrieval queries through a tag graph?",
        "What is a good baseline for lexical search evaluation?",
    ]
    keys = [
        "Graph-based retrieval augments vector search with hierarchical tags.",
        "BM25 provides a strong lexical baseline for document ranking.",
        "Hybrid retrieval blends semantic and lexical evidence to improve recall.",
    ]
    tag_tree = [
        {
            "tag": "Retrieval",
            "key_ids": [0, 1, 2],
            "children": [
                {"tag": "Graph Retrieval", "key_ids": [0, 2], "children": []},
                {"tag": "Lexical Search", "key_ids": [1], "children": []},
            ],
        }
    ]
    return queries, keys, tag_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank keys using tag graph traversal with a reward model.")
    parser.add_argument("--queries", type=Path, help="JSON file containing query strings or objects with a 'query' field.")
    parser.add_argument("--keys", type=Path, help="JSON file containing key strings or objects with a 'text' field.")
    parser.add_argument("--tag2key", type=Path, help="JSON file with the tag graph annotated with 'key_ids'.")
    parser.add_argument("--k-tag", type=int, default=2, help="Number of top tags explored per layer.")
    parser.add_argument("--k-key", type=int, default=5, help="Number of keys retrieved per query.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of reward model worker processes.")
    parser.add_argument("--model-name", type=str, required=True, help="Path or identifier of the reward model.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for the reward model.")
    parser.add_argument("--max-model-len", type=int, default=2500, help="Maximum sequence length for the reward model.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker.")
    parser.add_argument("--output", type=Path, help="Optional path to store ranked results in JSON format.")
    parser.add_argument("--checkpoint", type=Path, help="Directory for caching intermediate search outputs.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory made available to the reward model workers.",
    )
    return parser.parse_args()


def main() -> None:
    logging.getLogger("vllm").setLevel(logging.ERROR)
    args = parse_args()

    if args.queries and args.keys and args.tag2key:
        queries = _load_sequence(args.queries)
        keys = _load_sequence(args.keys)
        tag_tree = _read_json(args.tag2key)
    else:
        queries, keys, tag_tree = _sample_inputs()
        print("Using in-memory sample queries, keys, and tag graph.")

    rm = None
    outputs: List[Dict[str, Any]] = []
    try:
        rm, run_search = build_search_backend(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            num_instances=args.num_instances,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        outputs = search_key(
            queries,
            keys,
            tag_tree,
            search_fn=run_search,
            k_tag=args.k_tag,
            k_key=args.k_key,
            checkpoint=args.checkpoint,
        )
    finally:
        if rm is not None:
            rm.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved search results to {args.output}")


if __name__ == "__main__":
    main()
