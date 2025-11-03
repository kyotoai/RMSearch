"""Rerank embedding candidates with a reward model and output evaluation format."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from rmsearch.utils.vllm_reward import build_llm, search

__all__ = ["rerank_candidates"]

logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_dataset_from_paths(
    queries_path: Path,
    corpus_path: Path,
    qrels_path: Path | None = None,
    split: str = "test"
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load queries, corpus, and qrels from BEIR JSONL/TSV files.
    
    Returns:
        (queries, corpus, qrels)
    """
    logger.info("Loading dataset from file paths...")
    
    # Load queries
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries[data['_id']] = data['text']
    
    # Load corpus
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                title = data.get('title', '')
                text = data.get('text', '')
                combined = f"{title} {text}".strip() if title else text
                corpus[data['_id']] = combined
    
    # Load qrels if provided
    qrels = defaultdict(dict)
    if qrels_path and qrels_path.exists():
        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        qid, docid, rel = parts
                        if rel == 'score':  # Skip header
                            continue
                        qrels[qid][docid] = int(rel)
                except (IndexError, ValueError):
                    continue
    
    logger.info("Loaded %d queries, %d documents, %d qrels", len(queries), len(corpus), len(qrels))
    return queries, corpus, dict(qrels)


def load_embed_output(embed_json_path: Path) -> tuple[Dict[int, str], Dict[int, str], List[Dict]]:
    """
    Load embed output JSON and extract queries, corpus, and ranking records.
    
    The embed output contains mappings we can use to reconstruct the original data.
    Returns:
        (query_mapping, corpus_mapping, embed_records)
    """
    logger.info("Loading embed output from %s", embed_json_path)
    
    with open(embed_json_path, 'r', encoding='utf-8') as f:
        embed_records = json.load(f)
    
    if not isinstance(embed_records, list):
        raise TypeError(f"Expected list in {embed_json_path}")
    
    # Extract unique query and key IDs from records
    query_ids = set()
    key_ids = set()
    
    for record in embed_records:
        query_ids.add(int(record['query_id']))
        key_ids.update(int(k) for k in record['key_ids'])
    
    logger.info("Found %d unique queries and %d unique keys in embed output", 
                len(query_ids), len(key_ids))
    
    # We don't have the actual text in embed output, so we need to load from source files
    # Return empty mappings that will be populated from BEIR files
    return {}, {}, embed_records


def create_id_mappings(
    queries: Dict[str, str],
    corpus: Dict[str, str]
) -> tuple[Dict[int, str], Dict[int, str], Dict[str, int], Dict[str, int]]:
    """
    Create numeric ID mappings from string IDs.
    
    Returns:
        (query_text_by_numeric_id, corpus_text_by_numeric_id, 
         query_str_to_numeric, corpus_str_to_numeric)
    """
    query_text_by_id = {}
    query_mapping = {}
    for idx, (qid, text) in enumerate(queries.items()):
        query_mapping[qid] = idx
        query_text_by_id[idx] = text
    
    corpus_text_by_id = {}
    corpus_mapping = {}
    for idx, (docid, text) in enumerate(corpus.items()):
        corpus_mapping[docid] = idx
        corpus_text_by_id[idx] = text
    
    return query_text_by_id, corpus_text_by_id, query_mapping, corpus_mapping


# ============================================================================
# Reranking Functions
# ============================================================================

def _llm_template(tokenizer):
    def format_row(row: Dict[str, str]) -> str:
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

    return format_row


def _parse_device_groups(spec: str | None, tensor_parallel_size: int, num_instances: int):
    if spec is None:
        return None
    groups: List[List[int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        group = [int(token) for token in chunk.split(",") if token.strip()]
        if not group:
            continue
        groups.append(group)
    if not groups:
        return None
    if len(groups) != num_instances:
        raise ValueError(f"Expected {num_instances} device groups, got {len(groups)}")
    for group in groups:
        if len(group) != tensor_parallel_size:
            raise ValueError(f"Each device group must contain {tensor_parallel_size} devices (got {group})")
    return groups


def rerank_candidates(
    query_text_by_id: Mapping[int, str],
    key_text_by_id: Mapping[int, str],
    embed_records: Sequence[Dict[str, object]],
    *,
    model_name: str,
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    device_groups: List[List[int]] | None = None,
    max_model_len: int = 2500,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.90,
    request_batch_size: int = 128,
    timeout_s: float = 10_000.0,
    top_k: int | None = None,
    positive_pairs: Mapping[int, List[int]] | None = None,
) -> List[Dict[str, object]]:
    if not query_text_by_id:
        raise ValueError("Query mapping is empty.")
    if not key_text_by_id:
        raise ValueError("Key mapping is empty.")
    if not embed_records:
        raise ValueError("Embedding rankings list is empty.")

    logger.info("Launching reward model %s for reranking", model_name)
    rm = build_llm(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=device_groups,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        runner="pooling",
    )
    tokenizer = rm.tokenizer
    template_fn = _llm_template(tokenizer)

    requests: List[Dict[str, object]] = []
    id_maps: List[List[int]] = []
    request_query_ids: List[int] = []
    embed_scores_by_request: List[List[float]] = []

    for record in embed_records:
        query_id = int(record["query_id"])
        query_text = query_text_by_id.get(query_id)
        if query_text is None:
            logger.warning("Query id %d missing from query mapping, skipping", query_id)
            continue
        candidate_ids = [int(k) for k in record["key_ids"]]  # type: ignore[index]
        if not candidate_ids:
            continue
        embed_scores = [float(score) for score in record.get("embed_relevances", [])]  # type: ignore[arg-type]
        if len(embed_scores) != len(candidate_ids):
            raise ValueError(f"Embedding relevances for query {query_id} do not align with key ids.")
        key_texts: List[str] = []
        valid_ids: List[int] = []
        valid_embed_scores: List[float] = []
        for kid, score in zip(candidate_ids, embed_scores):
            key_text = key_text_by_id.get(kid)
            if key_text is None:
                logger.warning("Key id %d missing from key mapping, skipping", kid)
                continue
            key_texts.append(key_text)
            valid_ids.append(kid)
            valid_embed_scores.append(score)
        
        if not key_texts:
            continue
            
        requests.append({"query": query_text, "keys": key_texts, "return_relevance": True})
        id_maps.append(valid_ids)
        request_query_ids.append(query_id)
        embed_scores_by_request.append(valid_embed_scores)

    if not requests:
        rm.close()
        raise ValueError("No requests were constructed from embedding rankings.")

    logger.info("Scoring %d requests with reward model", len(requests))
    try:
        outputs = search(
            rm,
            requests,
            template_fn,
            topk=max(len(ids) for ids in id_maps),
            batch_size=request_batch_size,
            timeout_s=timeout_s,
        )
    finally:
        rm.close()

    positive_lookup = positive_pairs or {}
    reranked: List[Dict[str, object]] = []
    for req_query_id, result, original_ids, embed_scores in zip(
        request_query_ids, outputs, id_maps, embed_scores_by_request
    ):
        ordered_ids: List[int] = []
        scores: List[float] = []
        for item in result.get("keys", []):
            local_id = int(item["key_id"])
            if local_id < 0 or local_id >= len(original_ids):
                continue
            ordered_ids.append(original_ids[local_id])
            scores.append(float(item.get("relevance", 0.0)))
        pre_key_ids = list(original_ids)
        limit = top_k if top_k is not None else len(ordered_ids)
        limit = min(limit, len(ordered_ids))
        entry: Dict[str, object] = {
            "query_id": req_query_id,
            "pre_key_ids": pre_key_ids,
            "key_ids": ordered_ids[:limit],
            "embed_relevances": embed_scores,
            "rerank_relevances": scores[:limit],
            "positive_key_ids": positive_lookup.get(req_query_id, []),
        }
        reranked.append(entry)
    return reranked


# ============================================================================
# Output Transformation
# ============================================================================

def transform_to_eval_format(reranked_output: List[Dict]) -> Dict[int, Dict[int, float]]:
    """
    Transform reranked output to evaluation format.
    Input: [{"query_id": X, "key_ids": [...], "rerank_relevances": [...]}, ...]
    Output: {query_id: {key_id: score, ...}, ...}
    """
    result = {}
    
    for query in reranked_output:
        query_id = query.get("query_id")
        key_ids = query.get("key_ids", [])
        rerank_scores = query.get("rerank_relevances", [])
        
        # Create dictionary mapping key_id to rerank score
        query_results = {}
        for key_id, score in zip(key_ids, rerank_scores):
            query_results[key_id] = score
        
        result[query_id] = query_results
    
    return result


# ============================================================================
# Main Function
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank embedding candidates with a reward model for BEIR datasets."
    )
    
    # Input: BEIR dataset files
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to BEIR dataset folder (for loading original text). Alternative to manual paths."
    )
    parser.add_argument("--queries", type=Path, help="BEIR queries JSONL file (manual mode)")
    parser.add_argument("--corpus", type=Path, help="BEIR corpus JSONL file (manual mode)")
    parser.add_argument("--qrels", type=Path, help="BEIR qrels TSV file (manual mode, optional)")
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to use for qrels (default: test)"
    )
    
    # Input: Embed output (required)
    parser.add_argument(
        "--embed-output",
        type=Path,
        required=True,
        help="Embedding output JSON from embed.py (e.g., relevant_emb.json)"
    )
    
    # Output files
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for intermediate reranked results (original format). If not specified, only eval output is saved."
    )
    parser.add_argument(
        "--output-eval",
        type=Path,
        required=True,
        help="Output JSON file for evaluation reranked results (transformed format)"
    )
    
    # Model configuration
    parser.add_argument("--model-name", type=str, required=True, help="Reward model identifier or checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for reward model")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of reward model worker instances")
    parser.add_argument(
        "--device-groups",
        type=str,
        help="Explicit GPU mapping, e.g. '0,1;2,3' for tensor_parallel_size=2 and num_instances=2",
    )
    parser.add_argument("--max-model-len", type=int, default=2500, help="Maximum sequence length for the reward model")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory assigned to the reward model",
    )
    
    # Reranking options
    parser.add_argument("--request-batch-size", type=int, default=128, help="Number of (query,key) pairs per scoring batch")
    parser.add_argument("--timeout", type=float, default=10_000.0, help="Timeout in seconds for reward scoring batches")
    parser.add_argument("--top-k", type=int, default=10, help="Number of most relevant keys to keep per query in output")
    
    # Other
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    mp.set_start_method("spawn", force=True)
    
    logger.info("=" * 60)
    logger.info("BEIR Reranking Evaluation Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load embed output
    logger.info("\nStep 1: Loading embed output...")
    _, _, embed_records = load_embed_output(args.embed_output)
    
    # Step 2: Load BEIR dataset for text
    logger.info("\nStep 2: Loading BEIR dataset for text...")
    if args.queries and args.corpus:
        # Manual mode
        queries, corpus, qrels = load_dataset_from_paths(
            args.queries,
            args.corpus,
            args.qrels,
            args.split
        )
    elif args.dataset_path:
        # Auto-load mode
        queries_path = args.dataset_path / "queries.jsonl"
        corpus_path = args.dataset_path / "corpus.jsonl"
        qrels_path = args.dataset_path / "qrels" / f"{args.split}.tsv"
        
        queries, corpus, qrels = load_dataset_from_paths(
            queries_path,
            corpus_path,
            qrels_path if qrels_path.exists() else None,
            args.split
        )
    else:
        raise ValueError(
            "Must specify either:\n"
            "  1. --queries and --corpus (manual mode), OR\n"
            "  2. --dataset-path (auto-load mode)"
        )
    
    # Step 3: Create ID mappings
    logger.info("\nStep 3: Creating ID mappings...")
    query_text_by_id, corpus_text_by_id, query_mapping, corpus_mapping = create_id_mappings(
        queries, corpus
    )
    
    # Extract positive pairs from embed records
    positive_pairs = {}
    for record in embed_records:
        if "positive_key_ids" in record and isinstance(record["positive_key_ids"], list):
            query_id = int(record["query_id"])
            positive_pairs[query_id] = [int(k) for k in record["positive_key_ids"]]
    
    # Step 4: Parse device groups
    device_groups = _parse_device_groups(
        args.device_groups,
        args.tensor_parallel_size,
        args.num_instances
    )
    
    # Step 5: Rerank
    logger.info("\nStep 4: Reranking with reward model...")
    reranked = rerank_candidates(
        query_text_by_id,
        corpus_text_by_id,
        embed_records,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        device_groups=device_groups,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        request_batch_size=args.request_batch_size,
        timeout_s=args.timeout,
        top_k=args.top_k,
        positive_pairs=positive_pairs,
    )
    
    # Step 6: Save intermediate output (optional)
    if args.output:
        logger.info("\nStep 5: Saving intermediate output...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(reranked, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        logger.info("✓ Saved intermediate output to %s", args.output)
    
    # Step 7: Transform and save evaluation output
    logger.info("\nStep 6: Transforming to evaluation format...")
    eval_output = transform_to_eval_format(reranked)
    
    args.output_eval.parent.mkdir(parents=True, exist_ok=True)
    args.output_eval.write_text(
        json.dumps(eval_output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    logger.info("✓ Saved evaluation output to %s", args.output_eval)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Complete!")
    logger.info("  Queries reranked: %d", len(reranked))
    if args.output:
        logger.info("  Intermediate output: %s", args.output)
    logger.info("  Evaluation output: %s", args.output_eval)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()