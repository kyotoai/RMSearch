"""Generate embedding-based relevance rankings for BEIR datasets. Using vllm"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.request import urlretrieve

import torch

from rmsearch.utils import vllm_embed

__all__ = ["build_relevance_dict"]

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Download and Management
# ============================================================================

def download_beir_dataset(dataset_path: Path, dataset_name: str):
    """Download BEIR dataset if not exists using beir utility."""
    try:
        from beir import util
    except ImportError:
        raise ImportError(
            "beir package is required for auto-download. "
            "Install it with: pip install beir"
        )
    
    logger.info("Dataset not found at %s", dataset_path)
    logger.info("Downloading BEIR/%s...", dataset_name)
    
    # Create parent directory
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(dataset_path.parent))
    
    logger.info("Dataset downloaded to %s", data_path)


# ============================================================================
# BEIR Data Loading Functions
# ============================================================================

def load_dataset(dataset_path: Path, split: str = "test") -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load queries, corpus, and qrels from the dataset folder.
    Downloads dataset if it doesn't exist.
    
    Returns:
        Tuple of (queries, corpus, qrels)
        - queries: {query_id: query_text}
        - corpus: {doc_id: doc_text}
        - qrels: {query_id: {doc_id: relevance_score}}
    """
    from collections import defaultdict
    
    if not dataset_path.exists():
        dataset_name = dataset_path.name
        download_beir_dataset(dataset_path, dataset_name)
    
    logger.info("Loading dataset from %s...", dataset_path)
    
    # Load queries
    queries_file = dataset_path / "queries.jsonl"
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries[data['_id']] = data['text']
    
    # Load corpus
    corpus_file = dataset_path / "corpus.jsonl"
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                title = data.get('title', '')
                text = data.get('text', '')
                combined = f"{title} {text}".strip() if title else text
                corpus[data['_id']] = combined
    
    # Load qrels
    qrels_file = dataset_path / "qrels" / f"{split}.tsv"
    qrels = defaultdict(dict)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    qid, docid, rel = parts
                    # Skip header
                    if rel == 'score':
                        continue
                    qrels[qid][docid] = int(rel)
                else:
                    logger.debug("Skipping line with incorrect field count: %s", line.strip())
            except (IndexError, ValueError) as e:
                logger.debug("Error parsing line: %s", e)
                continue
    
    logger.info("Loaded %d queries, %d documents, %d qrels", len(queries), len(corpus), len(qrels))
    
    # Debug: Check for mismatches
    qrel_qids = set(qrels.keys())
    query_qids = set(queries.keys())
    missing_queries = qrel_qids - query_qids
    if missing_queries:
        logger.warning("%d query IDs in qrels but not in queries", len(missing_queries))
        logger.warning("Example missing: %s", list(missing_queries)[:5])
    
    # Check corpus coverage
    all_doc_ids_in_qrels = set()
    for qid, docs in qrels.items():
        all_doc_ids_in_qrels.update(docs.keys())
    missing_docs = all_doc_ids_in_qrels - set(corpus.keys())
    if missing_docs:
        logger.warning("%d document IDs in qrels but not in corpus", len(missing_docs))
        logger.warning("Example missing: %s", list(missing_docs)[:5])
    
    return queries, corpus, dict(qrels)


def convert_to_lists(
    queries: Dict[str, str],
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]]
) -> Tuple[List[str], List[int], Dict[str, int], List[str], List[int], Dict[str, int], Dict[int, List[int]]]:
    """
    Convert dictionary format to list format needed for embedding.
    
    Returns:
        (query_texts, query_ids, query_mapping, 
         corpus_texts, corpus_ids, corpus_mapping,
         positive_pairs)
    """
    # Convert queries
    query_texts = []
    query_ids = []
    query_mapping = {}  # original_id -> numeric_id
    
    for idx, (qid, text) in enumerate(queries.items()):
        query_mapping[qid] = idx
        query_texts.append(text)
        query_ids.append(idx)
    
    # Convert corpus
    corpus_texts = []
    corpus_ids = []
    corpus_mapping = {}  # original_id -> numeric_id
    
    for idx, (docid, text) in enumerate(corpus.items()):
        corpus_mapping[docid] = idx
        corpus_texts.append(text)
        corpus_ids.append(idx)
    
    # Convert qrels to positive pairs
    positive_pairs = {}
    for qid, docs in qrels.items():
        if qid in query_mapping:
            numeric_qid = query_mapping[qid]
            positive_pairs[numeric_qid] = []
            for docid in docs.keys():
                if docid in corpus_mapping:
                    numeric_docid = corpus_mapping[docid]
                    positive_pairs[numeric_qid].append(numeric_docid)
    
    logger.info("Converted to numeric IDs: %d queries, %d documents", len(query_texts), len(corpus_texts))
    
    return query_texts, query_ids, query_mapping, corpus_texts, corpus_ids, corpus_mapping, positive_pairs


# ============================================================================
# Embedding Functions
# ============================================================================

def _embed_texts(
    embedder: vllm_embed.EmbeddingWorkerModel,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> torch.Tensor:
    vectors = vllm_embed.embed(
        embedder,
        list(texts),
        batch_size=batch_size,
    )
    if not vectors:
        raise ValueError("Embedding backend returned no vectors.")
    tensor = torch.tensor(vectors, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(tensor.shape)}")
    return tensor


def _maybe_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, p=2.0, dim=-1)


def build_relevance_dict(
    queries: Sequence[str],
    keys: Sequence[str],
    *,
    model_name: str,
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    max_model_len: int = 4000,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.90,
    query_batch_size: int = 256,
    key_batch_size: int = 128,
    top_k: int = 100,
    normalize: bool = True,
    similarity_device: str = "cpu",
) -> Tuple[List[List[int]], List[List[float]]]:
    if not queries:
        raise ValueError("Query list is empty.")
    if not keys:
        raise ValueError("Key list is empty.")
    logger.info("Embedding %d queries and %d keys with model %s", len(queries), len(keys), model_name)

    embedder = vllm_embed.build_embedding_model(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        output_to_cpu=True,
    )
    try:
        query_tensor = _embed_texts(embedder, queries, batch_size=query_batch_size)
        key_tensor = _embed_texts(embedder, keys, batch_size=key_batch_size)
    finally:
        embedder.close()

    if query_tensor.shape[1] != key_tensor.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: queries {query_tensor.shape[1]} vs keys {key_tensor.shape[1]}"
        )

    if normalize:
        query_tensor = _maybe_normalize(query_tensor)
        key_tensor = _maybe_normalize(key_tensor)

    device_choice = (similarity_device or "cpu").strip().lower()
    if device_choice == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = similarity_device
    device = torch.device(resolved_device)
    query_tensor = query_tensor.to(device)
    key_tensor = key_tensor.to(device)

    logger.info("Computing similarity matrix on %s", device)
    relevance = query_tensor @ key_tensor.T

    k = min(top_k, key_tensor.shape[0])
    if k <= 0:
        raise ValueError("top_k must be positive.")
    scores, indices = torch.topk(relevance, k=k, dim=1)
    del relevance

    return indices.cpu().tolist(), scores.cpu().tolist()


# ============================================================================
# Output Transformation Functions
# ============================================================================

def create_intermediate_output(
    query_ids: List[int],
    indices: List[List[int]],
    scores: List[List[float]],
    key_ids: List[int],
    positive_pairs: Dict[int, List[int]],
) -> List[Dict]:
    """
    Create intermediate output format (original format before transformation).
    Output: [{"query_id": X, "key_ids": [...], "embed_relevances": [...], "positive_key_ids": [...]}, ...]
    """
    payload = []
    
    for idx, key_idx_list in enumerate(indices):
        query_id = int(query_ids[idx])
        resolved_key_ids = [int(key_ids[key_idx]) for key_idx in key_idx_list]
        resolved_scores = [float(score) for score in scores[idx]]
        
        if len(resolved_scores) != len(resolved_key_ids):
            raise RuntimeError("Mismatch between resolved key ids and embedding relevances.")
        
        entry: Dict[str, object] = {
            "query_id": query_id,
            "key_ids": resolved_key_ids,
            "embed_relevances": resolved_scores,
        }
        
        if positive_pairs:
            entry["positive_key_ids"] = positive_pairs.get(query_id, [])
        
        payload.append(entry)
    
    return payload


def transform_to_eval_format(intermediate_output: List[Dict]) -> Dict[int, Dict[int, float]]:
    """
    Transform from intermediate format to evaluation format.
    Output: {query_id: {key_id: score, ...}, ...}
    """
    result = {}
    
    for query in intermediate_output:
        query_id = query.get("query_id")
        key_ids = query.get("key_ids", [])
        relevant_scores = query.get("embed_relevances", [])
        
        # Create dictionary mapping key_id to relevant score
        query_results = {}
        for key_id, score in zip(key_ids, relevant_scores):
            query_results[key_id] = score
        
        result[query_id] = query_results
    
    return result


# ============================================================================
# Main Function
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embedding-based rankings for BEIR datasets with automatic download."
    )
    
    # BEIR dataset input - supports both auto-download and manual paths
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to BEIR dataset folder (will auto-download if doesn't exist). Alternative to --queries/--corpus/--qrels."
    )
    
    # Manual file paths (takes precedence over --dataset-path)
    parser.add_argument("--queries", type=Path, help="BEIR queries JSONL file (manual mode)")
    parser.add_argument("--corpus", type=Path, help="BEIR corpus JSONL file (manual mode)")
    parser.add_argument("--qrels", type=Path, help="BEIR qrels TSV file (manual mode)")
    
    # Auto-download options
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to use for qrels (default: test)"
    )
    
    # Output files
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("relevance_dict_embed.json"),
        help="Output JSON file for intermediate rankings (original format)"
    )
    parser.add_argument(
        "--output-eval",
        type=Path,
        default=Path("relevance_dict_embed_eval.json"),
        help="Output JSON file for evaluation rankings (transformed format)"
    )
    
    # Model configuration
    parser.add_argument("--model-name", type=str, required=True, help="Embedding model identifier or checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for embedding workers")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of embedding worker instances")
    parser.add_argument("--max-model-len", type=int, default=4000, help="Maximum sequence length for the model")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum concurrent sequences per worker")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory allocated to embedding workers",
    )
    
    # Batch sizes and ranking
    parser.add_argument("--query-batch-size", type=int, default=256, help="Batch size for query embedding calls")
    parser.add_argument("--key-batch-size", type=int, default=128, help="Batch size for key embedding calls")
    parser.add_argument("--top-k", type=int, default=100, help="Number of keys to keep per query")
    
    # Other options
    parser.add_argument("--no-normalize", action="store_true", help="Skip L2 normalisation before scoring")
    parser.add_argument("--similarity-device", type=str, default="cpu", help="Device used for similarity (e.g. cpu or cuda)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("BEIR Embedding Evaluation Pipeline")
    logger.info("=" * 60)
    
    # Step 0: Load dataset (manual paths take precedence)
    if args.queries and args.corpus and args.qrels:
        # Manual mode: load individual files
        logger.info("Using manual file paths:")
        logger.info("  Queries: %s", args.queries)
        logger.info("  Corpus: %s", args.corpus)
        logger.info("  Qrels: %s", args.qrels)
        
        # Load using the new load_dataset function (simpler approach)
        # We'll create a temporary dict structure
        queries_dict = {}
        with open(args.queries, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    queries_dict[data['_id']] = data['text']
        
        corpus_dict = {}
        with open(args.corpus, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    title = data.get('title', '')
                    text = data.get('text', '')
                    combined = f"{title} {text}".strip() if title else text
                    corpus_dict[data['_id']] = combined
        
        from collections import defaultdict
        qrels_dict = defaultdict(dict)
        with open(args.qrels, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        qid, docid, rel = parts
                        if rel == 'score':
                            continue
                        qrels_dict[qid][docid] = int(rel)
                except (IndexError, ValueError):
                    continue
        
        queries, corpus, qrels = queries_dict, corpus_dict, dict(qrels_dict)
        
    elif args.dataset_path:
        # Auto-download mode: use dataset path
        logger.info("Loading dataset from: %s", args.dataset_path)
        queries, corpus, qrels = load_dataset(args.dataset_path, args.split)
    else:
        raise ValueError(
            "Must specify either:\n"
            "  1. --queries, --corpus, and --qrels (manual mode), OR\n"
            "  2. --dataset-path (auto-download mode)"
        )
    
    # Step 1: Convert to list format
    logger.info("\nStep 1: Converting to numeric IDs...")
    query_texts, query_ids, query_mapping, corpus_texts, corpus_ids, corpus_mapping, positive_pairs = convert_to_lists(
        queries, corpus, qrels
    )
    
    # Step 2: Compute embeddings and rankings
    logger.info("\nStep 2: Computing embeddings and rankings...")
    indices, scores = build_relevance_dict(
        query_texts,
        corpus_texts,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        query_batch_size=args.query_batch_size,
        key_batch_size=args.key_batch_size,
        top_k=args.top_k,
        normalize=not args.no_normalize,
        similarity_device=args.similarity_device,
    )
    
    if len(indices) != len(query_ids) or len(scores) != len(query_ids):
        raise RuntimeError("Mismatch between embedded queries and loaded query ids.")
    
    # Step 3: Create intermediate output (original format)
    logger.info("\nStep 3: Creating intermediate output...")
    intermediate_output = create_intermediate_output(
        query_ids,
        indices,
        scores,
        corpus_ids,
        positive_pairs,
    )
    
    # Step 4: Save intermediate output
    logger.info("\nStep 4: Saving intermediate output...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(intermediate_output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    logger.info("✓ Saved intermediate output to %s", args.output)
    
    # Step 5: Transform to evaluation format
    logger.info("\nStep 5: Transforming to evaluation format...")
    eval_output = transform_to_eval_format(intermediate_output)
    
    # Step 6: Save evaluation output
    logger.info("\nStep 6: Saving evaluation output...")
    args.output_eval.parent.mkdir(parents=True, exist_ok=True)
    args.output_eval.write_text(
        json.dumps(eval_output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    logger.info("✓ Saved evaluation output to %s", args.output_eval)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Complete!")
    logger.info("  Queries processed: %d", len(intermediate_output))
    logger.info("  Intermediate output: %s", args.output)
    logger.info("  Evaluation output: %s", args.output_eval)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()