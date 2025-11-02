import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval

DATASETS_BASE_PATH = Path("/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets")
DATASET_NAMES = [
    # "webis-touche2020",
    # "trec-covid",
    # "nfcorpus",
    "scifact",
    # "fiqa",
]

MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
# MODEL_NAME = "jinaai/jina-reranker-v3"
# MODEL_NAME = "zeroentropy/zerank-1"
# MODEL_NAME = "/workspace/Mingkwan/RMSearch/models/Pra1_1240-converted-model"
# EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Embedding model for initial retrieval
# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"  # Embedding model for initial retrieval
EMBEDDING_MODEL_NAME = "intfloat/e5-mistral-7b-instruct"  # Embedding model for initial retrieval

BATCH_SIZE = 16
TOP_K = 100  
RERANK_TOP_K = 10  
EMBEDDING_GPU_ID = 1 
RERANKER_GPU_ID = 0  
CLEAR_CACHE_BETWEEN_DATASETS = True 
# USE_PRECOMPUTED_RANKINGS = True  
# PRECOMPUTED_RANKINGS_FILE = "/workspace/Mingkwan/RMSearch/beir_out/scifact/relevance_dict_rerank1240.json"

def download_beir_dataset(dataset_path: Path, dataset_name: str):
    """Download BEIR dataset if not exists."""
    from beir import util
    
    print(f"Dataset not found at {dataset_path}")
    print(f"Downloading BEIR/{dataset_name}...")
    
    # Create parent directory
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(dataset_path.parent))
    
    print(f"Dataset downloaded to {data_path}")

 
def load_dataset(dataset_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load queries, corpus, and qrels from the dataset folder."""
    
    if not dataset_path.exists():
        dataset_name = dataset_path.name
        download_beir_dataset(dataset_path, dataset_name)
        
    print("Loading dataset...")
    # Load queries
    queries_file = dataset_path / "queries.jsonl"
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            queries[data['_id']] = data['text']
    
    # Load corpus
    corpus_file = dataset_path / "corpus.jsonl"
    corpus = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('title', '') + ' ' + data.get('text', '')
            corpus[data['_id']] = text.strip()
    
    qrels_file = dataset_path / "qrels" / "test.tsv"
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            try:
                # print(line.strip().split('\t'))
                parts= line.strip().split('\t')
                if len(parts) == 3:
                    qid, docid, rel = parts
                else:
                    print(f"Skipping line with incorrect field count: {line.strip()}")
                    continue # Skip the line    
                if rel == 'score':
                    pass
                else:
                    qrels[qid][docid] = int(rel)
            except IndexError as e:
                continue            
    
    print(f"Loaded {len(queries)} queries, {len(corpus)} documents, {len(qrels)} qrels")
    
     ### Debug: Check for mismatches
    qrel_qids = set(qrels.keys())
    query_qids = set(queries.keys())
    missing_queries = qrel_qids - query_qids
    if missing_queries:
        print(f"WARNING: {len(missing_queries)} query IDs in qrels but not in queries")
        print(f"  Example missing: {list(missing_queries)[:5]}")
    
    # Check corpus coverage
    all_doc_ids_in_qrels = set()
    for qid, docs in qrels.items():
        all_doc_ids_in_qrels.update(docs.keys())
    missing_docs = all_doc_ids_in_qrels - set(corpus.keys())
    if missing_docs:
        print(f"WARNING: {len(missing_docs)} document IDs in qrels but not in corpus")
        print(f"  Example missing: {list(missing_docs)[:5]}")
        
    return queries, corpus, dict(qrels)

def load_embedding_model(model_name: str, device: str = 'cuda'):
    """Load the embedding model and tokenizer for initial retrieval."""
    print(f"Loading embedding model: {model_name}")
    
    from sentence_transformers import SentenceTransformer
    embedding_device = f'cuda:{EMBEDDING_GPU_ID}' if torch.cuda.is_available() else 'cpu'
    # embedding_device = "cpu"
    model = SentenceTransformer(model_name, device=embedding_device)
    print(f"Embedding model loaded on {embedding_device}")
    return model

def load_reranker(model_name: str, device: str = 'cuda'):
    """Load the reranker model and tokenizer."""
    print(f"Loading reranker model: {model_name}")
    
    if torch.cuda.is_available():
        reranker_device = f'cuda:{RERANKER_GPU_ID}'
        print(f"Loading reranker on {reranker_device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    # Set pad_token_id in model config
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        
    model.to(reranker_device)
    model.eval()
    
    # print(f"Model loaded on {device}")
    return tokenizer, model, reranker_device

def load_precomputed_rankings(precomputed_file: Path, corpus: Dict, top_k: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Load pre-computed rankings from a JSON file.
    
    Args:
        precomputed_file: Path to the JSON file with pre-computed rankings
        corpus: Corpus dictionary to validate doc IDs
        top_k: Number of top documents to use
    
    Returns:
        initial_rankings: Dict[query_id, Dict[doc_id, score]]
    """
    print(f"Loading pre-computed rankings from: {precomputed_file}")
    
    with open(precomputed_file, 'r') as f:
        data = json.load(f)
    
    initial_rankings = {}
    
    for item in data:
        query_id = str(item['query_id'])
        pre_key_ids = item['pre_key_ids'][:top_k]  # Take top-k
        
        # Convert to string IDs and create dummy scores (decreasing)
        results_dict = {}
        for i, doc_idx in enumerate(pre_key_ids):
            doc_id = str(doc_idx)
            # Check if doc exists in corpus
            # if doc_id in corpus:
            results_dict[doc_id] = float(top_k - i)  # Higher rank = higher score
        # print(f"results_dict: {results_dict}")
        if results_dict:  # Only add if we have valid documents
            initial_rankings[query_id] = results_dict
    
    print(f"Loaded rankings for {len(initial_rankings)} queries")
    return initial_rankings


def get_initial_ranking(
    queries: Dict, 
    corpus: Dict, 
    qrels: Dict, 
    embedding_model,
    dataset_path: Path,
    top_k: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Get initial ranking using dense retrieval with embedding model.
    Caches embeddings to avoid recomputing.
    
    Args:
        queries: Query ID to query text mapping
        corpus: Document ID to document text mapping
        qrels: Query relevance judgments
        embedding_model: SentenceTransformer model for encoding
        dataset_path: Path to dataset for caching embeddings
        top_k: Number of top documents to retrieve
    
    Returns:
        results: Dict[query_id, Dict[doc_id, score]]
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("Getting initial rankings with embedding model...")
    
    # Create cache directory
    cache_dir = dataset_path / "embeddings_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache filenames based on model name
    model_name_safe = embedding_model._model_card_vars.get('model_name', EMBEDDING_MODEL_NAME).replace('/', '_')
    doc_embeddings_file = cache_dir / f"doc_embeddings_{model_name_safe}.npy"
    doc_ids_file = cache_dir / f"doc_ids_{model_name_safe}.json"
    query_embeddings_file = cache_dir / f"query_embeddings_{model_name_safe}.npy"
    query_ids_file = cache_dir / f"query_ids_{model_name_safe}.json"
    
    # Load or compute document embeddings
    if doc_embeddings_file.exists() and doc_ids_file.exists():
        print("Loading cached document embeddings...")
        doc_embeddings = np.load(doc_embeddings_file)
        with open(doc_ids_file, 'r') as f:
            doc_ids = json.load(f)
        print(f"Loaded {len(doc_ids)} document embeddings from cache")
    else:
        print("Computing document embeddings...")
        doc_ids = sorted(list(corpus.keys()))
        doc_texts = [corpus[doc_id] for doc_id in doc_ids]
        
        # Encode in batches to avoid memory issues
        doc_embeddings = embedding_model.encode(
            doc_texts, 
            batch_size=8, 
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Save to cache
        np.save(doc_embeddings_file, doc_embeddings)
        with open(doc_ids_file, 'w') as f:
            json.dump(doc_ids, f)
        print(f"Saved {len(doc_ids)} document embeddings to cache")
    
    # Load or compute query embeddings
    sorted_qids = sorted(qrels.keys())
    
    if query_embeddings_file.exists() and query_ids_file.exists():
        print("Loading cached query embeddings...")
        query_embeddings = np.load(query_embeddings_file)
        with open(query_ids_file, 'r') as f:
            cached_query_ids = json.load(f)
        
        # Check if cached queries match current queries
        if cached_query_ids == sorted_qids:
            print(f"Loaded {len(cached_query_ids)} query embeddings from cache")
        else:
            print("Cached queries don't match, recomputing...")
            query_embeddings = None
    else:
        query_embeddings = None
    
    # Compute query embeddings if not cached or cache invalid
    if query_embeddings is None:
        print("Computing query embeddings...")
        query_texts = [queries[qid] for qid in sorted_qids if qid in queries]
        valid_qids = [qid for qid in sorted_qids if qid in queries]
        
        query_embeddings = embedding_model.encode(
            query_texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Save to cache
        np.save(query_embeddings_file, query_embeddings)
        with open(query_ids_file, 'w') as f:
            json.dump(valid_qids, f)
        print(f"Saved {len(valid_qids)} query embeddings to cache")
        sorted_qids = valid_qids
    
    initial_rankings = {}
    
    print("Computing similarities...")
    for i, qid in enumerate(tqdm(sorted_qids, desc="Initial retrieval")):
        if qid not in queries:
            continue
        
        # Get query embedding
        query_embedding = query_embeddings[i]
        
        # Compute cosine similarity (already normalized, so just dot product)
        similarities = np.dot(doc_embeddings, query_embedding)
        
        # Get top-k documents
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results dict with scores
        results_dict = {}
        for idx in top_k_indices:
            doc_id = doc_ids[idx]
            score = float(similarities[idx])
            results_dict[doc_id] = score
        
        initial_rankings[qid] = results_dict
    # print(f"initial ranikngs: {initial_rankings}")
    return initial_rankings

def rerank_documents(
    queries: Dict,
    corpus: Dict,
    initial_rankings: Dict[str, Dict[str, float]],
    tokenizer,
    model,
    device: str = 'cuda',
    batch_size: int = 32,
    top_k: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Rerank documents using the reranker model.
    
    Returns:
        results: Dict[query_id, Dict[doc_id, score]] in BEIR format
    """
    print("Reranking documents...")
    
    reranked_results = {}
    
    for qid, doc_scores in tqdm(initial_rankings.items(), desc="Reranking"):
        if qid not in queries:
            continue
            
        query = queries[qid]
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Prepare pairs
        pairs = []
        valid_doc_ids = []
        
        for doc_id, _ in sorted_docs:
            if doc_id in corpus:
                pairs.append([query, corpus[doc_id]])
                valid_doc_ids.append(doc_id)
        
        if not pairs:
            continue
        
        # Score in batches
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                inputs = tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)
                
                outputs = model(**inputs)
                
                logits = outputs.logits.cpu()
                
                if logits.dim() == 2:
                    batch_scores = logits[:, 0].tolist()
                else:
                    batch_scores = [logits[0].item()]
                    
                if isinstance(batch_scores, float):
                    all_scores.append(batch_scores)
                else:
                    all_scores.extend(batch_scores)

        reranked_results[qid] = {
            doc_id: float(score) for doc_id, score in zip(valid_doc_ids, all_scores)
        }

    return reranked_results

def evaluate_single_dataset(
    dataset_name: str,
    dataset_path: Path,
    embedding_model,
    tokenizer,
    reranker_model,
    reranker_device: str
):
    """
    Evaluate a single dataset with the reranker model.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset
        embedding_model: Loaded embedding model
        tokenizer: Loaded tokenizer
        reranker_model: Loaded reranker model
        device: Device to run on
    """
    print(f"\n{'='*80}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load dataset
    queries, corpus, qrels = load_dataset(dataset_path)
    
    # if USE_PRECOMPUTED_RANKINGS and PRECOMPUTED_RANKINGS_FILE:
    #     precomputed_path = Path(PRECOMPUTED_RANKINGS_FILE)
    #     if not precomputed_path.is_absolute():
    #         # If relative path, make it relative to dataset path
    #         precomputed_path = dataset_path / precomputed_path
        
    #     if precomputed_path.exists():
    #         initial_rankings = load_precomputed_rankings(precomputed_path, corpus, TOP_K)
    #     else:
    #         print(f"Warning: Pre-computed file not found at {precomputed_path}")
    #         print("Falling back to embedding model...")
    #         initial_rankings = get_initial_ranking(queries, corpus, qrels, embedding_model, dataset_path, TOP_K)
    # else:
    #     initial_rankings = get_initial_ranking(queries, corpus, qrels, embedding_model, dataset_path, TOP_K)
    # Get initial rankings
    initial_rankings = get_initial_ranking(queries, corpus, qrels, embedding_model, dataset_path, TOP_K)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Rerank documents
    reranked_results = rerank_documents(
        queries,
        corpus,
        initial_rankings,
        tokenizer,
        reranker_model,
        reranker_device,
        BATCH_SIZE,
        RERANK_TOP_K
    )
    
    # Evaluate using BEIR
    print("\nEvaluating with BEIR...")
    evaluator = EvaluateRetrieval()
    
    sample_qid = list(reranked_results.keys())[0] if reranked_results else None
    if sample_qid:
        print(f"\nDebug - Sample query: {sample_qid}")
        print(f"  Reranked docs: {len(reranked_results[sample_qid])}")
        if sample_qid in qrels:
            print(f"  Relevant docs in qrels: {len(qrels[sample_qid])}")
            # Check if any relevant docs are in results
            relevant_in_results = set(qrels[sample_qid].keys()) & set(reranked_results[sample_qid].keys())
            print(f"  Relevant docs in reranked results: {len(relevant_in_results)}")
        else:
            print(f"  WARNING: Query {sample_qid} not found in qrels!")
            
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels,
        reranked_results,
        [1, 3, 5, 10]
    )
    
    print(f"\n{'='*60}")
    print(f"BEIR Evaluation Results for {dataset_name}:")
    print(f"{'='*60}")
    print("\nNDCG Scores:")
    for k, score in ndcg.items():
        print(f"  {k}: {score:.4f}")
    
    print("\nMAP Scores:")
    for k, score in _map.items():
        print(f"  {k}: {score:.4f}")
    
    print("\nRecall Scores:")
    for k, score in recall.items():
        print(f"  {k}: {score:.4f}")
    
    print("\nPrecision Scores:")
    for k, score in precision.items():
        print(f"  {k}: {score:.4f}")
    
    print(f"{'='*60}")
    print(f"\n🎯 Main Result: NDCG@10 = {ndcg['NDCG@10']:.4f}")
    print(f"{'='*60}")
    
    # Save results
    reranker_name_safe = MODEL_NAME.replace('/', '_')
    embedding_name_safe = EMBEDDING_MODEL_NAME
    # .replace('/', '_')
    output_file = dataset_path / f"reranker_results_{reranker_name_safe}_e5.json"
    results_to_save = {
        'dataset': dataset_name,
        'embedding_model': EMBEDDING_MODEL_NAME,
        'reranker_model': MODEL_NAME,
        'ndcg': ndcg,
        'map': _map,
        'recall': recall,
        'precision': precision,
        'num_queries': len(reranked_results)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_to_save

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    
    print(f"\n{'='*80}")
    print(f"Multi-Dataset Reranker Evaluation")
    print(f"{'='*80}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Reranker Model: {MODEL_NAME}")
    print(f"Datasets: {', '.join(DATASET_NAMES)}")
    print(f"{'='*80}\n")
    
    # if USE_PRECOMPUTED_RANKINGS:
    #     print("Using pre-computed rankings - skipping embedding model loading")
    #     embedding_model = None
    # else:
    #     embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME, device)
        
    tokenizer, reranker_model, reranker_device = load_reranker(MODEL_NAME)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    
    all_results = {}
    
    for dataset_name in DATASET_NAMES:
        dataset_path = DATASETS_BASE_PATH / dataset_name
        
        try:
            results = evaluate_single_dataset(
                dataset_name,
                dataset_path,
                embedding_model,
                tokenizer,
                reranker_model,
                reranker_device
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\n❌ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - All Datasets")
    print(f"{'='*80}\n")
    
    for dataset_name, results in all_results.items():
        ndcg_10 = results['ndcg'].get('NDCG@10', 0.0)
        print(f"{dataset_name:20s} - NDCG@10: {ndcg_10:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete for {len(all_results)}/{len(DATASET_NAMES)} datasets")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()