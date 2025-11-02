import os
import json
import logging
import pathlib
import time
import argparse
from tqdm.autonotebook import tqdm
from typing import List, Dict, Union

from beir.retrieval.evaluation import EvaluateRetrieval

from sentence_transformers import CrossEncoder

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def load_corpus_from_csv(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Loads a corpus CSV into BEIR format."""
    logging.info(f"Loading corpus from CSV: {corpus_path}...")
    df = pd.read_csv(corpus_path)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Corpus CSV must have 'id' and 'text' columns.")
    
    corpus = {}
    for _, row in df.iterrows():
        doc_id = str(row['id'])
        corpus[doc_id] = {
            "text": str(row['text']),
            "title": str(row.get('title', ''))
        }
    logging.info(f"Loaded {len(corpus)} documents from corpus.")
    return corpus

def load_queries_from_csv(queries_path: str) -> Dict[str, str]:
    """Loads a queries CSV into BEIR format."""
    logging.info(f"Loading queries from CSV: {queries_path}...")
    df = pd.read_csv(queries_path)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Queries CSV must have 'id' and 'text' columns.")
    
    queries = {}
    for _, row in df.iterrows():
        query_id = str(row['id'])
        queries[query_id] = str(row['text'])
    logging.info(f"Loaded {len(queries)} aqueries.")
    return queries

def load_qrels_from_csv(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Loads a qrels CSV/TSV into BEIR format."""
    logging.info(f"Loading qrels from CSV/TSV: {qrels_path}...")
    sep = ',' if qrels_path.endswith('.csv') else '\t'
    df = pd.read_csv(qrels_path, sep=sep)
    
    if "query-id" not in df.columns or "corpus-id" not in df.columns or "score" not in df.columns:
        raise ValueError("Qrels file must have 'query-id', 'corpus-id', and 'score' columns.")

    qrels = {}
    for _, row in df.iterrows():
        # *** FIX ***: Check for NaN (missing values) in any critical column
        if pd.isna(row['query-id']) or pd.isna(row['corpus-id']) or pd.isna(row['score']):
            logging.warning(f"Skipping qrels row with missing data: {row}")
            continue # Skip this row

        query_id = str(row['query-id'])
        corpus_id = str(row['corpus-id'])
        score = int(row['score']) # This is now safe
        
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score
    logging.info(f"Loaded qrels for {len(qrels)} queries.")
    return qrels

def load_corpus_from_json(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Loads a corpus JSON (BEIR format)"""
    logging.info(f"Loading corpus from JSON: {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    corpus = {str(k): v for k, v in corpus.items()}
    logging.info(f"Loaded {len(corpus)} documents from corpus.")
    return corpus

def load_queries_from_json(queries_path: str) -> Dict[str, str]:
    """Loads a queries JSON (BEIR format)"""
    logging.info(f"Loading queries from JSON: {queries_path}...")
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    queries = {str(k): v for k, v in queries.items()}
    logging.info(f"Loaded {len(queries)} queries.")
    return queries

def load_qrels_from_json(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Loads a qrels JSON (BEIR format)"""
    logging.info(f"Loading qrels from JSON: {qrels_path}...")
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    qrels_string_keys = {str(k): v for k, v in qrels.items()}
    logging.info(f"Loaded qrels for {len(qrels_string_keys)} queries.")
    return qrels_string_keys

def load_corpus_from_jsonl(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Loads a corpus from a .jsonl file (standard BEIR format)."""
    logging.info(f"Loading corpus from JSONL: {corpus_path}...")
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus[str(doc['id'])] = {
                    "text": str(doc['text']),
                    "title": str(doc.get('title', ''))
                }
    logging.info(f"Loaded {len(corpus)} documents from corpus.")
    return corpus

def load_queries_from_jsonl(queries_path: str) -> Dict[str, str]:
    """Loads queries from a .jsonl file (standard BEIR format)."""
    logging.info(f"Loading queries from JSONL: {queries_path}...")
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                query = json.loads(line)
                queries[str(query['id'])] = str(query['text'])
    logging.info(f"Loaded {len(queries)} queries.")
    return queries

def load_file(file_path, file_type):
    """Generic file loader dispatch function."""
    if not file_path:
        return None
    
    logging.info(f"Loading {file_type} from {file_path}...")
    if file_path.endswith(".jsonl"):
        if file_type == "corpus":
            return load_corpus_from_jsonl(file_path)
        elif file_type == "queries":
            return load_queries_from_jsonl(file_path)
        elif file_type == "qrels":
            raise ValueError(".jsonl format is not supported for qrels. Please use .json, .csv, or .tsv.")
    elif file_path.endswith(".json"):
        if file_type == "corpus":
            return load_corpus_from_json(file_path)
        elif file_type == "queries":
            return load_queries_from_json(file_path)
        elif file_type == "qrels":
            return load_qrels_from_json(file_path)
    elif file_path.endswith(".csv") or file_path.endswith(".tsv"):
        if file_type == "corpus":
            return load_corpus_from_csv(file_path)
        elif file_type == "queries":
            return load_queries_from_csv(file_path)
        elif file_type == "qrels":
            return load_qrels_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Please use .csv, .tsv, .json, or .jsonl.")
    return None

# --- Core Reranking Function ---

def run_reranking(reranker: CrossEncoder, corpus: Dict, queries: Dict, candidate_results: Dict, top_k: int, batch_size: int = 16) -> Dict[str, Dict[str, float]]:
    """
    Reranks candidate results using a CrossEncoder.
    """
    logging.info("Starting reranking...")
    reranked_results = {}
    
    # Iterate over each query in the candidate results
    for query_id, doc_scores in tqdm(candidate_results.items(), desc="Reranking queries"):
        
        # Get the query text
        if query_id not in queries:
            logging.warning(f"Query ID {query_id} found in results but not in queries file. Skipping.")
            continue
        
        query_text = queries[query_id]
        
        # Create (query, doc_text) pairs for the reranker
        rerank_input_pairs = []
        doc_ids_for_this_query = []
        
        for doc_id in doc_scores.keys():
            if doc_id not in corpus:
                logging.warning(f"Doc ID {doc_id} found in results but not in corpus file. Skipping.")
                continue
            
            doc_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
            rerank_input_pairs.append([query_text, doc_text])
            doc_ids_for_this_query.append(doc_id)
            
        if not rerank_input_pairs:
            logging.warning(f"No valid documents to rerank for query ID {query_id}.")
            continue
            
        # Get new scores from the CrossEncoder
        new_scores = reranker.predict(rerank_input_pairs, show_progress_bar=False, batch_size=batch_size)
        
        # Map new scores back to doc_ids
        reranked_scores_for_query = {}
        for i in range(len(new_scores)):
            doc_id = doc_ids_for_this_query[i]
            reranked_scores_for_query[doc_id] = float(new_scores[i])
            
        # Sort by new score and trim to top_k
        sorted_reranked_scores = sorted(reranked_scores_for_query.items(), key=lambda item: item[1], reverse=True)
        reranked_results[query_id] = {doc_id: score for doc_id, score in sorted_reranked_scores[:top_k]}

    logging.info(f"Reranking complete. Processed {len(reranked_results)} queries.")
    return reranked_results

# --- Evaluation & Save Functions (Copied from other script) ---

def save_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Saves the retrieval results to a JSON file."""
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {output_file}")

def evaluate_results(results_file_path: str, qrels: Dict, k_values: List[int], dataset_name: str = "custom"):
    """Loads results from a file and calculates nDCG@k."""
    logging.info(f"Loading results from {results_file_path} for evaluation...")
    
    if not qrels:
        logging.error("No qrels data loaded. Cannot evaluate.")
        return
        
    try:
        with open(results_file_path, "r") as f:
            results = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load results file: {e}")
        return

    if not isinstance(results, dict):
        logging.error("Results file is not a valid JSON dictionary.")
        return
        
    evaluator = EvaluateRetrieval()
    logging.info("Calculating nDCG scores...")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    
    print("\n--- Reranking Evaluation Results ---")
    print(f"Dataset: {dataset_name}")
    print(f"Results File: {results_file_path}")
    print("--------------------------------------")
    
    header = "Metric"
    values_ndcg = "nDCG"
    values_map = "MAP"
    values_recall = "Recall"
    values_precision = "P"
    separator = "------"
    
    for k in k_values:
        header += f" | @{k:<7}"
        values_ndcg += f" | {ndcg[f'NDCG@{k}']*1.0:7.2f}"
        values_map += f" | {_map[f'MAP@{k}']*1.0:7.2f}"
        values_recall += f" | {recall[f'Recall@{k}']*1.0:7.2f}"
        values_precision += f" | {precision[f'P@{k}']*1.0:7.2f}"
        separator += " | --------"
        
    print(header)
    print(separator)
    print(values_ndcg)
    print(values_map)
    print(values_recall)
    print(values_precision)
    print("--------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Rerank and Evaluate Retrieval Results")
    
    parser.add_argument("--candidate_results_file", type=str, required=True, 
                        help="Path to the JSON results file from a first-stage retrieval (e.g., results/candidates_bge.json)")
    parser.add_argument("--corpus_file", type=str, required=True, 
                        help="Path to local corpus file (.csv, .json, or .jsonl).")
    parser.add_argument("--queries_file", type=str, required=True, 
                        help="Path to local queries file (.csv, .json, or .jsonl).")
    parser.add_argument("--qrels_file", type=str, required=True, 
                        help="Path to local qrels file (.csv, .tsv, or .json).")
    parser.add_argument("--reranker_model_name", type=str, required=True, 
                        help="Hugging Face model name for the CrossEncoder (e.g., 'zeroentropy/zerank-1')")
    parser.add_argument("--top_k", type=int, default=10, 
                        help="Number of documents to *keep* after reranking.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the output JSON file (default: results/[reranker_name].json)")
    parser.add_argument("--k_values", nargs='+', type=int, default=[1, 3, 5, 10, 100],
                        help="List of k-values for nDCG calculation.")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Set to True if the reranker model requires trusting remote code (e.g., Qwen)")
    parser.add_argument("--rerank_batch_size", type=int, default=16, 
                        help="Batch size for the reranker prediction step.")
    args = parser.parse_args()

    try:
        corpus = load_file(args.corpus_file, "corpus")
        queries = load_file(args.queries_file, "queries")
        qrels = load_file(args.qrels_file, "qrels")
        
        logging.info(f"Loading candidate results from {args.candidate_results_file}...")
        with open(args.candidate_results_file, 'r') as f:
            candidate_results = json.load(f)
        
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return
        
    if not all([corpus, queries, qrels, candidate_results]):
        logging.error("One or more input files failed to load. Aborting.")
        return
    
    model_load_args = {}
    if args.trust_remote_code:
        model_load_args['trust_remote_code'] = True
        logging.info("Using trust_remote_code=True to load model.")
        
    try:
        logging.info(f"Loading CrossEncoder model: {args.reranker_model_name}...")
        # from transformers import AutoTokenizer, AutoModelForCausalLM

        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B")
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-4B")
        
        reranker = CrossEncoder(args.reranker_model_name, trust_remote_code = True)
        logging.info("Model loaded successfully.")
        
        if reranker.tokenizer.pad_token is None or reranker.tokenizer.pad_token_id is None:
            logging.info("No pad token found. Setting pad_token = eos_token.")
            reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
        # Also update the model's config, which is what the collator often uses
        if reranker.model.config.pad_token_id is None:
            reranker.model.config.pad_token_id = reranker.tokenizer.eos_token_id
                
    except Exception as e:
        logging.error(f"Failed to load CrossEncoder model: {e}")
        return

    reranked_results = run_reranking(reranker, corpus, queries, candidate_results, args.top_k, args.rerank_batch_size)

    if not reranked_results:
        logging.error("Reranking failed to produce results.")
        return

    # --- Save & Evaluate ---
    output_file = args.output_file
    if not output_file:
        model_name_safe = args.reranker_model_name.replace("/", "_")
        output_file = f"results/reranked_{model_name_safe}.json"
    
    save_results(reranked_results, output_file)
    
    evaluate_results(output_file, qrels, args.k_values, dataset_name="Custom Data (Reranked)")

if __name__ == "__main__":
    main()
