import os
import json
import logging
import pathlib
import time
import argparse
from tqdm.autonotebook import trange
from typing import List, Dict, Union

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DenseRetrieval
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT

from sentence_transformers import SentenceTransformer
import openai
import google.generativeai as genai

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

class RetrievalModel:
    """
    Base class for a retrieval model.
    Requires embed_corpus and embed_queries methods.
    """
    def __init__(self, model_path: str = None, **kwargs):
        self.model_path = model_path
        self.model = None

    def embed_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128) -> Union[List[List[float]], None]:
        """Embeds a list of documents."""
        raise NotImplementedError

    def embed_queries(self, queries: List[str], batch_size: int = 128) -> Union[List[List[float]], None]:
        """Embeds a list of query strings."""
        raise NotImplementedError

    def get_model(self):
        """
        Returns the underlying model object.
        For HF, this is the SentenceTransformer object.
        For API models, this is self, as self implements
        the .encode_corpus and .encode_queries methods.
        """
        return self.model


# --- Hugging Face Model ---
# class HuggingFaceModel(RetrievalModel):
#     """
#     Wrapper for SentenceTransformer models from Hugging Face.
#     These models run locally.
#     """
#     def __init__(self, model_path: str = "thenlper/gte-large", **kwargs):
#         super().__init__(model_path)
#         logging.info(f"Loading Hugging Face model: {model_path}")
#         # Initialize the SentenceTransformer model
#         self.model = SentenceTransformer(model_path)
#         logging.info("Model loaded successfully.")
    
#     # Note: We don't need to implement embed_corpus or embed_queries
#     # because the SentenceTransformer object returned by get_model()
#     # already has .encode() methods that BEIR knows how to use.

#     def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128, **kwargs) -> List[List[float]]:
#         logging.info(f"Embedding corpus with {len(corpus)} documents using {self.model_path}...")
#         sentences = [(doc.get("title", "") + " " + doc.get("text", "")).strip() for doc in corpus]
#         return self.sbert_model.encode(sentences, batch_size=batch_size, show_progress_bar=True, **kwargs)

#     # BEIR will call this method
#     def encode_queries(self, queries: List[str], batch_size: int = 128, **kwargs) -> List[List[float]]:
#         logging.info(f"Embedding {len(queries)} queries using {self.model_path}...")
#         return self.sbert_model.encode(queries, batch_size=batch_size, show_progress_bar=True, **kwargs)
        
#     def embed_corpus(self, *args, **kwargs):
#         pass # Not needed, handled by self.model

#     def embed_queries(self, *args, **kwargs):
#         pass # Not needed, handled by self.model


class OpenAIModel(RetrievalModel):
    """
    Wrapper for OpenAI Embedding API.
    Requires an API key.
    """
    def __init__(self, model_path: str = "text-embedding-3-large", api_key: str = None, **kwargs):
        super().__init__(model_path)
        if not api_key:
            raise ValueError("OpenAI API key is required. Set via --openai_api_key or OPENAI_API_KEY env var.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = self 

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Helper to embed a single batch with retry logic."""
        max_retries = 5
        delay = 5  
        for attempt in range(max_retries):
            try:
              
                res = self.client.embeddings.create(model=self.model_path, input=texts)
                return [item.embedding for item in res.data]
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
                logging.warning(f"API Error: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  
        raise Exception(f"Failed to get embeddings from OpenAI after {max_retries} retries.")

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128, **kwargs) -> List[List[float]]:
        logging.info(f"Embedding corpus with {len(corpus)} documents using OpenAI...")
        sentences = [(doc.get("title", "") + " " + doc.get("text", "")).strip() for doc in corpus]
        all_embeddings = []
        
        for i in trange(0, len(sentences), batch_size, desc="Embedding Corpus (OpenAI)"):
            batch = sentences[i:i+batch_size]
            all_embeddings.extend(self._embed_batch(batch, "retrieval_document"))
        return all_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 128, **kwargs) -> List[List[float]]:
        logging.info(f"Embedding {len(queries)} queries using OpenAI...")
        all_embeddings = []
        for i in trange(0, len(queries), batch_size, desc="Embedding Queries (OpenAI)"):
            batch = queries[i:i+batch_size]
            all_embeddings.extend(self._embed_batch(batch, "retrieval_query"))
        return all_embeddings
    
    def embed_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128, **kwargs):
        return self.encode_corpus(corpus, batch_size=batch_size, **kwargs)
    
    def embed_queries(self, queries: List[str], batch_size: int = 128, **kwargs):
        return self.encode_queries(queries, batch_size=batch_size, **kwargs)


class GoogleModel(RetrievalModel):
    """
    Wrapper for Google Gemini Embedding API.
    Requires an API key.
    """
    def __init__(self, model_path: str = "models/text-embedding-004", api_key: str = None, **kwargs):
        super().__init__(model_path)
        if not api_key:
            raise ValueError("Google API key is required. Set via --google_api_key or GOOGLE_API_KEY env var.")
        genai.configure(api_key=api_key)
        self.model = self # BEIR's DenseRetrieval will call our methods

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Helper to embed a single batch with retry logic."""
        max_retries = 5
        delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                # Use batch_embed_texts for efficiency
                result = genai.embed_content(
                    model=self.model_path,
                    content=texts,
                    task_type=task_type # e.g., "retrieval_document" or "retrieval_query"
                )
                return result['embedding']
            except Exception as e:
                logging.warning(f"API Error: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        raise Exception(f"Failed to get embeddings from Google after {max_retries} retries.")

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 100, **kwargs) -> List[List[float]]:
        # Google's batch API has a limit of 100
        batch_size = min(batch_size, 100)
        logging.info(f"Embedding corpus with {len(corpus)} documents using Google...")
        sentences = [(doc.get("title", "") + " " + doc.get("text", "")).strip() for doc in corpus]
        all_embeddings = []

        for i in trange(0, len(sentences), batch_size, desc="Embedding Corpus (Google)"):
            batch = sentences[i:i+batch_size]
            all_embeddings.extend(self._embed_batch(batch, "retrieval_document"))
        return all_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 100, **kwargs) -> List[List[float]]:
        # Google's batch API has a limit of 100
        batch_size = min(batch_size, 100)
        logging.info(f"Embedding {len(queries)} queries using Google...")
        all_embeddings = []
        for i in trange(0, len(queries), batch_size, desc="Embedding Queries (Google)"):
            batch = queries[i:i+batch_size]
            all_embeddings.extend(self._embed_batch(batch, "retrieval_query"))
        return all_embeddings

    # Implement embed_corpus/embed_queries as aliases for BEIR
    def embed_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 100, **kwargs):
        return self.encode_corpus(corpus, batch_size=batch_size, **kwargs)
    
    def embed_queries(self, queries: List[str], batch_size: int = 100, **kwargs):
        return self.encode_queries(queries, batch_size=batch_size, **kwargs)


def download_dataset(dataset_name: str = "scifact") -> (Dict, Dict, Dict):
    """Downloads and loads the specified BEIR dataset."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    logging.info(f"Loaded BEIR/{dataset_name} test split: {len(corpus)} documents, {len(queries)} queries.")
    return corpus, queries, qrels

def load_corpus_from_csv(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Loads a corpus CSV into BEIR format."""
    logging.info(f"Loading corpus from {corpus_path}...")
    df = pd.read_csv(corpus_path)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Corpus CSV must have 'id' and 'text' columns.")
    
    corpus = {}
    for _, row in df.iterrows():
        doc_id = str(row['id'])
        corpus[doc_id] = {
            "text": str(row['text']),
            "title": str(row.get('title', '')) # Add title if it exists, else empty string
        }
    logging.info(f"Loaded {len(corpus)} documents from corpus.")
    return corpus

def load_queries_from_csv(queries_path: str) -> Dict[str, str]:
    """Loads a queries CSV into BEIR format."""
    logging.info(f"Loading queries from {queries_path}...")
    df = pd.read_csv(queries_path)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Queries CSV must have 'id' and 'text' columns.")
    
    queries = {}
    for _, row in df.iterrows():
        query_id = str(row['id'])
        queries[query_id] = str(row['text'])
    logging.info(f"Loaded {len(queries)} queries.")
    return queries

def load_qrels_from_csv(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Loads a qrels CSV/TSV into BEIR format."""
    logging.info(f"Loading qrels from {qrels_path}...")
    # Autodetect separator (CSV or TSV)
    sep = ',' if qrels_path.endswith('.csv') else '\t'
    df = pd.read_csv(qrels_path, sep=sep)
    
    if "query-id" not in df.columns or "corpus-id" not in df.columns or "score" not in df.columns:
        raise ValueError("Qrels file must have 'query_id', 'corpus_id', and 'score' columns.")

    qrels = {}
    for _, row in df.iterrows():
        query_id = str(row['query-id'])
        corpus_id = str(row['corpus-id'])
        score = int(row['score'])
        # print(F"QID: {query_id}")
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score
    logging.info(f"Loaded qrels for {len(qrels)} queries.")
    return qrels

def load_corpus_from_json(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Loads a corpus JSON (BEIR format)"""
    logging.info(f"Loading corpus from JSON: {corpus_path}...")
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    # Ensure keys are strings, just in case
    corpus = {str(k): v for k, v in corpus.items()}
    
    if not isinstance(next(iter(corpus.values()), None), dict):
        raise ValueError("Corpus JSON format is incorrect. Expected: {'doc_id': {'text': '...', 'title': '...'}}")
        
    logging.info(f"Loaded {len(corpus)} documents from corpus.")
    return corpus

def load_queries_from_json(queries_path: str) -> Dict[str, str]:
    """Loads a queries JSON (BEIR format)"""
    logging.info(f"Loading queries from JSON: {queries_path}...")
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    # Ensure keys are strings
    queries = {str(k): v for k, v in queries.items()}

    if not isinstance(next(iter(queries.values()), None), str):
        raise ValueError("Queries JSON format is incorrect. Expected: {'query_id': 'query text'}")

    logging.info(f"Loaded {len(queries)} queries.")
    return queries

def load_corpus_from_jsonl(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Loads a corpus from a .jsonl file (standard BEIR format)."""
    logging.info(f"Loading corpus from JSONL: {corpus_path}...")
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus[str(doc['_id'])] = {
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
                queries[str(query['_id'])] = str(query['text'])
    logging.info(f"Loaded {len(queries)} queries.")
    return queries

def run_retrieval(model_wrapper, corpus: Dict, queries: Dict, top_k: int) -> Dict[str, Dict[str, float]]:
    """
    Performs dense retrieval using the provided model.
    """
    
    # A crude but effective check: corpus items are dicts, query items are strings.
    if (isinstance(corpus, dict) and isinstance(queries, dict)):
        # Get first item to check structure
        first_corpus_val = next(iter(corpus.values()), None)
        first_query_val = next(iter(queries.values()), None)

        # if corpus values are strings and query values are dicts, they are swapped
        if isinstance(first_corpus_val, str) and isinstance(first_query_val, dict):
            logging.warning("Corpus and Queries arguments appear to be swapped. Correcting.")
            corpus, queries = queries, corpus # Swap them back
        elif not (isinstance(first_corpus_val, dict) and isinstance(first_query_val, str)):
             # This is the expected case, but we log if it's not, just in case.
             if not (isinstance(first_corpus_val, dict) and isinstance(first_query_val, str)):
                 logging.warning(f"Unexpected input types: corpus values are {type(first_corpus_val)}, query values are {type(first_query_val)}")
             
    elif not isinstance(queries, dict):
        logging.error(f"FATAL: 'queries' argument is a {type(queries)}, but must be a dict.")
        logging.error("Please ensure you are passing the queries dictionary (query_id: query_text) as the 3rd argument.")
        return {}
    elif not isinstance(corpus, dict):
        logging.error(f"FATAL: 'corpus' argument is a {type(corpus)}, but must be a dict.")
        logging.error("Please ensure you are passing the corpus dictionary (doc_id: doc_text) as the 2nd argument.")
        return {}
    beir_model_wrapper = DenseRetrieval(model_wrapper, batch_size=128)
    retriever = EvaluateRetrieval(beir_model_wrapper, score_function="cos_sim")
    retriever.top_k = top_k
    
    logging.info("Starting retrieval...")
    start_time = time.time()
    
    results = retriever.retrieve(corpus, queries)
    
    end_time = time.time()
    logging.info(f"Retrieval complete in {end_time - start_time:.2f} seconds.")
    if not results:
        logging.error("Retrieval returned no results. Aborting.")
        return {}
        
    first_query_id = list(queries.keys())[0]
    if first_query_id not in results:
        logging.error(f"Results dict missing query ID: {first_query_id}. Results format might be incorrect.")
        return {}
        
    logging.info(f"Generated results for {len(results)} queries.")
    return results


def save_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Saves the retrieval results to a JSON file."""
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {output_file}")


def evaluate_results(results_file_path: str, qrels: Dict, k_values: List[int], dataset_name: str = "scifact"):
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
        
    # Initialize BEIR evaluation
    evaluator = EvaluateRetrieval()
    
    # Calculate scores
    logging.info("Calculating nDCG scores...")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    
    print("\n--- Retrieval Evaluation Results ---")
    print(f"Dataset: {dataset_name}")
    print(f"Results File: {results_file_path}")
    print("--------------------------------------")
    
    header = "Metric"
    values = "Score"
    separator = "------"
    
    for k in k_values:
        header += f" | nDCG@{k:<4}"
        values += f" | {ndcg[f'NDCG@{k}']*100:6.2f}"
        separator += " | --------"
        
    print(header)
    print(separator)
    print(values)
    print("--------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Run Retrieval Evaluation on BEIR/scifact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- "run" command ---
    run_parser = subparsers.add_parser("run", help="Run retrieval and save results")
    run_parser.add_argument("--model", type=str, required=True, choices=["hf", "openai", "google"],
                            help="Model type to use for retrieval.")
    run_parser.add_argument("--hf_model_name", type=str, default="thenlper/gte-large",
                            help="Hugging Face model name (e.g., 'thenlper/gte-large')")
    run_parser.add_argument("--openai_model_name", type=str, default="text-embedding-3-large",
                            help="OpenAI embedding model name")
    run_parser.add_argument("--google_model_name", type=str, default="models/text-embedding-004",
                            help="Google embedding model name")
    run_parser.add_argument("--top_k", type=int, default=100, help="Number of documents to retrieve per query")
    run_parser.add_argument("--output_file", type=str, default= "/workspace/Mingkwan/RMSearch/beir_out/scifact/openai/relevant_dict_hf.json",
                            help="Path to save the output JSON file (default: results/[model_name].json)")
    run_parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                            help="OpenAI API key")
    run_parser.add_argument("--google_api_key", type=str, default=os.environ.get("GOOGLE_API_KEY"),
                            help="Google API key")
    # New CSV arguments
    run_parser.add_argument("--corpus_file", type=str, default="/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/corpus.jsonl", help="Path to local corpus CSV file. Overrides BEIR download.")
    run_parser.add_argument("--queries_file", type=str, default="/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/queries.jsonl", help="Path to local queries CSV file. Overrides BEIR download.")
    run_parser.add_argument("--qrels_file", type=str, default="/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/qrels/test.tsv", help="Path to local qrels CSV/TSV file. Overrides BEIR download.")

    # --- "evaluate" command ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an existing results JSON file")
    eval_parser.add_argument("--results_file", type=str, required=True,
                             help="Path to the results JSON file to evaluate.")
    eval_parser.add_argument("--qrels_file", type=str, default="/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/qrels/test.tsv", help="Path to local qrels CSV/TSV file. Overrides BEIR download.")
    eval_parser.add_argument("--k_values", nargs='+', type=int, default=[1, 3, 5, 10],
                             help="List of k-values for nDCG calculation.")
    args = parser.parse_args()

    corpus, queries, qrels = None, None, None
    dataset_name = "BEIR/scifact" 

    def load_file(file_path, file_type):
        if not file_path:
            return None
        
        logging.info(f"Loading {file_type} from {file_path}...")
        if file_path.endswith(".json"):
            if file_type == "corpus":
                return load_corpus_from_json(file_path)
            elif file_type == "queries":
                return load_queries_from_json(file_path)
            # elif file_type == "qrels":
            #     return load_qrels_from_json(file_path)
        elif file_path.endswith(".csv") or file_path.endswith(".tsv"):
            if file_type == "corpus":
                return load_corpus_from_csv(file_path)
            elif file_type == "queries":
                return load_queries_from_csv(file_path)
            elif file_type == "qrels":
                return load_qrels_from_csv(file_path)
        elif file_path.endswith(".jsonl"):
            if file_type == "corpus":
                return load_corpus_from_jsonl(file_path)
            elif file_type == "queries":
                return load_queries_from_jsonl(file_path)
            elif file_type == "qrels":
                return load_qrels_from_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}. Please use .csv, .tsv, or .json.")
        return None
    # --- Load Data ---
    try:
        # Check if we are using local files (applies to both 'run' and 'evaluate')
        qrels_file_path = args.qrels_file if hasattr(args, 'qrels_file') else None

        if args.command == "run":
            # For "run", we need corpus, queries, and qrels
            if args.corpus_file and args.queries_file and qrels_file_path:
                logging.info("Loading local data from provided files...")
                corpus = load_file(args.corpus_file, "corpus")
                queries = load_file(args.queries_file, "queries")
                # corpus = load_corpus_from_csv(args.corpus_file)
                # queries = load_queries_from_csv(args.queries_file)

                try:
                    corpus1, queries1, qrels = download_dataset(dataset_name="scifact")
                except Exception as e:
                    logging.error(f"Failed to load data: {e}")
                    return
                # qrels = load_qrels_from_csv(qrels_file_path)
                dataset_name = "Custom CSV Data"
            elif args.corpus_file or args.queries_file or qrels_file_path:
                # User provided some but not all files for a 'run'
                logging.error("For a local 'run', --corpus_file, --queries_file, and --qrels_file are all required.")
                return
            else:
                # No local files provided for 'run', download BEIR
                logging.info("No local files provided, downloading BEIR/scifact...")
                corpus, queries, qrels = download_dataset(dataset_name="scifact")
                dataset_name = "BEIR/scifact"
        
        elif args.command == "evaluate":
            try:
                # corpus1, queries1, qrels = download_dataset(dataset_name="scifact")
                
            # For "evaluate", we only need qrels
            
                if qrels_file_path:
                    logging.info(f"Loading local qrels file from {qrels_file_path}...")
                    qrels = load_qrels_from_csv(qrels_file_path)
                    dataset_name = "Custom CSV Data"
            # else:
            #     # No local qrels provided for 'evaluate', download BEIR
            #     logging.info("No local qrels file provided, downloading BEIR/scifact qrels...")
            #     _, _, qrels = download_dataset(dataset_name="scifact")
            #     dataset_name = "BEIR/scifact"
            except Exception as e:
                logging.error(f"Failed to load data: {e}")
                return
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return
        
    if args.command == "run":
        if not corpus or not queries or not qrels:
            logging.error("Data loading failed, cannot proceed with 'run'.")
            return
            
        model = None
        model_name_safe = ""
        try:
            model_wrapper = None
            if args.model == "hf":
                logging.info(f"Loading Hugging Face model: {args.hf_model_name}")
                model_name_safe = args.hf_model_name.replace("/", "_")
                model_wrapper = SentenceBERT(args.hf_model_name)
                logging.info("Model loaded successfully.")
            elif args.model == "openai":
                model_name_safe = args.openai_model_name
                model = OpenAIModel(model_path=args.openai_model_name, api_key=args.openai_api_key)
                model_wrapper = OpenAIModel(model_path=args.openai_model_name, api_key=args.openai_api_key)
            elif args.model == "google":
                model_name_safe = args.google_model_name.replace("/", "_")
                model = GoogleModel(model_path=args.google_model_name, api_key=args.google_api_key)
                model_wrapper = GoogleModel(model_path=args.google_model_name, api_key=args.google_api_key)

        
        except ValueError as e:
            logging.error(f"Failed to initialize model: {e}")
            return
            
        # Run the retrieval
        results = run_retrieval(model_wrapper, corpus, queries, args.top_k)
        
        if not results:
            logging.error("Retrieval failed, no results to save or evaluate.")
            return
            
        # Determine output file path
        output_file = args.output_file
        if not output_file:
            dataset_name_safe = "custom" if args.corpus_file else "scifact"
            output_file = f"results/{dataset_name_safe}_{model_name_safe}.json"
        
        # Save the results
        save_results(results, output_file)
        
        # Automatically evaluate the results just generated
        evaluate_results(output_file, qrels, k_values=[1, 3, 5, 10], dataset_name=dataset_name)

    elif args.command == "evaluate":
        print(f"qrel:{qrels}")
        if not qrels:
            logging.error("Qrels loading failed, cannot proceed with 'evaluate'.")
            return
        evaluate_results(args.results_file, qrels, args.k_values, dataset_name=dataset_name)


if __name__ == "__main__":
    main()

