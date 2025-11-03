"""
Allows for evaluating dense retrievers, i.e., in the tevatron format on BEIR datasets.
models.HuggingFace allows for multi-gpu inference with DDP.

Example usage: CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_huggingface.py (for multi-gpu inference)
"""

import logging
import os
import pathlib
import random
import numpy as np
import pandas as pd
import json
from time import time

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dataset = "scifact"
data_path = f"/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/{dataset}"

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
qrels_tsv_path = "/workspace/kentarrito/beir_out/scifact/qrels.tsv"
emb_results_filepath = "/workspace/Prakhar/beir_out/scifact/relevance_dict_rerank_exp2-qwen-reward_adj.json"
emb_results_filepath = "/workspace/Prakhar/beir_out/scifact/relevance_dict_rerank_exp5_llama_adj.json"

QUERY_COL = 'query-id'  # e.g., 'qid'
DOC_COL = 'corpus-id'  # e.g., 'docid'
SCORE_COL = 'score' # e.g., 'label' or 'score'

# --- Step 1: Load and Process the TSV using Pandas ---
try:
    # Read the TSV file, specifying the separator is a tab ('\t')
    df = pd.read_csv(qrels_tsv_path, sep='\t') 

    # --- Data Cleaning and Validation ---
    
    # 1. Rename columns to the standard format for clarity/compatibility
    df.rename(columns={
        QUERY_COL: 'query-id', 
        DOC_COL: 'corpus-id', 
        SCORE_COL: 'score'
    }, inplace=True)

    # 2. Select the three required columns and drop any rows with missing values
    df = df[['query-id', 'corpus-id', 'score']].dropna()
    
    # 3. Ensure IDs are treated as strings (to match typical BEIR data format)
    df['query-id'] = df['query-id'].astype(str)
    df['corpus-id'] = df['corpus-id'].astype(str)
    
    # 4. Ensure score is an integer
    df['score'] = df['score'].astype(int) 

    # --- Step 2: Convert DataFrame to the Required QRELS Dictionary Structure ---
    # The required format is: {'query_id': {'doc_id': score, ...}}
    qrels = {}
    for qid, group in df.groupby('query-id'):
        # For each query ID, convert the document IDs and scores into a dictionary
        qrels[qid] = group.set_index('corpus-id')['score'].to_dict()

    print(f"✅ Successfully loaded and converted {df['query-id'].nunique()} queries into QRELS format.")

except FileNotFoundError:
    print(f"❌ Error: Qrels TSV file not found at {qrels_tsv_path}")
    qrels = {} # Initialize empty dictionary on failure

max_length = 512
pooling = "eos"
normalize = True
append_eos_token = True


if os.path.exists(emb_results_filepath):
    with open(emb_results_filepath, 'r') as f:
        results = json.load(f)
    print("found file emb!! no need for E5 model")
    retriever = EvaluateRetrieval(None, score_function="cos_sim")

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info(f"Retriever evaluation for k in: 10")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,5,10])
mrr = retriever.evaluate_custom(qrels, results, [1,3,5,10], metric="mrr")

results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{data_path}/rerank_qwen_score.trec"), results)
util.save_results(os.path.join(results_dir, f"{data_path}/rerank_qwen_score.json"), ndcg, _map, recall, precision, mrr)