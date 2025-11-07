"""
Allows for evaluating dense retrievers, i.e., in the tevatron format on BEIR datasets.
models.HuggingFace allows for multi-gpu inference with DDP.

Example usage: CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_huggingface.py (for multi-gpu inference)
score of 0-1
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

dataset = "trec-covid"
data_path = f"/workspace/Mingkwan/beir_out/{dataset}"
qrels_tsv_path = f"{data_path}/csv_files/pair.csv"

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

## TREC-COVID
# emb_results_filepath = "/workspace/Mingkwan/beir_out/trec-covid/output/relevant_emb_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/trec-covid/output/relevant_rerank_q4_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/trec-covid/output/relevant_rerank_q4_640_eval.json"

##SCIFACT
# emb_results_filepath = "/workspace/Mingkwan/beir_out/scifact/output/relevant_emb_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/scifact/output/relevant_rerank_q4_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/scifact/output/relevant_rerank_q4_640_eval.json"

##FIQA
# emb_results_filepath = "/workspace/Mingkwan/beir_out/fiqa/output/relevant_emb_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/fiqa/output/relevant_rerank_q4_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/fiqa/output/relevant_rerank_q4_640_eval.json"

##SCIDOCS
# emb_results_filepath = "/workspace/Mingkwan/beir_out/scidocs/output/relevant_emb_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/scidocs/output/relevant_rerank_q4_eval.json"
# emb_results_filepath = "/workspace/Mingkwan/beir_out/scidocs/output/relevant_rerank_q4_640_eval.json"

QUERY_COL = 'query_id'  
DOC_COL = 'key_id' 
SCORE_COL = 'score'

# --- Step 1: Load and Process the TSV using Pandas ---
try:
    # Read the TSV file, specifying the separator is a tab ('\t')
    # df = pd.read_csv(qrels_tsv_path, sep='\t') 
    df = pd.read_csv(qrels_tsv_path, sep=',') 

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
    retriever = EvaluateRetrieval(None, score_function="cos_sim")

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info(f"Retriever evaluation for k in: 10")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,5,10,100])
mrr = retriever.evaluate_custom(qrels, results, [1,3,5,10,100], metric="mrr")

results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{data_path}/rerank_qwen_640_score.trec"), results)
util.save_results(os.path.join(results_dir, f"{data_path}/rerank_qwen_640_score.json"), ndcg, _map, recall, precision, mrr)