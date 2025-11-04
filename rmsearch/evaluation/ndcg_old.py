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
# from beir.reranking.models import CrossEncoder
from sentence_transformers import CrossEncoder
from beir.reranking import Rerank

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dataset = "scifact"


# class DummyRetriever:
#     def __init__(self, k_values=None):
#         # BEIR reads this to know what k's to print
#         self.k_values = k_values or [1, 3, 5, 10]

#     # just to be safe in case someone calls .retrieve()
#     def retrieve(self, *args, **kwargs):
#         raise RuntimeError("Dummy retriever: use precomputed `results` instead.")



#### Download nfcorpus.zip dataset and unzip the dataset
# url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

data_path = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact"
# data_path = "/workspace/kentarrito/beir_out/_raw/scifact"

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#qrels_tsv_path = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/new_emb_results_adj.tsv"
qrels_tsv_path = "/workspace/kentarrito/beir_out/scifact/qrels.tsv"

# 2. Define the column names in YOUR TSV file
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

# print(f"qrels:{qrels}")
model_name_or_path = "intfloat/e5-mistral-7b-instruct"
# model_name_or_path = "workspace/Prakar/exp2/model1/checkpoint-120"  # local path to finetuned model
max_length = 512
pooling = "eos"
normalize = True
append_eos_token = True

#### Configuration for E5-Mistral
# query_prompt = "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: "
# passage_prompt = ""
# dense_model = models.HuggingFace(
#     model_path=model_name_or_path,
#     max_length=max_length,
#     append_eos_token=append_eos_token,  # add [EOS] token to the end of the input
#     pooling=pooling,
#     normalize=normalize,
#     prompts={"query": query_prompt, "passage": passage_prompt},
#     attn_implementation="flash_attention_2",
#     torch_dtype="bfloat16",
# )

# model = DRES(dense_model, batch_size=128)
# retriever = EvaluateRetrieval(model, score_function="cos_sim")

# emb_results_filepath = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/embeddings_cache/nfcorpus_results.json"
#emb_results_filepath = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/new_emb_results_adj.json"
#emb_results_filepath = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/new_rerank_results.json"
emb_results_filepath = "/workspace/Prakhar/beir_out/scifact/relevance_dict_rerank_exp2-qwen-reward_adj.json"
# emb_results_filepath = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/relevant_rerank_eval.json"
#emb_results_filepath = "/workspace/kentarrito/beir_out/scifact/relevance_dict_embed_adj.json"
if os.path.exists(emb_results_filepath):
    # 1. LOAD FROM FILE 
    # logging.info(f"--- Loading cached results from {emb_results_filepath} ---")
    # print(f"--- Loading cached results from {emb_results_filepath} ---")
    with open(emb_results_filepath, 'r') as f:
        results = json.load(f)
    # query_prompt = "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: "
    # passage_prompt = ""
    # dense_model = models.HuggingFace(
    #     model_path=model_name_or_path,
    #     max_length=max_length,
    #     append_eos_token=append_eos_token,  # add [EOS] token to the end of the input
    #     pooling=pooling,
    #     normalize=normalize,
    #     prompts={"query": query_prompt, "passage": passage_prompt},
    #     attn_implementation="flash_attention_2",
    #     torch_dtype="bfloat16",
    # )

    # model = DRES(dense_model, batch_size=128)
    print("found file emb!! no need for E5 model")
    # retriever = EvaluateRetrieval(DummyRetriever(), score_function="cos_sim")
    retriever = EvaluateRetrieval(None, score_function="cos_sim")
    # print(f"results: {results}")
else:
    print("using else block")
    # 2. RUN RETRIEVAL 
    query_prompt = "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: "
    passage_prompt = ""
    dense_model = models.HuggingFace(
        model_path=model_name_or_path,
        max_length=max_length,
        append_eos_token=append_eos_token,  # add [EOS] token to the end of the input
        pooling=pooling,
        normalize=normalize,
        prompts={"query": query_prompt, "passage": passage_prompt},
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
    )

    model = DRES(dense_model, batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    print(f"--- No cached results found. Running retrieval... ---")
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

    # 3. SAVE TO FILE
    print(f"--- Saving results to {emb_results_filepath} ---")
    with open(emb_results_filepath, 'w') as f:
        json.dump(results, f)

# ### Load REranker model
# reranker_model_name = "Qwen/Qwen3-Reranker-4B"
# cross_encoder_model = CrossEncoder(
#     reranker_model_name,
#     max_length=512,
#     device='cuda',
#     trust_remote_code=True
# )
# if cross_encoder_model.tokenizer.pad_token is None:
#     cross_encoder_model.tokenizer.pad_token = cross_encoder_model.tokenizer.eos_token
#     print("Warning: pad_token is None. Setting pad_token to eos_token.")

# # Ensure the model's config also knows about this
# if cross_encoder_model.model.config.pad_token_id is None:
#     cross_encoder_model.model.config.pad_token_id = cross_encoder_model.tokenizer.eos_token_id

# rerank_results_filepath = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/reranker_results.json"
# if os.path.exists(rerank_results_filepath):
#     # 1. LOAD FROM FILE 
#     logging.info(f"--- Loading cached results from {rerank_results_filepath} ---")
#     print(f"--- Loading cached results from {rerank_results_filepath} ---")
#     with open(rerank_results_filepath, 'r') as f:
#         rerank_results = json.load(f)
#     reranker = Rerank(cross_encoder_model, batch_size=16)
# else:
#     # 2. RUN RERANKER
#     reranker = Rerank(cross_encoder_model, batch_size=16)

#     #### Retrieve dense results (format of results is identical to qrels)
#     start_time = time()
#     rerank_results = reranker.rerank(corpus, queries, results, top_k=100)
#     end_time = time()
#     print(f"Time taken to rerank: {end_time - start_time:.2f} seconds")
    
#     print(f"--- Saving results to {rerank_results_filepath} ---")
#     with open(rerank_results_filepath, 'w') as f:
#         json.dump(rerank_results, f)
        
# reranker = Rerank(cross_encoder_model, batch_size=16)

#### Retrieve dense results (format of results is identical to qrels)
# start_time = time()
# rerank_results = reranker.rerank(corpus, queries, results, top_k=100)
# end_time = time()
# print(f"Time taken to rerank: {end_time - start_time:.2f} seconds")

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info(f"Retriever evaluation for k in: 10")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,5,10])
mrr = retriever.evaluate_custom(qrels, results, [1,3,5,10], metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{data_path}/rerank_qwen.trec"), results)
util.save_results(os.path.join(results_dir, f"{data_path}/rerank_qwen.json"), ndcg, _map, recall, precision, mrr)

#### Print top-k documents retrieved ####
# top_k = 10

# query_id, ranking_scores = random.choice(list(results.items()))
# scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
# logging.info(f"Query : {queries[query_id]}\n")

# for rank in range(top_k):
#     doc_id = scores_sorted[rank][0]
#     # Format: Rank x: ID [Title] Body
#     logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")

### NDCG@K results should look like this:
# NDCG@1: 0.4830
# NDCG@3: 0.4287
# NDCG@5: 0.4102
# NDCG@10: 0.3845
# NDCG@100: 0.3520
# NDCG@1000: 0.4360