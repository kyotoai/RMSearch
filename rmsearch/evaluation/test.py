# import json
# results = [{"doc1": [0.9, 0.8, 0.7], "doc2": [0.6, 0.5, 0.4]}]
# emb_file = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/embeddings_cache/embeddings.json"
# if emb_file:
#     print("Loading cached document embeddings...")
#     with open(emb_file, 'r') as f:
#         results = json.load(f)
#     print(f"Loaded {results} document embeddings from cache")
# else:
#     print("Computing document embeddings...")
#     np.save(emb_file, results)