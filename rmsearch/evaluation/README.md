# Run evalaution

Some checkpoints for you to consider before running evaluation
- If your model hasn't been converted, it need to be convert with utils.py
- This only work for BEIR dataset with ones that have a (0,1) score
- If the dataset folder exist already, you might need to delete it first before running the scripts
- The --output is the output you use as the input of rerank.py script, the --output-eval is for input of ndcg.py script (scoring)

## Inital Setup

```
cd /workspace/Mingkwan/RMSearch
source .venv/bin/activate
#if any missing dependancies
pip install -r rmsearch/evaluation/
```

## `embed.py`

```bash
python -m rmsearch.evaluation.embed \
  --dataset-path /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact \
  --split test \
  --output /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/relevant_emb.json \
  --output-eval /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/relevant_emb_eval.json \
  --model-name /workspace/e5-mistral7b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --top-k 100 \
  --similarity-device auto
```

**Highlights**
- Autmatically detect existance of dataset and download them if needed.
- Output two separate json files for next step reranking and scoring with nDCGå
- **Arguments**
  - `--dataset-path` path to your dataset
  - `--top-k`: Maximum number of keys kept per query (default `100`).
  - `--similarity-device`: Device used for the similarity matrix (`cpu`, `cuda`, or `auto`).
- **Outputs**
  - `relevance_emb.json` containing ordered candidate ids per query.
  - Example entry:
    ```json
    {
      "query_id": 42,
      "key_ids": [12, 7, 105, 3],
      "positive_key_ids": [7],
      "embed_relevances": [0.6149212121963501]
    }
    ```

  - `relevance_emb_eval.json` for evaluating with ndcg.py
  - Example entry:
    ```json
    {
  "0": {
    "766": 0.6149212121963501,}
    }
    ```

## `rerank.py`

Consume `relevance_emb.json` and re-score each candidate set with a
reward model to produce `relevance_rerank_eval.json`. The output mirrors the
embed file while adding `relevance` scores.

```bash
python -m rmsearch.evaluation.rerank \
  --dataset-path /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact \
  --embed-output /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/relevant_emb.json \
  --output-eval /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/scifact/relevant_rerank_eval.json \
  --model-name /workspace/Mingkwan/RMSearch/models/Pra1_1240-converted-model \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --request-batch-size 128 \
  --timeout 100000
```

- **Arguments**
  - `--dataset-path` Input your dataset path here
  - `--embed-output`: Output from `embed.py` supplying `pre_key_ids`.

- **Outputs**
  - `relevance_rerank_eval.json` preserving `pre_key_ids` while trimming `key_ids`.
  - Example entry:
    ```json
    {
  "0": {
    "1383": 0.7981379628181458,
    "1377": 0.788284957408905,
    "1382": 0.778290867805481,
    "1": 0.7754657864570618,
    "1379": 0.7737131714820862,
    "1375": 0.771685004234314,
    "1373": 0.7712595462799072,
    "1104": 0.7644272446632385,
    "2776": 0.761962711811065}
    }
    ```

  ## ndcg.py
Go inside the script and change the 
dataset = "nfcorpus"
data_path = "/workspace/kentarrito/beir_out/_raw/scifact"
emb_results_filepath = "/workspace/Prakhar/beir_out/scifact/relevance_dict_rerank_exp2-qwen-reward_adj.json"

then run python/rmsearch/evaluation/ndcg.py
