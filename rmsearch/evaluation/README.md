# RMSearch Evaluation Utilities

Utilities in `rmsearch.evaluation` reproduce the notebook evaluation
pipeline from the CLI. They materialise BEIR-style splits, build embedding
candidate sets, and optionally rerank those candidates with a reward model.

## Dataset Preparation

Two options exist for producing the `query.csv`, `key.csv`, and `pair.csv`
artifacts consumed by the downstream scripts:

```bash
# Convert BEIR datasets directly
python rmsearch/evaluation/dataset/beir_to_pairs.py \
  --outdir ./beir_out \
  --split test \
  scifact nq

# Or download from HuggingFace using datasets.load_dataset
python -m rmsearch.evaluation.process_data \
  --dataset-name BeIR/fiqa \
  --output-dir ./data/BeIR/fiqa \
  --query-split queries \
  --key-split corpus \
  --pair-split qrels \
  --max-queries 1000 \
  --max-keys 5000
```

- `query.csv`: ordered list of query records (`id`, `original_query_id`, `text`).
- `key.csv`: ordered list of candidate sentence records (`id`, `original_key_id`, `text`).
- `pair.csv`: positive relations between the two (`query_id`, `key_id`, plus originals).

Both scripts fall back to deterministic stub outputs when the dataset cannot
be downloaded.

## `embed.py`

Embed `query.csv` and `key.csv` with a vLLM embedding model and compute
dot-product similarity to retrieve the top-N keys per query. Results are
written to `relevance_dict_embed.json` as
`{"query_id": int, "key_ids": [int, ...], "positive_key_ids": [...]}`.

```bash
python -m rmsearch.evaluation.embed \
  --query-csv ./beir_out/scifact/query.csv \
  --key-csv ./beir_out/scifact/key.csv \
  --pair-csv ./beir_out/scifact/pair.csv \
  --model-name /workspace/e5-large \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --top-k 100 \
  --similarity-device auto
```

**Highlights**
- Shares batching and checkpointing logic with `rmsearch.utils.vllm_embed`.
- Optional L2 normalisation before similarity to mimic cosine scoring.
- Automatically maps embedding indices back to the dataset ids.

## `rerank.py`

Consume `relevance_dict_embed.json` and re-score each candidate set with a
reward model to produce `relevance_dict_rerank.json`. The output mirrors the
embed file while adding `relevance` scores.

```bash
python -m rmsearch.evaluation.rerank \
  --query-csv ./beir_out/scifact/query.csv \
  --key-csv ./beir_out/scifact/key.csv \
  --pair-csv ./beir_out/scifact/pair.csv \
  --embed-json ./beir_out/scifact/relevance_dict_embed.json \
  --model-name /workspace/llama3b-rm-converted-model \
  --tensor-parallel-size 1 \
  --num-instances 4 \
  --request-batch-size 128 \
  --timeout 10000
```

**Highlights**
- Reuses the notebook chat template (“Without Graph”) for consistent scoring.
- Device groups can be pinned explicitly for multi-GPU layouts.
- Preserves embedding order while attaching reward model scores and positive ids.

## `retrieval.py`

Run the reward model across every query–key pair (or the legacy tag-tree
evaluation) to generate `relevance_dict.json`.

```bash
python -m rmsearch.evaluation.retrieval \
  --query-csv ./beir_out/scifact/query.csv \
  --key-csv ./beir_out/scifact/key.csv \
  --pair-csv ./beir_out/scifact/pair.csv \
  --model-name /workspace/llama3b-rm-converted-model \
  --tensor-parallel-size 1 \
  --num-instances 4 \
  --batch-size 512 \
  --k-key 100 \
  --output ./beir_out/scifact/relevance_dict.json
```

If `--query-csv` / `--key-csv` are unavailable, the script falls back to the
original notebook inputs (`df_small.csv`, `query_dict.json`, and
`tag2query-tag_tree.json`) beneath `--working-dir`.

**Highlights**
- Direct “without graph” mode scores every key per query using BEIR pairs.
- Legacy mode still supports tag-tree traversal when those artifacts exist.
- Outputs include the reward scores plus optional `positive_key_ids`.

## Package Init

`rmsearch/evaluation/__init__.py` re-exports the primary helpers so you can
write `from rmsearch.evaluation import process_data, build_relevance_dict,
rerank_candidates, retrieval_evaluation`.

---

All scripts assume local access to the required embedding and reward models
and benefit from GPU acceleration. Run them in the order above to reproduce
the evaluation artifacts referenced throughout the project.
