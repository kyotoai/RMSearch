# RMSearch Evaluation Utilities

Utilities in `rmsearch.evaluation` reproduce the notebook evaluation
pipeline from the CLI. They materialise benchmark splits, build embedding
candidate sets, and optionally rerank those candidates with a reward model.

## `process_data.py`

Download a dataset split from HuggingFace (default: `BeIR/fiqa`) and export
three artefacts consumable by downstream steps:

- `query.json`: ordered list of query strings.
- `key.json`: ordered list of candidate sentences.
- `pair.csv`: two-column CSV mapping `query_id` → `key_id`.

```bash
python -m rmsearch.evaluation.process_data \
  --dataset-name BeIR/fiqa \
  --output-dir ./exp_eval/data \
  --query-split queries \
  --key-split corpus \
  --pair-split qrels \
  --max-queries 1000 \
  --max-keys 5000
```

**Highlights**
- Works with streaming and offline environments; falls back to deterministic
  stubs when HuggingFace is unavailable.
- Column, split, and size limits can be tuned per dataset.
- Outputs are pure text/CSV so they can be inspected or versioned easily.

## `embed.py`

Embed the `query.json` and `key.json` strings with a vLLM embedding model
and compute cosine-like similarity to retrieve the top-N keys per query.
Results are saved to `relevance_dict_embed.json` in the form
`{"query_id": int, "key_ids": [int, ...]}`.

```bash
python -m rmsearch.evaluation.embed \
  --query-json ./exp_eval/data/query.json \
  --key-json ./exp_eval/data/key.json \
  --model-name /workspace/e5-large \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --top-k 100 \
  --similarity-device auto
```

**Highlights**
- Shares batching and checkpointing logic with `rmsearch.utils.vllm_embed`.
- Optional L2 normalisation before similarity to mimic cosine scoring.
- Supports CPU/GPU similarity computation and automatic device selection.

## `rerank.py`

Consume `relevance_dict_embed.json` and re-score each candidate set with a
reward model (LLM) to produce `relevance_dict_rerank.json`. The file mirrors
the embedding output but includes a `relevance` score list for debugging.

```bash
python -m rmsearch.evaluation.rerank \
  --query-json ./exp_eval/data/query.json \
  --key-json ./exp_eval/data/key.json \
  --embed-json ./exp_eval/data/relevance_dict_embed.json \
  --model-name /workspace/llama3b-rm-converted-model \
  --tensor-parallel-size 1 \
  --num-instances 4 \
  --request-batch-size 128 \
  --timeout 10000
```

**Highlights**
- Reuses the notebook chat template (“Without Graph” section) for consistent
  scoring.
- Device groups can be pinned explicitly for multi-GPU layouts.
- Preserves the embedding shortlist order while attaching reward model
  scores.

## `retrieval.py`

Expose `retrieval_evaluation`, a helper that walks the tag tree to locate
candidate sentences before scoring them with a provided search function.
The module is imported when you run the legacy notebook evaluation path.

```python
from rmsearch.evaluation import retrieval_evaluation
```

**Highlights**
- Accepts synchronous or asynchronous search functions.
- Annotates each key with both the relevance score and the original key id.

## Package Init

`rmsearch/evaluation/__init__.py` re-exports `retrieval_evaluation` for
convenient imports (`from rmsearch.evaluation import retrieval_evaluation`).

---

All scripts assume local access to the required models (embedding or reward)
and benefit from GPU acceleration. When running the full pipeline, execute
the scripts in the order shown above.
