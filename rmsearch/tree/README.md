# RMSearch Tag Tree Utilities

The modules in this directory orchestrate tag generation, embedding, tree
construction, and query assignment. They expose CLI entrypoints that mirror the
steps described in the "Generate Tag Graph2" section of `train_en.ipynb`.

All commands expect a GPU environment with access to the required models.

## `generate_tag.py`

Generate candidate tags for each key using a vLLM generation worker pool.

```bash
python -m rmsearch.tree.generate_tag \
  --keys-file ./data/smollm-corpus/keys.txt \
  --model-name /workspace/qwen7b \
  --output ./data/smollm-corpus/tag_records.json \
  --tensor-parallel-size 1 \
  --num-instances 2
```

**Arguments**
- `--keys-file`: Plain-text file with one key (sentence/title) per line.
- `--output`: Destination JSON file for tag records.
- `--model-name`: Generation model checkpoint.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Control worker topology; `--device-groups` accepts strings like `"0,1;2,3"`.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling and scheduling knobs.

**Outputs**
- `tag_records.json`: List of dictionaries with `key`, `key_id`, and generated `tags`.
- Example entry:
  ```json
  {
    "key": "Graph-based retrieval augmentation",
    "key_id": 0,
    "tags": ["graph retrieval", "enterprise search", "augmentation"]
  }
  ```

**Notices**
- The script creates and tears down the worker pool in-process. Ensure CUDA
  visibility matches the provided device groups.

## `embed_tags.py`

Embed tag strings using the vLLM embedding API.

```bash
python -m rmsearch.tree.embed_tags \
  --tag-records ./data/smollm-corpus/tag_records.json \
  --model-name intfloat/e5-mistral-7b-instruct \
  --embeddings-out ./data/smollm-corpus/key_embeddings.pt \
  --tag-meta-out ./data/smollm-corpus/tag_meta.json
```

**Arguments**
- `--tag-records`: JSON output from `generate_tag.py`.
- `--model-name`: Embedding model name/path.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Worker placement.
- `--worker-batch-size`, `--timeout`: Throughput controls.
- `--quantize`: Optional precision (e.g. `int8`) if sentence-transformers quantisation is installed.
- `--reduce-dim`: Optional dimensionality reduction target.

**Outputs**
- `key_embeddings.pt`: Torch tensor of embeddings.
- `tag_meta.json`: `(key_id, tag_idx)` metadata aligning rows with the original tags.
- Example `tag_meta.json` slice: `[[0, 0], [0, 1], [1, 0]]`

**Notices**
- Set `--device-groups` to pin embedding workers to specific GPUs if you launch
  more than one instance.

## `build_representative_tags.py`

Traverse the tag tree and populate representative tags for internal nodes using
vLLM generation.

```bash
python -m rmsearch.tree.build_representative_tags \
  --tag-tree ./data/smollm-corpus/tag_tree_recs.json \
  --model-name /workspace/qwen7b \
  --output ./data/smollm-corpus/tag_tree_recs.json
```

**Arguments**
- `--tag-tree`: Existing tag tree JSON (leaf nodes should already have `tags`).
- `--output`: File to overwrite or create with enriched tags (defaults to input path).
- `--model-name`: Generation checkpoint for tag summarisation.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Worker topology.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling controls.
- `--n-tag-sample`: Number of child tags sampled when summarising parent nodes.

**Outputs**
- Updated tag tree JSON with `tag` fields populated for internal nodes.
- Example node after enrichment:
  ```json
  {
    "tag": "Enterprise Retrieval",
    "children": [
      {"tag": "Graph Retrieval", "tags": ["graph retrieval", "knowledge graph"]},
      {"tag": "Hybrid Search", "tags": ["hybrid retrieval", "bm25"]}
    ]
  }
  ```

**Notices**
- Leaves without explicit `tag` fields are initialised with their first `tags`
  entry before generation.

## `assign_key.py`

Assign free-form queries to the tag tree using the reward-model scoring helper.

```bash
python -m rmsearch.tree.assign_key \
  --tag-tree ./data/smollm-corpus/tag_tree_recs.json \
  --queries-json ./data/smollm-corpus/query_prompts.json \
  --model-name /workspace/llama3b-rm-converted-model \
  --query2tag-out ./data/smollm-corpus/query2tag_ids.json \
  --tag2query-out ./data/smollm-corpus/tag2query.json
```

**Arguments**
- `--tag-tree`: Tag tree JSON produced earlier.
- `--queries-json` / `--queries-csv`: Input queries (list of strings or objects containing `"query"`); for CSV, use `--query-column` to select the column.
- `--query2tag-out`: JSON file storing the best path IDs per query.
- `--tag2query-out`: JSON file storing the augmented tree with `query_ids` annotations.
- `--model-name`: Reward model checkpoint used for scoring query/tag pairs.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Reward worker placement.
- `--batch-size`, `--timeout`: Search helper parameters.
- `--k-tag`: Number of top tags explored at each level.

**Outputs**
- `query2tag_ids.json`: List of `{ "tag_ids": [[...]] }` structures per query.
- `tag2query.json`: Tag tree annotated with `query_ids` for downstream retrieval.
- Example `query2tag_ids.json` item:
  ```json
  {
    "tag_ids": [[0, 1, 2], [3, 0]]
  }
  ```

**Notices**
- Queries should mirror the prompts produced by `make_queries.py` or whatever
  downstream text you intend to route through the tree.
- The reward model must be converted to the inference format expected by
  `rmsearch.utils.vllm_reward2`.

## `search_key.py`

Route queries through the tag graph and re-rank their candidate keys with the
reward model.

```bash
python -m rmsearch.tree.search_key \
  --queries ./data/smollm-corpus/queries.json \
  --keys ./data/smollm-corpus/keys.json \
  --tag2key ./data/smollm-corpus/tag2key.json \
  --model-name /workspace/llama3b-rm-converted-model \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --k-tag 2 \
  --k-key 5 \
  --max-model-len 2500 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90
```

**Arguments**
- `--queries`: JSON file containing query strings (plain list or objects with `"query"`).
- `--keys`: JSON file containing key strings (plain list or objects with `"text"`).
- `--tag2key`: Tag graph JSON where nodes include `"key_ids"` and optional `children`.
- `--model-name`: Reward model checkpoint or identifier.
- `--tensor-parallel-size`, `--num-instances`: Worker layout for the reward model.
- `--k-tag`, `--k-key`: Tag branching factor and final top-k keys per query.
- `--output`: Optional path to persist the ranked results as JSON.
- `--checkpoint`: Optional directory that caches intermediate `search_fn` responses for reuse.
- `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`: vLLM runtime limits.

**Outputs**
- Console prints the ranked list for each query when `--output` is omitted.
- With `--output`, writes a pretty-printed JSON file mirroring the console payload.

**Notices**
- When input paths are omitted, the script falls back to a small in-memory sample.
- Ensure the tag graph is aligned with the key indices you pass via `--keys`.
- Checkpoint caching skips repeated vLLM calls when the cached JSON files already exist.

## `hierarchical_kmeans.py`

This module exposes the `HierarchicalKMeans` class used inside notebooks to
construct the initial tree structure. It does not expose a CLI, but you can
instantiate it from Python to cluster embeddings prior to running the steps
above.

```python
from rmsearch.tree.hierarchical_kmeans import HierarchicalKMeans
```

**Notices**
- Requires scikit-learn, NumPy, and PyTorch. Use it offline to build the initial
  tree dictionary (`tag_tree_recs.json`).
