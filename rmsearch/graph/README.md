# RMSearch Tag Graph Utilities

The modules in this directory orchestrate tag generation, embedding, graph
construction, and update graph topology. 

All commands expect a GPU environment with access to the required models.



## Overview

(add exlanations refering to rmsearch/train/README.md)

## `generate_tag.py`

Generate candidate tags for each key using a vLLM generation worker pool.

```bash
python -m rmsearch.tree.generate_tag \
  --keys-file ./data/smollm-corpus/df.csv \
  --key-column text \
  --model-name /workspace/qwen4b \
  --output ./data/smollm-corpus/tag_records.json \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --max-model-len 10_000
```

**Arguments**
- `--keys-file`: Plain-text file with one key (sentence/title) per line.
- `--key-column`: Column name of keys_file if it's a csv file.
- `--output`: Destination JSON file for tag records.
- `--model-name`: Generation model checkpoint.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Control worker topology; `--device-groups` accepts strings like `"0,1;2,3"`.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling and scheduling knobs.
- `--gpu-memory-utilization`, `--max-model-len`, `--dtype`, `--trust-remote-code`: Options forwarded to `vllm.LLM`.

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
  --model-name /workspace/e5-mistral7b \
  --embeddings-out ./data/smollm-corpus/tag_embeddings.pt \
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
- `tag_embeddings.pt`: Torch tensor of embeddings.
- `tag_meta.json`: `(key_id, tag_idx)` metadata aligning rows with the original tags.
- Example `tag_meta.json` slice: `[[0, 0], [0, 1], [1, 0]]`

**Notices**
- Set `--device-groups` to pin embedding workers to specific GPUs if you launch
  more than one instance.

## `make_tag_tree.py`

Cluster tag embeddings into a hierarchical k-means tree.

```bash
python -m rmsearch.tree.make_tag_tree \
  --working-dir . \
  --data-name smollm-corpus \
  --branching-factor 10 \
  --max-leaf-size 60 \
  --random-state 0
```

**Arguments**
- `--working-dir`: Root directory that contains the `data/` folder (defaults to current directory).
- `--data-name`: Dataset subdirectory inside `data/` that holds tag artifacts.
- `--embeddings`: Optional override path for `tag_embeddings.pt`.
- `--tag-meta`: Optional override path for `tag_meta.json`; used to validate alignment.
- `--output`: Optional destination for `tag_tree_recs.json`.
- `--branching-factor`: Maximum number of clusters per split (`n_clusters` passed to k-means).
- `--max-leaf-size`: Leaf size threshold before splitting further.
- `--random-state`: Seed controlling k-means initialisation.

**Outputs**
- `tag_tree_recs.json`: Hierarchical tree structure with leaf `tag_ids` lists.
- Example output:


**Notices**
- The script validates that the embedding rows match `tag_meta.json` length before clustering.
- Uses `HierarchicalKMeans` under the hood; adjust `--branching-factor` and `--max-leaf-size` to tune tree shape.



## `build_representative_tags_v2.py`

Traverse the tag tree and populate representative tags for internal nodes using
vLLM generation.

```bash
python -m rmsearch.graph.build_representative_tags_v2 \
  --tag-tree ./data/smollm-corpus/tag_tree_recs.json \
  --model-name /workspace/qwen4b \
  --max-sample-children 20 \
  --max-sample-other 20 \
  --max-model-len 10_000 \
  --output ./data/smollm-corpus/tag_tree_recs.json
```

**Arguments**
- `--tag-tree`: Existing tag tree JSON (leaf nodes should already have `tags`).
- `--output`: File to overwrite or create with enriched tags (defaults to input path).
- `--model-name`: Generation checkpoint for tag summarisation.
- `--max-sample-children`: Number of samples from representative tag's children.
- `--max-sample-other`: Number of counter samples from other tag's children.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Worker topology.
- `--worker-batch-size`, `--timeout`, `--temperature`, `--top-p`, `--max-tokens`: Sampling controls.
- `--n-tag-sample`: Number of child tags sampled when summarising parent nodes.
- `--gpu-memory-utilization`, `--max-model-len`, `--dtype`, `--trust-remote-code`: Options forwarded to `vllm.LLM`.

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
- Changes from build_representative_tags_v2.py
  * Get as many samples as possible from other tree branches and think what's common only among the children tags
  * Add argument for the number of samples from children and other branches' children
  * Change prompt so that the representative tag gets more concrete and long.





## `convert_tree_to_graph.py`

Convert tag_tree_recs.json created in make_tag_tree.py into tag_graph.parquet.

```bash
python -m rmsearch.tree.convert_tree_to_graph \
  --tag-tree ./data/smollm-corpus/tag_tree_recs.json \
  --output ./data/smollm-corpus/tag_graph.parquet
```

**Arguments**
- `--tag-tree`: Existing tag tree JSON (leaf nodes should already have `tags`).
- `--output`: File to overwrite or create with tag graph files (defaults to tag_graph.parquet).

**Outputs**
- Files inside output_dir:
  * `tag_graph.parquet` – stores node info. {"tag_id":id1, "tag":"machine learning", "children_tag_ids":[id2, id3 ...]}
- Examples:

**Notices**




## `assign_key_graph.py`

Assign free-form queries to the tag graph using the reward-model scoring helper.

```bash
python -m rmsearch.graph.assign_key_graph \
  --tag-graph ./data/smollm-corpus/tag_graph.parquet \
  --keys-json ./data/smollm-corpus/keys.json \
  --model-name /workspace/llama3b-rm-converted-model \
  --key2tag-out ./data/smollm-corpus/key2tag_ids.parquet \
  --tag2key-out ./data/smollm-corpus/tag2key_ids.parquet
```

**Arguments**
- `--tag-graph`: Tag graph parquet produced earlier.
- `--keys-json` / `--keys-csv`: Input queries (list of strings or objects containing `"key"`); for CSV, use `--key-column` to select the column.
- `--key2tag-out`: parquet file storing the best path IDs per query.
- `--tag2key-out`: parquet file storing the augmented tree with `key_ids` annotations.
- `--model-name`: Reward model checkpoint used for scoring query/tag pairs.
- `--tensor-parallel-size`, `--num-instances`, `--device-groups`: Reward worker placement.
- `--batch-size`, `--timeout`: Search helper parameters.
- `--k-tag`: Number of top tags explored at each level.

**Outputs**
- `key2tag_ids.parquet`: List of `{ "tag_ids": [[...]] }` structures per query.
- `tag2key_ids.parquet`: Tag graph annotated with `tag_ids` for downstream retrieval.
- Example `tag2key_ids.parquet` item:
  ```
  {
    "tag_id": 0,
    "tag":"machine learning",
    "children_tag_ids":[1, 2 ...],
    "key_ids": [0, 3]
  }
  ```
- Example `key2tag_ids.parquet` item:
  ```
  {
    "tag_ids": [[0, 1, 2], [3, 0]]
  }
  ```

**Notices**
- Queries should mirror the prompts produced by `make_queries.py` or whatever
  downstream text you intend to route through the tree.
- The reward model must be converted to the inference format expected by
  `rmsearch.utils.vllm_reward`.





## `search_key_graph.py`

Route queries through the tag graph and re-rank their candidate keys with the
reward model.

```bash
python -m rmsearch.graph.search_key_graph \
  --queries ./data/smollm-corpus/queries.json \
  --keys ./data/smollm-corpus/keys.json \
  --tag2key ./data/smollm-corpus/tag2key_ids.parquet \
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
- `--tag2key`: Tag graph parquet where nodes include `"tag_id"`, `"tag"`, `"children_tag_ids"` and optional `"key_ids"`.
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





## `make_query_recs.py`

Flatten the generated titles, keywords, questions, and irrelevant questions into
per-query recommendation records while reusing the same vLLM backend.

```bash
python -m rmsearch.train.make_query_recs \
  --input-csv ./data/smollm-corpus/df.csv \
  --text-column text \
  --model-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --batch-size 8 \
  --max-model-len 10000 \
  --output ./data/smollm-corpus/query_recs.json
```

**Arguments**
- Inherits the same CLI as `make_queries.py`; see above for detailed flag descriptions.

**Outputs**
- `{output}`: JSON list where each element contains `query`, `df_id`, and `query-type`, covering every generated title/keyword/question/irrelevant question.
- Example entry:
  ```json
  [
    {"query": "Graph Retrieval Overview", "df_id": 42, "query-type": "titles"},
    {"query": "How does graph retrieval work?", "df_id": 42, "query-type": "questions"}
  ]
  ```

**Notices**
- Shares batching, sampling, and fallback behaviour with `make_queries.py`; refer to that section for runtime considerations.





## `add_edges_by_query_key_set.py`

Bridge tag to tag so that reward model can search key from query. 

```bash
python -m rmsearch.graph.add_edges_by_query_key_set \
  --keys-file ./data/smollm-corpus/df.csv \
  --key-column text \
  --queries ./data/smollm-corpus/query_recs.json \
  --tag2key ./data/smollm-corpus/tag2key_ids.parquet \
  --output ./data/smollm-corpus/tag2key_ids2.parquet \
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
...
- `--tag2key`: Tag graph parquet where nodes include `"tag_id"`, `"tag"`, `"children_tag_ids"` and optional `"key_ids"`.
- `--model-name`: Reward model checkpoint or identifier.
- `--tensor-parallel-size`, `--num-instances`: Worker layout for the reward model.
- `--k-tag`, `--k-key`: Tag branching factor and final top-k keys per query.
- `--output`: Optional path to persist the ranked results as JSON.
- `--checkpoint`: Optional directory that caches intermediate `search_fn` responses for reuse.
- `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`: vLLM runtime limits.

**Outputs**
- Improved tag2key where new edges (more children_tag_ids) are added to connect query to key.

**Notices**









