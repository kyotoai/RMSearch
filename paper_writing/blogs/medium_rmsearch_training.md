Title: Teaching a Reward Model to Rank: How We Built RMSearch’s Training Pipeline

Subtitle: From raw text to preference data to LoRA‑fine‑tuned reward models — a practical walkthrough.

If you’ve ever tried to improve retrieval or reranking quality in an LLM stack, you’ve likely run into a chicken‑and‑egg problem: you want a good reward model to learn what “relevance” looks like, but you need consistent preference data to teach it in the first place. In RMSearch, we turned this into a reproducible pipeline you can run on your own GPUs. This post walks through the end‑to‑end training flow we use to create a reward model that scores relevance for query–key pairs.

What you’ll learn
- How we bootstrap training data from a raw corpus
- How we generate queries, retrieve candidates, and build preference pairs
- How we train a lightweight LoRA reward model with TRL’s RewardTrainer
- Practical tips for offline runs and observability with Weights & Biases

Prereqs and setup
- Repo: kyotoai/RMSearch (develop branch)
- Environment: GPU box (Runpod or equivalent), local model weights
- Optional observability: Weights & Biases

Install RMSearch
```
git clone --branch develop https://github.com/kyotoai/RMSearch.git
pip install -e RMSearch/.
```

Step 1 — Materialize a working dataset
Script: rmsearch/train/process_data.py

This command downloads a HuggingFace dataset (or writes a stub if you’re offline), shuffles it, and exports convenient CSVs.

```
python -m rmsearch.train.process_data \
  --dataset-name HuggingFaceTB/smollm-corpus \
  --output-dir ./data/smollm-corpus \
  --dataset-config cosmopedia-v2 \
  --n-sample 1000 \
  --stream
```

Outputs
- ./data/smollm-corpus/df.csv
- ./data/smollm-corpus/df_small.csv

Tips
- Set HF_HUB_OFFLINE=1 to force stub CSVs when the network is unavailable.
- Use --n-sample to control disk usage when exploring.

Step 2 — Turn rows into queries and candidate keys
Scripts: rmsearch/train/make_query_recs.py, rmsearch/train/filter_query_recs.py

- make_query_recs.py: creates query records against the corpus
- filter_query_recs.py: keeps a subset by query-type (e.g., only “factoid”)

Expected artifact names in this phase include query_recs.json and filtered_query_recs.json, which capture query text, metadata, and pointers into df.csv.

Step 3 — Retrieve top candidates with embeddings (fast baseline)
Script: rmsearch/train/get_top_relevant_keys_embed.py

Given filtered queries and a key bank (e.g., the text column from df.csv), we embed both sides and compute similarity. The script writes a compact record per query with the top‑K key_ids and similarity scores.

Key flags
- --model-name: embedding model (e.g., intfloat/e5‑mistral‑7b‑instruct)
- --k-key: how many keys per query (default 50)
- --similarity-device cpu|cuda for the scoring stage

Output
- relevance_records_embed.json

Step 4 — Sample DPO‑style pairs from candidates
Script: rmsearch/train/sample_dpo_batch.py

We convert the retrieval results into small query–pair bundles suitable for judging. Each sample holds the query and two keys (with their original df_id mapping restored) so a judge model can decide which key is more relevant.

```
python -m rmsearch.train.sample_dpo_batch \
  --relevance-json ./data/smollm-corpus/relevance_records_embed.json \
  --filtered-queries-json ./data/smollm-corpus/filtered_query_recs.json \
  --source-csv ./data/smollm-corpus/df.csv \
  --output ./data/smollm-corpus/sampled_query_key_set.json
```

Step 5 — Ask a model to judge which key wins
Script: rmsearch/train/judge_dataset.py

We prompt a (separate) LLM to act as a “relevance judge” on each pair and persist the responses. The script can run with vLLM for throughput, checkpoint progress, and resume.

Typical invocation
```
python -m rmsearch.train.judge_dataset \
  --query-key-set ./data/smollm-corpus/sampled_query_key_set.json \
  --model-name /workspace/qwen4b \
  --tokenizer-name /workspace/qwen4b \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --progress-dir ./data/smollm-corpus/progress \
  --output ./data/smollm-corpus/dataset_list.json
```

The result is a compact list of preference pairs (chosen vs rejected) per query.

Step 6 — Fine‑tune a reward model with LoRA
Script: rmsearch/train/lora_example.py

With preference data in hand, we fine‑tune a reward model using TRL’s RewardTrainer and LoRA adapters. This keeps training light‑weight while preserving base weights.

```
python -m rmsearch.train.lora_example \
  --dataset-list-train ./data/smollm-corpus/dataset_list.json \
  --model-name /workspace/llama3b-rm \
  --output-dir ./exp1/model1 \
  --wandb-project rmsearch \
  --wandb-run-name example-lora
```

Outputs
- LoRA checkpoints under ./exp1/model1
- trainer_state.json / trainer_config.json
- Optional W&B run with losses and eval metrics

Observability with W&B
```
export WANDB_API_KEY=... && wandb login
```
Disable W&B by omitting --wandb-project.

Hardware and model notes
- Keep base reward model weights locally (e.g., Ray2333/GRM‑Llama3.2‑3B‑rewardmodel‑ft)
- For candidate generation/judging, smaller instruct models (e.g., Qwen3‑4B‑Instruct‑2507) are often sufficient
- Embedding stage benefits from larger context and batched inference; tune --query-batch-size and --key-batch-size

Why this pipeline works
- Data bootstrapping: we don’t need ground‑truth relevance; we manufacture consistent preferences from a strong judge
- Narrow supervision: the reward head learns exactly the scoring behavior we want for reranking
- Efficient adaptation: LoRA lets us iterate fast without touching full base weights

What’s next
- Swap in your corpus, adjust retrieval/top‑K, and iterate on judge quality
- Try the ADPO variant (adpo_lora_example.py) and advanced batch sampling for tougher negatives
- Integrate the trained reward model into your retrieval stack to rerank candidates

Closing thought
Training a reranking reward model doesn’t need to be an opaque research project. With a clear dataset flow, consistent judging, and small LoRA heads, you can ship measurable ranking gains quickly and repeatably.

