title: RMSearchで報酬モデルを学習する実践パイプライン（LoRA + 参照データ生成）
emoji: 📚
type: tech
topics: ["LLM","RAG","LoRA","DPO","vLLM"]
published: false

はじめに
RMSearch では、クエリに対する「関連度」をスコアリングする報酬モデル（Reward Model, RM）を内製しています。本記事では、rmsearch/train のユーティリティを使い、ゼロからデータを用意して LoRA で RM を微調整するまでを俯瞰します。

ポイント
- コーパスからクエリと候補文を作り、ペアワイズの好みデータに変換
- 生成系モデルで「どちらがより関連するか」を判定（ジャッジ）
- TRL の RewardTrainer + LoRA で軽量に RM を学習

セットアップ
```
git clone --branch develop https://github.com/kyotoai/RMSearch.git
pip install -e RMSearch/.
```

GPU 環境を前提とします。Weights & Biases を使う場合は次の通り：
```
export WANDB_API_KEY=YOUR_KEY
wandb login
```

Step 1：データの整備（CSV 出力）
スクリプト：rmsearch/train/process_data.py

HuggingFace からデータセットを取得（オフライン時はスタブ CSV を作成）し、df.csv / df_small.csv を出力します。
```
python -m rmsearch.train.process_data \
  --dataset-name HuggingFaceTB/smollm-corpus \
  --output-dir ./data/smollm-corpus \
  --dataset-config cosmopedia-v2 \
  --n-sample 1000 \
  --stream
```
Tips：`HF_HUB_OFFLINE=1` でネットワーク無しでも試せます。`--n-sample` でサイズ調整。

Step 2：クエリ作成とフィルタリング
スクリプト：rmsearch/train/make_query_recs.py, rmsearch/train/filter_query_recs.py

- make_query_recs.py で各行からクエリを生成
- filter_query_recs.py で query-type に基づきサブセット化（例：factoid のみ）

この段階で `query_recs.json` および `filtered_query_recs.json` が得られます。

Step 3：Embedding による高速候補検索
スクリプト：rmsearch/train/get_top_relevant_keys_embed.py

埋め込みモデルでクエリとキー（df.csv のテキスト）をベクトル化し、類似度で上位 K 件を取得します。
```
python -m rmsearch.train.get_top_relevant_keys_embed \
  --queries-json ./data/smollm-corpus/filtered_query_recs.json \
  --keys-csv ./data/smollm-corpus/df.csv \
  --key-column text \
  --model-name intfloat/e5-mistral-7b-instruct \
  --k-key 50 \
  --output ./data/smollm-corpus/relevance_records_embed.json
```
出力：`relevance_records_embed.json`

Step 4：DPO 風のサンプリング（クエリ＋2候補）
スクリプト：rmsearch/train/sample_dpo_batch.py

ジャッジ用に、各クエリにつき 2 文を束ねたペアを作ります（元の df.csv に基づきテキストを復元）。
```
python -m rmsearch.train.sample_dpo_batch \
  --relevance-json ./data/smollm-corpus/relevance_records_embed.json \
  --filtered-queries-json ./data/smollm-corpus/filtered_query_recs.json \
  --source-csv ./data/smollm-corpus/df.csv \
  --output ./data/smollm-corpus/sampled_query_key_set.json
```

Step 5：判定（どちらがより関連？）
スクリプト：rmsearch/train/judge_dataset.py

vLLM 経由で小さめの指示特化モデルをジャッジとして回し、好みデータ（chosen/rejected）を生成します。途中経過を `--progress-dir` に保存し、再開も可能です。
```
python -m rmsearch.train.judge_dataset \
  --query-key-set ./data/smollm-corpus/sampled_query_key_set.json \
  --model-name /workspace/qwen4b \
  --tokenizer-name /workspace/qwen4b \
  --progress-dir ./data/smollm-corpus/progress \
  --output ./data/smollm-corpus/dataset_list.json
```

Step 6：LoRA で報酬モデルを微調整
スクリプト：rmsearch/train/lora_example.py

TRL の RewardTrainer を用い、LoRA アダプタで軽量に RM を学習します。
```
python -m rmsearch.train.lora_example \
  --dataset-list-train ./data/smollm-corpus/dataset_list.json \
  --model-name /workspace/llama3b-rm \
  --output-dir ./exp1/model1 \
  --wandb-project rmsearch \
  --wandb-run-name example-lora
```

成果物
- `./exp1/model1` 以下に LoRA チェックポイント
- `trainer_state.json` / `trainer_config.json`
- W&B に損失・評価メトリクス（任意）

運用のコツ
- ベースの RM ウェイトはローカル配置（例：Ray2333/GRM-Llama3.2-3B-rewardmodel-ft）
- ジャッジ用モデルは小型の Instruct 系でも十分に機能
- Embedding 段階はバッチを大きくしてスループット向上

なぜこの流れが効くのか
- 教師データの自給：強いジャッジで一貫したペアワイズ好みを作れる
- 目的特化の学習：RM が「関連度」をピンポイントに学ぶ
- LoRA による高速反復：大規模なフルチューニング不要

次の一手
- 自社コーパスに差し替え、Top‑K・サンプリング戦略を最適化
- ADPO 版（adpo_lora_example.py）で難易度の高い負例に挑戦
- 学習済み RM をリランキングに組み込み、検索品質を定量化

おわりに
「関連度の良し悪し」を教える報酬モデルは、検索・RAG の土台です。RMSearch の手順をなぞれば、チーム内で再現可能な形で RM を育て、改善サイクルを回せます。

