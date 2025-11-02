# The Evaluation evaluate_retrieval.py have to argument "run" or "evaluate"

"run" allow you to do both used the model to evaluate a "query", "corpus" and output a json file, then evaluate that file on nDCG. How you run is like this. Chnage the --model hf to openai, or hf according to model. The --hf_model_name paramet is set to the correct hf_model. query, corpus, qrels file location is correct unless you wanna test it on a new dataset. The nDCG output is in the command prompt.

```
cd /workspace/Mingkwan/RMSearch
source .venv/bin/activate
#if any missing dependancies
pip install -r rmsearch/evaluation/
```

### If we are running a model from RMSearch

Run the rerank.py. Change the output and model-name to your desire. Don't need to change the embed json.

```
python -m rmsearch.evaluation.rerank \
  --query-csv ./beir_out/scifact/query.csv \
  --key-csv ./beir_out/scifact/key.csv \
  --pair-csv ./beir_out/scifact/pair.csv \
  --embed-json ./beir_out/scifact/relevance_dict_embed.json \
  --output ./beir_out/scifact/relevance_dict_rerankYOURMODELNAME.json \
  --model-name /workspace/Mingkwan/RMSearch/models/YOURMODELNAMEl \
  --tensor-parallel-size 1 \
  --num-instances 1 \
  --request-batch-size 128 \
  --timeout 10000
```


## Run in background with nohup and direct stdout and stderr to output.log
```
nohup python -m rmsearch.evaluation.rerank   --query-csv ./beir_out/scifact/query.csv   --key-csv ./beir_out/scifact/key.csv   --pair-csv ./beir_out/scifact/pair.csv   --embed-json ./beir_out/scifact/relevance_dict_embed.json   --output ./beir_out/scifact/relevance_dict_rerank1240.json   --model-name /workspace/Mingkwan/RMSearch/models/Pra1_1240-converted-model   --tensor-parallel-size 1   --num-instances 2   --request-batch-size 128   --timeout 10000 > output.log 2>&1 &
```

We need to adjust the dataset a bit. Firstly, if we are NOT changing the dataset or downloading new one, then run the /workspace/Mingkwan/RMSearch/beir_out/scifact/transform_json.py line, but change the INPUT and OUTPUT file location inside the transform_json.py.

```
python beir_out/scifact/transform_json.py
```

Then we run the evaluation on the new model. This command will create an output file to the output folder and also run nDCG file.

```
python rmsearch/evaluation/evaluate_retrieval.py run \
  --model hf \
  --hf_model_name BAAI/bge-large-en-v1.5 \
  --corpus_file /workspace/Mingkwan/RMSearch/beir_out/scifact/key.csv \
  --queries_file /workspace/Mingkwan/RMSearch/beir_out/scifact/query.csv \
  --qrels_file /workspace/Mingkwan/RMSearch/beir_out/scifact/qrels.csv \
  --output_file /workspace/Mingkwan/RMSearch/beir_out/scifact/hf/OUTPUTFIELNAME.json \
  --top_k 100 
```

### If the model is already load, nDCG we can run "evaluate"

Only the results_file (model path) need to be change.

```
python rmsearch/evaluation/evaluate_retrieval.py evaluate \
--results_file /workspace/Mingkwan/RMSearch/beir_out/scifact/OUR_MODEL.json \
--qrels_file /workspace/Mingkwan/RMSearch/beir_out/scifact/qrels.csv 
```


```
python rmsearch/evaluation/evaluate_retrieval.py evaluate \
--results_file /workspace/Mingkwan/RMSearch/beir_out/scifact/relevance_dict_adj_rerank1240.json \
--qrels_file /workspace/Mingkwan/RMSearch/beir_out/scifact/qrels.csv 
```