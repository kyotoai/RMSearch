
# Commands to process dataset

## Key 1 -> Query, Key2

* Biomedical
qiaojin/PubMedQA : question, context, answer -> generate similar answer2 with llm
BeIR/bioasq-generated-queries : title (sometimes not good), text -> extract only runnable data

```bash
python -m rmsearch.train.process_data \
  --dataset-name qiaojin/PubMedQA \
  --dataset-config pqa_labeled \
  --split train \
  --n-sample 1000 \
  --output-dir ./data/pubmedqa \
  --stream
```

* Finance
next-tat/TAT-QA -> didn't work
ibm-research/finqa -> didn't work

```bash
python -m rmsearch.train.process_data \
  --dataset-name ibm-research/finqa \
  --split train \
  --n-sample 1000 \
  --output-dir ./data/finqa \
  --stream
```

* Legal
coastalcph/lex_glue : context -> query, context 2

```bash
python -m rmsearch.train.process_data \
  --dataset-name coastalcph/lex_glue \
  --dataset-config case_hold \
  --split train \
  --n-sample 100 \
  --output-dir ./data/lex_glue \
  --stream
```

* Code



* Fact Checking 

fever/feverous



* General
smollm


