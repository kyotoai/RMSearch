# RMSearch paper TODO

## Sep 9 2025

### Experiment. 2 weeks
- [x] 1. Training
- [ ] 2. Evaluation (rms vs embedding, rerank)
- [ ] 3. Add Dataset
- [ ] 4. Repeat 1~3 several times and obtain the model
- [ ] 5. Scalability comparison (graph rms vs normal rms)

### Coding for Experiment
kenta
- [x] Data Parallel
- [x] Evaluation Baseline 2h
- [ ] Tag Graph Dataset 10h
- [ ] Scalability 5h

mingkwan
- [x] Evaluation 2 days
  - [x] nDCG@10
- [ ] Implement Deepspeed

### Code RMSearch Github  1 week
- [ ] Data Parallel 1h
- [ ] Graph Search 1h
- [ ] Training methods variation (DPO, batch DPO, score …) 2h
- [ ] Graph 2h

### Paper writing. 1 week
- [x] Introduction (paper, similar work) 10h 
- [ ] Experimental setup 3h
- [ ] Result 3h 
- [ ] Evaluation 3h
- [ ] Others 3h


## Sep 30

TODO for Minimal Experiment to write the paper
- [ ] 1. Train the model with dataset smoll (exp3) \
    -> Confirm that the accuracy goes up with training dataset(smoll-corpus) \
    -> Confirm that the accuracy goes up with evaluation dataset(arguana) \
    *If it doesn't, probably improving the llm-generated queries would be a good start
- [x] 2. Generate Tag Graph with hierarchical k-means method
- [ ] 3. Do Evaluation based on tag graph
- [ ] 4. Compare the evaluation result with Embedding one
- [ ] 5. Repeat this until we get good result on benchmark

TODO for Better benchmark result
- [ ] GPU parallel for training with deepspeed
- [ ] Collect more dataset
- [ ] Make more various queries (which is necessary for stable training)
- [ ] Advanced DPO batch (ask kentarrito about this)
- [ ] Train the model for tag graph search too

TODO for better paper
- [ ] Collect same proportion of mteb benchmark

