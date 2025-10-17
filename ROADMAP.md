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
  * Get relationship among k_key, k_tag ...

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
  - [ ] accumulate batches
  - [ ] 
- [ ] Collect more dataset
- [ ] Make more various queries (which is necessary for stable training)
- [ ] Advanced DPO batch (ask kentarrito about this)
- [ ] Train the model for tag graph search too

TODO for better paper
- [ ] Collect same proportion of mteb benchmark


## Oct 15

* ToDo (all -> do it with everyone collaboration, later -> decide the detail later)
  * Improve Tag graph -> kentarrito
  * Advanced DPO batching -> kentarrito
  * GPU parallel for training with deepspeed -> Prakhar
  * Implement gpt-oss -> Prakhar
  * Make system to set aruguana as test dataset -> Mingk
  * Add more dataset -> Mingk
  * Code to train rm for tag search -> later
  * Write papers and blogs -> all
  * Make web service -> kentarrito & roshia
  * design seimei -> all
  * make agents for analying directory, implementing knowledge graph and improving system real-time -> kentarrito, Cameron
  * make agents for nuclear fusion simulation automation -> kentarrito, Cameron
  * Make seimei library -> later
  * Write papers about seimei -> later
  * Expand business over a lot of companies based on these technology!


## Oct 16

Rough Roadmap From Now

- [ ] Implement docker in Runpod API call

- [ ] Update LLM call system (rmsearch/utils)
  - [ ] Implement gpt-oss instead of qwen (qwen7b has sometimes lower performance in tag generation)
  - [ ] Implement vllm serve
  - [ ] Implement llm api call
  - [ ] Implement rm api call
  - [ ] update rmsearch.py

- [ ] Improve Graph system
  - [ ] Graph topology optimization. (8h)

- [ ] Train reward model for rmsearch paper
  - [x] Make system to set aruguana as test dataset. (5h). (kentarrito)
  - [ ] Code to add tag search dataset. (7h)
  - [ ] Add more dataset (rmsearch/train/process_data.py: modify argument and get dataset from multiple source). (5h)
  - [ ] Advanced DPO batching. (3h)
  - [ ] GPU parallel for training with deepspeed. (5h)

- [ ] Agents system
  - [ ] Make baseline to create agent system (rmsearch/agents/). (4h)
  - [ ] Make concrete agents
    - [ ] Agents for realtime knowledge graph update. (6h)
    - [ ] Agents for excel analysis. (5h)
    - [ ] Agents for code analysis. (8h)

- [ ] Goals
  - [ ] RMSearch paper
    - [ ] Achieve better nDCG than e5-mistral by reward model without graph (reranker) -> minimal result to write a paper
    - [ ] Achieve better nDCG than e5-mistral by reward model with graph (rmsearch) -> this is something new. Favorable result for the paper.
  - [ ] SEIMEI paper
    - [ ] Get good evaluation for any task



