# kentarrito note

## Sep 9 2025

- [x] Finish baseline of evaluation
    - [x] Debug vllm_reward.py: enable data_parallel for reward model
    - [x] Implement search_key function in Evaluation in train_en.ipynb
    - [x] Rename functions and comment out
    - [x] Debug

## Sep 12

- [x] Add progress_bar in vllm_reward.py
- [x] Implement data parallel in assigning tags
- [ ] Make relevance_dict.json with evaluation code.   Canceled because vllm_reward takes much time without more tags 

## Sep 13

- [x] Debug graph search code in train_en.ipynb
- [ ] Comment out more

## Sep 14

- [x] Debug

## Sep 15

- [x] Debug code to generate relevance_dict.json
- [x] Comment out common error and how to fix them

- Realized it takes so much time to calculate last step of search_keys with 2 tag layers. \
  1000 keys * 5000 queries -> 8GPUS * 10h \
  we should definitely generate deeper tag graph \

## Sep 16

- [x] Design tag graph generation
    Get tag from each key using LLM -> Embed tags -> Use k-means to get 1000, 100, 10 weight points -> Get representative tag using LLM
- [x] Add vllm_generation.py and its test
- [x] Make better log system (vllm_reward2.py)

## Sep 17

- [x] Add generate_tag_graph.py

## Sep 19

- [x] Debug generate_tag_graph.py

## Sep 23

- [ ] Make dataset with debugged generate_tag_graph.py
- [x] Implement vllm_generate.py functions in generate_tag_graph.py(generate_tag_graph2.py)
- [x] Make vllm_embed.py and optimize embedding in train_en.py

## Sep 24

- [x] Debug vllm_embed.py
- [x] Debug generate_tag_graph2.py
- [x] Make dataset with generate_tag_graph2.py
  - [x] Run until the end
  - [x] Debug that something wrong with final group_recs
- [ ] Setup runpod vscode
- [ ] Make .py files of train_en.ipynb for LLM debugging
- [ ] In vllm_embed and vllm_generate5, it should show an error if it encountes

- [x] Make tag_dict from tag_recs, group_recs
  - [x] Add generate_tag_tree function to generate_tag_graph2.py
  - [x] Debug generate_tag_tree function
  - [x] Somehow, it only generates empty query2tag_ids

## Sep 27

- [x] Debug search_tag
- [x] rmsearch.py backup
- [ ] vllm_reward.py often get an error when search function is used twice in a row
- [x] Get evaluation result of llama3b-generate -> with this mechanism it takes so much time in the last search
- [ ] Debug search_key
  - [ ] In the final search, the cpu memory gets full even with 200GB

* there are still more than 1000 keys in a final node. it's because
1. one key have multiple tags
2. each node doesn't have expected numbers of children
-> Fix this by adding more nodes, and use advanced k-means

* Issue to be fixed
1. tags are overlapped. this should be fixed because same tag can have different keys

## Sep 28

- [x] Pick what k-means method to use
- [ ] Make generate_tag_graph3.py with that k-means method
- [ ] Degenerate overlapped tags

## Oct 3

- [x] Debug hierarchical_kmeans.py and its usage in Generate Tag Graph2 in train_en.ipynb
- [x] Get tag_tree_recs.json from the code above
- [ ] Assign keys into the tag_tree_recs and get tag2query
- [ ] Get evaluation of llama3b-generate

- [x] Rewrite train_en.ipynb functions into train_en.py
- [ ] Debug train_en.py

- [ ] Rewrite train_en.ipynb functions into rmsearch directory
- [ ] Comment out more
- [ ] Make README.md about train_en.ipynb

- [x] Write paper baseline

- [ ] Code to train rm for tag search

## Oct 4

- [ ] First Solid Evaluation
  - [ ] Assign keys into the tag_tree_recs and get tag2query
  - [ ] Get evaluation of llama3b-generate

- [ ] Rewrite train_en.ipynb functions into rmsearch directory
  - [ ] Debug train_en.py
  - [ ] Comment out more
  - [ ] Make README.md about tag_graph and train

- [ ] Train reward model
  - [ ] (Mingk) Make system to set aruguana as test dataset
  - [ ] Code to train rm for tag search
  - [ ] Add more dataset
  - [ ] Advanced DPO batching
  - [ ] GPU parallel for training with deepspeed














