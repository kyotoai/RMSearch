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

- [x] Fix a bug (tag_recs have tag inside each key_id so tag_id is not defined well.)
  -> just fixed make_leaf_tag_recs function by adding tag_meta and tag_recs in its argument
- [x] Improve prompt to get representative_tag
- [x] Fix a bug in search_key and search_tag (This was a fatal error, which can collapse all the result from tag_tree_recs with different branch sizes(not all branches are the same length). You shouldn't belive the output of search_tag and search_key before fixing this.)

- [x] First Evaluation for untrained reward model
  - [x] Assign keys into the tag_tree_recs and get tag2query
  - [x] Get evaluation of llama3b-generate
* -> Ended up with 0.05 nDCG. Probably it can be improved by making more accurate tag_graph. Also need to implement more dataset.


## Oct 5

- [x] Figure out why it ended up with low nDCG score.(which is the cause, graph or model itself?)
  - [x] Make code for evaluation without tag graph
  - [x] Debug it and get relevance_dict_without_graph
  - [x] Make code for evaluation of embedding model
  - [x] Debug it and get relevance_dict_with_embedding
    - [x] In vllm_embed.py, output embedding should be detach to cpu to avoid cuda oom
* -> So the bottom line is nDCG(embedding)=0.78, nDCG(rm with graph)=0.05, nDCG(rm without graph)=0.61. I need to enhance the graph and make the drop by graph less than 0.10. 

- [ ] Improve vllm_reward, embed, generate
  - [x] Make query batch in search function in vllm_reward2.py to avoid cpu oom
  - [x] If it encounters an error, stop generating output and clearing notebook output, and show the error message
  - [x] add an argument checkpoint_path and save output at the end of every batch
  - [ ] Make vllm_test.ipynb in examples to test it and debug them


## Oct 6, 7

- [ ] Rewrite train_en.ipynb functions into rmsearch directory
  - [ ] Rewrite and debug train_en.py
  - [x] Comment out more in rmsearch/
  - [x] Think how to scatter functions inside rmsearch directory and do it
    * vllm_reward, generate, embed -> rmsearch/utils/
    * Generate Tag Graph 2 -> rmsearch/tree
      * generate_tag -> rmsearch/tree/generate_tag.py
      * embed_tags -> rmsearch/tree/embed_tags.py
      * HierarchicalKMeans -> rmsearch/tree/hierarchical_kmeans.py
      * _as_int_list, _sorted_keys_numeric, convert_tree_dict_to_json -> method in HierarchicalKMeans
      * build_representative_tags -> rmsearch/tree/build_representative_tags.py

    * make_dataset -> rmsearch/train
      * make queries -> rmsearch/train/make_queries.py
      * search_tag -> rename it to def assign_key_to_tag_tree and save it in rmsearch/tree/assign_key.py
      * judge sentence -> rmsearch/train/judge_dataset.py

    * process_data -> rmsearch/train/process_data
    * train -> rmsearch/train/lora_example.py
    * model conversion -> rmsearch/train/utils.py
    * evaluation -> rmsearch/evaluation
  - [x] Modify rmsearch more
    - [x] In build_representative_tag, it should use vllm_generate.py instead of All_Requests. You should refer to Generate Tag Graph2 section for its usage.
    - [x] In lora_example, just imitate train_en.py and load model and train it in __main__
    - [x] So as retrieval.py. show how to use the search function in __main__. Just write the same code in train_en.ipynb
    - [x] Add arg.parser to all the files in rmsearch/
  - [x] Make README.md about tag_graph and train


## Oct 6

- [ ] Improve tag_graph step by step
  - [x] Design the architexture
  * Saved rough overview about graph update methods in graph_update_methods.md
  - [x] Implement code in train_en.ipynb Update Graph section refering to graph_update_methods.md. Input: tag_tree_recs -> Output: tag_graph.
  - [x] Debug the code and get tag_graph.json
  - [ ] Probably it's not working good yet. Debug and test it.
  - [ ] Recode search_key and assign_tag with tag_graph
  - [ ] Get evaluation results from the tag_graph

## Oct 7

- [x] Think how to devide this project to collaborators
* The most important thing is to grow AI training skills of collaborators -> devide rmsearch, seimei related thing to collaborators. Not me.
* Decide bonus based on creativity, novelty, impact and number of code lines. $100 ~ $800
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


## Oct 8, 9

- [ ] Prepare for collaborators to join
  - [x] Make their directory
  - [x] Make kickoff meeting md

`git clone --branch develop https://github.com/kyotoai/RMSearch.git`

- [ ] Debug rmsearch/ directory
  - [x] Make rmsearch_test.ipynb
  - [x] Process data is not what it's expected
  - [ ] Run all the code and check if it's working
    - [x] when running vllm_generate.py and others in python kernel, log keeps flowing which should be modified.
    - [x] add __main__ to utils/vllm_*.py
    - [x] add stream in process_data
    * if stream=True, it loads the necessary file only
    * I've not debug it yet.
    - [x] Make rmsearch/tree/search_key.py
    - [x] add get_top_relevant_keys_rm.py, get_top_relevant_keys_embed.py
```
In examples/train_en.ipynb "Make Dataset ..." section there is "Reward Model Gets TopN-Relevant ..." and  "Embedding Model Gets TopN-Relevant ...". I want you to implement these two sections to /rmsearch/train/get_top_relevant_keys_rm.py, /rmsearch/train/get_top_relevant_keys_embed.py respectively. Follow the points below

1. In get_top_relevant_keys_rm.py, use rmsearch/tree/assign_key.py and rmsearch/tree/search_key.py.
2. In get_top_relevant_keys_rm.py, first assign key and make tag2key from tag_tree. and then, search key and make relevant_records. save it in relevance_records_rm.json in default.
3. In rmsearch/train/get_top_relevant_keys_embed.py, you should use vllm_embed.py instead of sentence_transformers. other than that, follow the code in train_en.ipynb. save the output in relevance_records_embed.json in default.
4. add all the valuables used in the code to parse argument
5. Add ## `get_top_relevant_keys_rm.py` and ## `get_top_relevant_keys_embed.py` section below ## `make_queries.py` in readme. 
```
    - [x] Debug stream in process_data.py
    - [x] Debug rmsearch/train/make_queries.py
    - [x] Make rmsearch/train/make_query_recs.py - [{"query":, "query_id":, "df_id":, "query-type":(like "title", "questions")}, ...]
    - [x] Add rmsearch/train/filter_query_recs.py -> [{"query":, "query_id":, "df_id":, "query-type":(like only "questions")}, ...]
    - [x] Modify get_top_relevant_keys_rm.py, get_top_relevant_keys_embed.py for taking filter_query_recs as inputs and debug them
    - [x] Add sample_dpo_batch.py - sample query & 2 keys from queries (& relevance_records)
```
make rmsearch/train/sample_dpo_batch.py following
1. If relevance_records are given, sample 1 key from top relevant keys. And 1 from df_id. Combining the 2 keys, make sampled_query_key_set = [{"query":, "query_id":, "keys":[], "key_ids":[], "query-type":(like only "questions")}, ...]
2. Add all the valuables used in the code to parse argument
3. save it ./data/smollm/sampled_query_key_set.json in default
```
    - [x] Modify judge_dataset.py for sample_dpo_batch.py
    - [x] Debug judge_dataset.py
    - [ ] I want to save all the chat history to codex to some directory. Let's make one in notes.
    - [ ] Debug tree/


- [ ] Make tag graph

- [ ] Make checkpoint branch

- [ ] Implement gpt-oss instead of qwen7b (qwen7b has sometimes lower performance in tag generation)

- [ ] Train reward model
  - [ ] (Mingk) Make system to set aruguana as test dataset
  - [ ] Code to train rm for tag search
  - [ ] Add more dataset
  - [ ] Advanced DPO batching
  - [ ] GPU parallel for training with deepspeed






