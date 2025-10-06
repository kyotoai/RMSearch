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


  ## Oct 6

- [ ] Rewrite train_en.ipynb functions into rmsearch directory
  - [ ] Rewrite and debug train_en.py
  - [ ] Comment out more
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
  - [ ] Debug the code and modify it
  - [ ] Make README.md about tag_graph and train


- [ ] Improve tag_graph step by step
  - [x] Design the architexture
  * Saved rough overview about graph update methods in graph_update_methods.md
  - [x] Implement code in train_en.ipynb Update Graph section refering to graph_update_methods.md. Input: tag_tree_recs -> Output: tag_graph.
  - [x] Debug the code and get tag_graph.json
  - [ ] Probably it's not working good yet. Debug and test it.
  - [ ] Recode search_key and assign_tag with tag_graph
  - [ ] Get evaluation results from the tag_graph

- [ ] Implement gpt-oss instead of qwen7b (qwen7b has sometimes lower performance in tag generation)

- [ ] Train reward model
  - [ ] (Mingk) Make system to set aruguana as test dataset
  - [ ] Code to train rm for tag search
  - [ ] Add more dataset
  - [ ] Advanced DPO batching
  - [ ] GPU parallel for training with deepspeed












