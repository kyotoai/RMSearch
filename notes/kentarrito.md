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

