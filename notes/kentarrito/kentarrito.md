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
  - [ ] Make employment contract

`git clone --branch develop https://github.com/kyotoai/RMSearch.git`

- [x] Debug rmsearch/train directory
  - [x] Make rmsearch_test.ipynb
  - [x] Process data is not what it's expected
  - [x] Run all the code and check if it's working
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
    - [x] Add chat_history.md. I want to save all the chat history to codex to some directory. Let's make one in notes.

- [ ] Debug rmsearch/tree/
  - [x] generate_tag.py
  - [x] embed_tags.py
  - [ ] build_representative_tags.py
  * there is no file to create tag_tree_recs from tag_tree and tag_embeddings using hirarchical_kmeans
  - [x] Create make_tag_tree.py
  - [ ] Debug make_tag_tree.py


## Oct 10, 11, 13

- [x] Make checkpoint branch -> Merge pull request #6

- [ ] Make rmsearch/graph
  - [x] Think about /notes/reference/graph_update_methods.md
  * there are 3 layers to optimize graph, improving tag, adding new edge and training rm for searching tags
  * for improving tag, llm should make extra representative tags and rm should pick some which fits children and conflicts with other children
  * for adding new edge, using retrieval dataset to find relevant but distant tags and connecting their parents to each tag is good way to update conflicted tag graph
  - [ ] Think what files to build
  * generate_tag.py, embed_tags.py, make_tag_graph.py, build_representative_tags_v2.py, search_key_graph.py, assign_key_graph.py
    - [x] Decide graph architecture (refer to ## make_tag_graph.py)
    * parquet is efficient to search
    * it basically saves {node -> children} for all node
    * if adding information about edge (like weight), need to add edge.parquet or edges column
    * BE CAREFUL THAT there can be mutual nodes. Search function should care about it.
    - [x] Start from writing ## build_representative_tags_v2.py in the README.md
  - [ ] Make prompt to code first sample in rmsearch/graph
  ```

  ```

## Oct 13, 14

- [ ] Make rmsearch/agents
  - [x] Design workflow and make readme in rmsearch/agents
  - [x] codex generates it based on the readme.
  - [ ] Debug them
    - [x] make_agents.py
    - [x] make_evaluation_dataset_code.py
      - [x] add algorism to readme
      - [x] modify the file with the algorism
      - [x] debug it
      * -> This only generates easy problems. Modify prompt to make more difficult and unobvious problems
      - [x] Improve prompt in make_evaluation_dataset_code_v1.py -> Created v2
      - [x] Improve it by implementing gpt api call and by summarizing other files with llm


## Oct 15

- [ ] Develop rmsearch/graph
  - [x] Modify rmsearch/graph/README.md
  - [x] Code files by codex
  - [ ] Debug them

* prakhar should be doing tag search dataset generation. Make readme for that and finish debugging graph thingy before that. 
* I gotta probably assign job to cameron on this Saturday. Need to make sure that llm_inference.py work on the agents.

* build_representative_tag_v2 has problem. All the tags become general


## Oct 16

Rough Roadmap From Now

- [ ] Update LLM call system (rmsearch/utils)
  - [ ] Implement gpt-oss instead of qwen (qwen7b has sometimes lower performance in tag generation)
  - [ ] Implement vllm serve
  - [ ] Implement llm api call
  - [ ] Implement rm api call
  - [ ] update rmsearch.py

- [ ] Improve Graph system
  - [ ] Graph topology optimization

- [ ] Train reward model for rmsearch paper
  - [ ] Make system to set aruguana as test dataset
  - [ ] Code to add tag search dataset
  - [ ] Add more dataset (rmsearch/train/process_data.py: modify argument and get dataset from multiple source)
  - [ ] Advanced DPO batching
  - [ ] GPU parallel for training with deepspeed

- [ ] Agents system
  - [ ] Make baseline to create agent system (rmsearch/agents/)
  - [ ] Make concrete agents
    - [ ] Agents for realtime knowledge graph update
    - [ ] Agents for excel analysis
    - [ ] Agents for code analysis

- [ ] Goals
  - [ ] RMSearch paper
    - [ ] Achieve better nDCG than e5-mistral by reward model without graph (reranker) -> minimal result to write a paper
    - [ ] Achieve better nDCG than e5-mistral by reward model with graph (rmsearch) -> this is something new. Favorable result for the paper.
  - [ ] SEIMEI paper
    - [ ] Get good evaluation for any task


kentarrito todo
- [x] Make system to set arguana as test dataset 1h
- [x] Debug all the training 2h  -> Pass this to Mingk tmrw morning and get more dataset.
- [x] Make better rmtrain system
  ```
  Now rmsearch/rmtrain.py and rmsearch/train/lora_example.py are so complex and hard to understand. I want you to make it much simpler. Follow
  1. I don't need rmtrain.py anymore. Migrate all important functions to lora_example.py
  2. Implement wandb so that I can track training history.
  3. Remove train_ids, test_ids thingy. That's probably causing bugs. Prepare train and test dataset separately.
  4. All update argument of lora_example.py for the modification. Also update train/README.md for your modification.
  ```


## Oct 17

- [ ] Make advanced DPO coding 3h -> Pass this to prakhar and conduct training with different batch size, 
  - [x] CustomRewardTrainer
  - [x] Make dataset_list.json
  - [ ] Adjust functions in CustomRewardTrainer

- [ ] Make baseline to create agent system (rmsearch/agents/) 1day -> pass it to cameron
- [ ] 



runpod: /workspace/kentarrito/exp1 -> lora_example.py, README.md, prepare_arguana_dataset.md
        /workspace/kentarrito/exp2 -> custom_trainer_lora_example.py, advanced_dpo_batching.md


## Oct 18

- [ ] Create better dataset accumulation system
- [x] Make train/READMEs for processing both dataset and recreate the result




## Oct 19

- [x] Finish adpo training debug -> leave it to prakhar 
  - [ ] Fix train cuda oom error
  - [ ] Fix num_gpu = 2 error
- [x] Make contract paper for Juan

- [x] relevance_record now take a lot of memory space, modifty this for rm too


## Oct 20

- [ ] Start creating YC application
  - [ ] Roughly created personal website
- [ ] Focus on agent rmsearch demo
  - [ ] Decide how to make excel finding project


## Oct 21

- [ ] Generating excel sheet by generator functions created by GPT5
  - [ ] Make prompt to create it
  - [ ] Gather 1000 datasets with 200 variable generators

- [ ] Making a system to automatically add agents and improve systems
  - [x] It might be better to create it in SEIMEI -> It's good for customizing and experiment. let's go with SEIMEI
  - [ ] vllm_serve for both generate & reward needed (because AscynEngine is not working now)
  - [ ] Access and analyze folder directly without processing folder
  - [ ] Inference -> Log
  - [ ] Log -> Agents
  - [ ] Agents -> Inference




## Oct 24

- [x] Create rmsearch/evaluation/process_data.py
  * Specify dataset, download it from huggingface and make pair.csv, query.json, key.json
  * query.json, key.json: list of query and key
  * pair.csv: query_id, key_id

- [x] Create rmsearch/evaluation/embed.py
  * from query.json and key.json, get relevance matrix of them
  * from the relevance matrix, create relevance_dict_embed.json
  * relevance_dict_embed.json :
  [
    {
      "query_id": , "key_ids":[]
    }
  ]
  * inside "keys", top relevant key ids to query with the query_id is there in order.
  * the number of keys are 100 in default.
  * Refer to embed_tags.py for how to make embedding.

- [x] Create rmsearch/evaluation/rerank.py
  * from relevance_dict_embed.json created, rerank the key_ids using examples/train_en.ipynb: Evaluation > Without Graph section? Output relevance_dict_rerank.json at the end.


## Oct 25

- [x] Add beir_to_pairs.py
- [x] Adjust input of embed.py, rerank.py and retrieval.py according to the dataset/beir_to_pair.py and update the readme.
- [x] Adjust rerank.py so that it generates output like 
    ```
    [
      {
        "query_id": , "key_ids":[], "pre_key_ids":[], "relevance":[], "positive_key_ids":[]
      }
    ]
    ```
    * add argument for top-k to extract top-k relevant "key_ids" from "pre_key_ids". "pre_key_ids" corresponds to "key_ids" in relevance_dict_embed.json. set top-k 10 in default.

- [x] Add more details about each file's argument, output, output example.




### download gpt-oss

```
from huggingface_hub import snapshot_download
import os

# Set your local directory path
local_dir = "./gpt-oss-20b"

# Download the repository, excluding 'original/' and 'metal/' directories
snapshot_download(
    repo_id="openai/gpt-oss-20b",
    local_dir=local_dir,
    ignore_patterns=["original/*", "metal/*"],
    # repo_type="model"
)

print(f"Download complete! Files saved to: {local_dir}")
```


### vllm with GPT oss

The model files are already downlaoded and just run
```
vllm serve /workspace/gpt-oss-20b     --host 0.0.0.0     --port 7000
```

from workspace, this will start the vllm server with GPT oss

here is s sample python inference code with API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7000/v1",
    api_key="EMPTY"
)

result = client.chat.completions.create(
    model="./gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what data vs task parallelism is."}
    ]
)

print(result.choices[0].message.content)
```


## Oct 27

- [x] Improve blogs 1
  * Put focus on the difference between normal dpo and advanced-batched dpo (in adpo, loss is calculated more combinations from each batch, so model can update weights from more comparison result. Also, model gets 1 input from each chosen key to compare them with 5 sampled rejected keys. In this sense, it supresses over-learning much more than scattering 5 same chosen keys over the dataset.) 
  * You must not write any code or command. Please explain the overview of how dataset is created plainly with some example content.
  * I added adpo.jpeg and dpo.png. This has evaluation accuracy for each training. Add pitcure to each blog.
  * You should make 5 sections, Summary (Overview), Make DPO Dataset, Make Advanced Batching Dataset, Training and Experiment.

## Oct 28

- [x] Add more dpo_pairs -> create judge_adpo_dataset.py

- [x] Modify judge_adpo_dataset.py
  * Set limit of each sentence for llm to judge. apply [:4000] to each sentence.
  * Change the judge prompt so that it can also generate tie. Add dpo_pairs only when there is a meaningful gap between sentence1 and sentence2.
  * Let's skip llm judgement for correspond_keys. Automatically add all possible dpo_pairs where correspond_keys win sampled_keys.

-> I found almost all pairs are tied. Need to make a better questions. 

Possible plan
* Include irr-questions
* Change queries and 


## Oct 29

- [ ] Make more queries
  * See inside evaluation
  * Make more general questions




scifact example

query:  Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA.

posi key:  IL-12 and IFN-gamma positively regulate each other and type 1 inflammatory responses, which are believed to cause tissue damage in autoimmune diseases. We investigated the role of the IL-12/IFN-gamma (Th1) axis in the development of autoimmune myocarditis. IL-12p40-deficient mice on a susceptible background resisted myocarditis. In the absence of IL-12, autospecific CD4(+) T cells proliferated poorly and showed increased Th2 cytokine responses. However, IFN-gamma-deficient mice developed fatal autoimmune disease, and blockade of IL-4R signaling did not confer susceptibility to myocarditis in IL-12p40-deficient mice, demonstrating that IL-12 triggers autoimmunity by a mechanism independent of the effector cytokines IFN-gamma and IL-4. In conclusion, our results suggest that the IL-12/IFN-gamma axis is a double-edged sword for the development of autoimmune myocarditis. Although IL-12 mediates disease by induction/expansion of Th1-type cells, IFN-gamma production from these cells limits disease progression.

top relevant key 1:  Experimental autoimmune myocarditis (EAM) represents a Th17 T cell-mediated mouse model of postinflammatory heart disease. In BALB/c wild-type mice, EAM is a self-limiting disease, peaking 21 days after alpha-myosin H chain peptide (MyHC-alpha)/CFA immunization and largely resolving thereafter. In IFN-gammaR(-/-) mice, however, EAM is exacerbated and shows a chronic progressive disease course. We found that this progressive disease course paralleled persistently elevated IL-17 release from T cells infiltrating the hearts of IFN-gammaR(-/-) mice 30 days after immunization. In fact, IL-17 promoted the recruitment of CD11b(+) monocytes, the major heart-infiltrating cells in EAM. In turn, CD11b(+) monocytes suppressed MyHC-alpha-specific Th17 T cell responses IFN-gamma-dependently in vitro. In vivo, injection of IFN-gammaR(+/+)CD11b(+), but not IFN-gammaR(-/-)CD11b(+), monocytes, suppressed MyHC-alpha-specific T cells, and abrogated the progressive disease course in IFN-gammaR(-/-) mice. Finally, coinjection of MyHC-alpha-specific, but not OVA-transgenic, IFN-gamma-releasing CD4(+) Th1 T cell lines, together with MyHC-alpha-specific Th17 T cells protected RAG2(-/-) mice from EAM. In conclusion, CD11b(+) monocytes play a dual role in EAM: as a major cellular substrate of IL-17-induced inflammation and as mediators of an IFN-gamma-dependent negative feedback loop confining disease progression.

top relevant key 2:  BACKGROUND Interferon-gamma (IFN-gamma) is an essential cytokine in the regulation of inflammatory responses in autoimmune diseases. Little is known about its role in inflammatory heart disease. METHODS AND RESULTS We showed that IFN-gamma receptor-deficient mice (IFN-gammaR(-/-)) on a BALB/c background immunized with a peptide derived from cardiac alpha-myosin heavy chain develop severe myocarditis with high mortality. Although myocarditis subsided in wild-type mice after 3 weeks, IFN-gammaR(-/-) mice showed persistent disease. The persistent inflammation was accompanied by vigorous in vitro CD4 T-cell responses and impaired inducible nitric oxide synthase expression, together with evidence of impaired nitric oxide production in IFN-gammaR(-/-) hearts. Treatment of wild-type mice with the nitric oxide synthetase inhibitor N:-nitro-l-arginine-methyl-ester enhanced in vitro CD4 T-cell proliferation and prevented healing of myocarditis. CONCLUSIONS Our data provide evidence that IFN-gamma protects mice from lethal autoimmune myocarditis by inducing the expression of inducible nitric oxide synthase followed by the downregulation of T-cell responses.

top relevant key 3:  BACKGROUND Interleukin (IL)-12 exerts a potent proinflammatory effect by stimulating T-helper (Th) 1 responses. This effect is believed to be mediated primarily through the activation of STAT4 and subsequent production of interferon (IFN)-gamma. Methods and Results- We examined the role of IL-12 receptor (IL-12R) signaling in the development of murine experimental autoimmune myocarditis (EAM) induced by cardiac myosin immunization. Both IL-12Rbeta1-deficient mice and STAT4-deficient mice were resistant to the induction of myocarditis. Treatment with exogenous IL-12 exacerbated disease. We questioned whether IFN-gamma is required for the disease-promoting activity of IL-12. On the contrary, we found that IFN-gamma suppresses EAM. Lack of IFN-gamma due to either depletion with an antibody or a genetic deficiency exacerbated myocarditis. Spleens from IFN-gamma-deficient mice immunized with cardiac myosin showed increased cellularity; greater numbers of CD3+, CD4+, CD8+, and IL-2-producing cells; and heightened ability to produce cytokines on stimulation in vitro. Treatment of mice with recombinant IFN-gamma suppressed the development of myocarditis. CONCLUSIONS IL-12/IL-12R/STAT4 signaling promotes the development of EAM. In contrast, IFN-gamma plays a protective role. The disease-limiting effects of IFN-gamma might be explained by its ability to control the expansion of activated T lymphocytes.

query: 0-dimensional biomaterials show inductive properties.
correspond_key: "Nanotechnologies are emerging platforms that could be useful in measuring, understanding, and manipulating stem cells. Examples include magnetic nanoparticles and quantum dots for stem cell labeling and in vivo tracking; nanoparticles, carbon nanotubes, and polyplexes for the intracellular delivery of genes/oligonucleotides and protein/peptides; and engineered nanometer-scale scaffolds for stem cell differentiation and transplantation. This review examines the use of nanotechnologies for stem cell tracking, differentiation, and transplantation. We further discuss their utility and the potential concerns regarding their cytotoxicity."

All hematopoietic stem cells segregate their chromosomes randomly.
Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.


* Dataset collect

* Biomedical
qiaojin/PubMedQA : question, context, answer -> generate similar answer2 with llm
BeIR/bioasq-generated-queries : title (sometimes not good), text -> extract only runnable data

* Finance
next-tat/TAT-QA : 
some other. But need to download them to see

* Legal
coastalcph/lex_glue : context -> query, context 2

* 


* General
smollm


## Oct 30

`ps aux | grep python`

- [x] Make make_query_and_less_relevant_keys_recs.py

- [x] Debug 1-2 make_query_and_less_relevant_keys_recs.py
  * modify so that it generates closer keys

- [x] Implement gpt-oss


## Oct 31

- [x] Make make_query_dpo_pairs.py
- [ ] Debug make_query_dpo_pairs.py

- [ ] Make a serve vllm for faster debug
- [ ] Add readme with api key setting
- [ ] 



* When killing nohop
`ps aux | grep accelerate`
`kill -9 <pid>`


## Nov 5

- [x] Delete rmtrain.py

- [x] API Call in rmsearch/rmsearch.py
  1. use utils/vllm_reward.py for search function
  2. handle both request types, string list and message list
  3. after RMSearch installation, make it available just by use command `uvicorn rmsearch:app --host 0.0.0.0 --port 8000`

- [x] Debug 1 API Call in rmsearch/rmsearch.py

- [x] Write detailed example usages as comment out in rmsearch.py. Also write basic usage in rmsearch/README.md

- [x] Debug 2 API Call in rmsearch/rmsearch.py
  * converted-model

- [x] Debug 3 API Call in rmsearch/rmsearch.py
  * add pooling_task
