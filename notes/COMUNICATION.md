# Communication Among Colaborators

## Sep 11 2025

kenta -> mingk

- [x] I said I would pass you relevance.pt but actually rmsearch doesn't complete all the elements inside the matrix. So I pass you relevance_dict.json instead, which should look like

    [{"query":questions[0], "query_id":0, "keys":[{"relevant_id":8,"key":sentences[8],}, ...]}, ...]

    In each dictionary, it has query and top relevant keys. This list of dictionary has enough information to derive evaluation metrics so use this instead.

- [x] I wanted to make the output from trained reward model and let you code and debug with it, but I need to modify train_en more to do it. So instead I want you to use the output from embedding model in path f"/workspace/RMS_exp/data/{data_name}/sentences_relevant_to_questions.json"

    Note that:
    sentences_relevant_to_questions.json <- Made by base_model search (I used embedding in this case)
    e.g. [{"query":, "query_id":, "correct_id":, "keys":[{"key":, "key_id":,}, ...]}, ... ]

    relevance_dict <- Made by trained model
    e.g. [{"query":questions[0], "query_id":0, "keys":[{"relevant_id":8,"key":sentences[8],}, ...]}, ...]

    To derive evaluation metrics, all you need is correct_id and key_id. Comparing these two, you can obtain the metrics


## Sep 16

kentarrito -> mingk

- [ ] We need to implement deepspeed to train_en so that we can train more parameters in models, which makes the model much better!! \
    ToDo
    - [ ] Read docs about deepspeed

        Refs about the concept:
            https://huggingface.co/docs/transformers/en/perf_train_gpu_many
            https://huggingface.co/docs/transformers/en/deepspeed

        Code:
            https://www.deepspeed.ai/getting-started/
            https://github.com/kyotoai/RMSearch/blob/develop/notes/deepspeed.md (note when I once tried to implement it on runpod)
        
    ↑ If you understand the basics of this, I will tell you how to implement this.

    - [ ] Try to implement it on runpod with /examples/deepspeed_test2.py (this code is the most stable one. need a bit more debug)
    -> I'm gonna tell you how to use it face to face


## Sep 30

