# Communication Among Colaborators

## Sep 11 2025

kenta -> mingk

- [ ] I said I would pass you relevance.pt but actually rmsearch doesn't complete all the elements inside the matrix. So I pass you relevance_dict.json instead, which should look like

    [{"query":questions[0], "query_id":0, "keys":[{"relevant_id":8,"key":sentences[8],}, ...]}, ...]

    In each dictionary, it has query and top relevant keys. This list of dictionary has enough information to derive evaluation metrics so use this instead.

