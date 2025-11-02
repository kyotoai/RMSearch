import json
import os
import pandas as pd

with open("relevance_dict_rerank1240.json") as f:
    relevance_dict_list = json.load(f)
query_df = pd.read_csv("query.csv")
key_df = pd.read_csv("key.csv")

n_multi_positive = 0
succeed = 0
fail = 0
no_chance = 0
id_list = [0 for _ in range(10)]
for relevance_dict in relevance_dict_list:
    positive_key_ids = relevance_dict["positive_key_ids"]
    if len(positive_key_ids)==1:
        pre_key_ids = relevance_dict["pre_key_ids"]
        key_ids = relevance_dict["key_ids"]
        query_id = relevance_dict["query_id"]
        posi_key_id = positive_key_ids[0]
        if posi_key_id in pre_key_ids:
            if posi_key_id in key_ids:
                id = key_ids.index(posi_key_id)
                id_list[id] += 1
                succeed += 1
            else:
                print()
                print("query: ", query_df.iloc[query_id]["text"])
                print("posi key: ", key_df.iloc[posi_key_id]["text"])
                for i, rel_key_id in enumerate(key_ids):
                    print(f"top relevant key {i+1}: ", key_df.iloc[rel_key_id]["text"])
                fail += 1
        else:
            no_chance += 1
    else:
        n_multi_positive += 1

print(succeed, fail, no_chance, id_list)
