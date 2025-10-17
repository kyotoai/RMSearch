from pathlib import Path
import json
import traceback
query_key_set_path = "data/arguana/adpo_sampled_query_key_set.json"
output_path = Path("exp2/dataset_list_test.json")

with open(query_key_set_path) as f:
  query_key_set = json.load(f)

def _format_prompt(query: str, key: str) -> str:
  return (
      "Give me relevant score between query and sentence;\n\n"
      f"Query:{query}\n\n"
      f"Sentence:```{key}```"
  )

dataset_list = []
n_error = 0
for query_key_dict in query_key_set:
  try:
    query_id = query_key_dict["query_id"]
    query = query_key_dict["query"]
    correspond_keys = query_key_dict["correspond_keys"]
    correspond_key_ids = query_key_dict["correspond_key_ids"]
    sampled_keys = query_key_dict["sampled_keys"]
    sampled_key_ids = query_key_dict["sampled_key_ids"]
    keys = correspond_keys + sampled_keys
    key_ids = correspond_key_ids + sampled_key_ids

    batch = []
    for i, key in enumerate(keys):
      batch.append({"msg": [{"role": "user", "content": _format_prompt(query, key)}], "query_id":query_id, "key_id":key_ids[i]})

    dpo_pairs = []
    for c_id in range(len(correspond_key_ids)):
      for s_id in range(len(sampled_key_ids)):
        dpo_pairs.append([c_id, s_id + len(correspond_key_ids)])

    dataset_list.append(
        {
            "batch": batch,
            #[
            #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[1])}], "query_id":query_id, "key_id":},
            #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[0])}], "query_id":query_id, "key_id":},
            #  {"msg": [{"role": "user", "content": _format_prompt(query, keys[0])}], "query_id":query_id, "key_id":}
            #],
            "dpo_pairs": dpo_pairs,
            #[
            #  [0,1],  # [(chosen_msg_id), (rejected_msg_id)]
            #  [0,2],
            #  [1,2]
            #]
        }
    )
  
  except Exception as e:
    n_error += 1
    traceback.print_exc()
    print(e)

print("n_error: ", n_error)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(dataset_list, ensure_ascii=False, indent=2))
print(f"Wrote dataset list with {len(dataset_list)} entries to {output_path}")