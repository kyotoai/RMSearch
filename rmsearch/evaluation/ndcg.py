import time 
import ijson
import math

file_path = '/workspace/Mingkwan/RMSearch/beir_out/scifact/relevance_dict_rerank.json'

item_count = 0
items_to_print = 300 

expected_id = []
retrieved_id = []
not_found_querie_ids = []
rank_scores = []
IDCG = 1
nDCG = 0
DCG = 0

def compute_DCG(pos):
    """
    Compute DCG score
    """
    # print(rel)
    if pos == 0:
        DCG = 0
    else:
        DCG = 1/math.log2(pos+1)
    return DCG 
def evaluate_ndcg(file_path):
    try:
        with open(file_path, 'rb') as file:
            print(f"Reading first {items_to_print} items from '{file_path}'...")
            start_time = time.time()
            nDCG = 0
            IDCG = 1
            for item in ijson.items(file, 'item'):
                if 'query_id' in item:
                    print(f"query id: {item['query_id']}")
                    found_match = False
                if 'positive_key_ids' in item:
                    print(f"correct: {item['positive_key_ids']}")
                    c_id = (item['positive_key_ids'])[0]
                    expected_id.append(c_id)

                if 'key_ids' in item:
                    key_list=item["key_ids"]
                    list_size = len(key_list)
                    print(f"The keys have a total of {len(key_list)} items")
                    print(f"keys: {item['key_ids']}")
                    for index, key in enumerate(key_list):
                        print(f"this is the id for this key: {key}")
                        print(f"this is the id for the correct id: {c_id}")
                        if key == c_id:
                            retrieved_id.append(key)
                            found_match = True
                            print("FOUND MATCH")
    
                            position = index + 1
                            score = int((list_size+1-position)/list_size)
                            rank_scores.append(score)
                            print(f"retrieved ids right now is: {retrieved_id}")
                            break
 
                if not found_match:
                    not_found_querie_ids.append(item['query_id'])
                    score = 0
                    position = 0
                    rank_scores.append(score)
    
                ##Calculate nDCG
                
                DCG = compute_DCG(position)
                print(f"DCG: {DCG}")
                nDCG = nDCG + DCG/IDCG/items_to_print
            elapsed_time = time.time() - start_time
            print(f"✅ Process completed in {elapsed_time:.2f} seconds.")
            print(f"The nDCG score is {nDCG}")
            
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    evaluate_ndcg(file_path)