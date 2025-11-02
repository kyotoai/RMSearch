import json

def transform_query_data(input_data):
    """
    Transform query data from input format to output format.
    
    Args:
        input_data: List of query objects or single query object
    
    Returns:
        Dictionary with transformed data
    """
    result = {}
    
    # Handle both single object and list of objects
    if isinstance(input_data, dict):
        input_data = [input_data]
    
    for query in input_data:
        query_id = query.get("query_id")
        key_ids = query.get("key_ids", [])
        relevant_scores = query.get("embed_relevances", [])
        
        # Create dictionary mapping key_id to relevant score
        query_results = {}
        for key_id, score in zip(key_ids, relevant_scores):
            query_results[key_id] = score
        
        result[query_id] = query_results
    
    return result


def main():
    # Example usage
    input_file = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/new_emb_results.json"  # Change to your input file path
    output_file = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/new_emb_results_adj.json"  # Change to your output file path
    
    # Read input data
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    # Transform data
    output_data = transform_query_data(input_data)
    
    # Write output data
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Transformation complete! Output saved to {output_file}")
    print(f"Processed {len(output_data)} queries")


if __name__ == "__main__":
    # If you want to test with your example data directly:
    # example_input = {
    #     "query_id": 0,
    #     "key_ids": [1383, 1377, 1382, 1, 1379],
    #     "relevant": [1, 1, 0.5, 0.8, 0.7],
    #     "positive_key_ids": None
    # }
    
    # # Transform example
    # result = transform_query_data(example_input)
    # print("Example output:")
    # print(json.dumps(result, indent=2))
    
    # Uncomment the line below to run with actual files
    main()