import json

def transform_json(input_filename, output_filename):
    """
    Reads a JSON file (list of objects) and transforms it into a new 
    JSON format (map of query_id to key_id/relevance scores).

    Args:
        input_filename (str): The name of the input JSON file.
        output_filename (str): The name of the output JSON file.
    """
    try:
        # 1. Read the input JSON file
        with open(input_filename, 'r', encoding='utf-8') as infile:
            input_data = json.load(infile)

        # Initialize the output structure
        output_data = {}

        # 2. Process each dictionary/object in the input list
        for item in input_data:
            # Extract the required fields
            query_id = str(item['query_id'])  # Use str for JSON keys
            key_ids = item['key_ids']
            relevance_scores = item['relevance']

            # Check if lengths match before zipping
            if len(key_ids) != len(relevance_scores):
                print(f"Warning: query_id {query_id} has mismatched lengths for key_ids and relevance. Skipping.")
                continue

            # Create the inner dictionary: "key_id": relevance_score
            inner_map = {}
            for key_id, score in zip(key_ids, relevance_scores):
                # key_id needs to be a string for a proper JSON key
                inner_map[str(key_id)] = score
            
            # Assign the inner map to the query_id in the final output
            output_data[query_id] = inner_map

        # 3. Write the output JSON file
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            # Use indent=4 for a human-readable, pretty-printed output
            json.dump(output_data, outfile, indent=4)

        print(f"✅ Transformation complete. Data written to {output_filename}")

    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_filename}' not found.")
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{input_filename}'. Check file format.")
    except KeyError as e:
        print(f"❌ Error: Missing required key in input data: {e}. Check the input structure.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# --- Configuration ---
INPUT_FILE = "/workspace/Mingkwan/RMSearch/beir_out/scifact/relevance_dict_rerank1240.json"
OUTPUT_FILE = "/workspace/Mingkwan/RMSearch/beir_out/scifact/relevance_dict_adj_rerank1240.json"

# Run the transformation
transform_json(INPUT_FILE, OUTPUT_FILE)