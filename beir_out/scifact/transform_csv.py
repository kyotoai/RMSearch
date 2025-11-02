import csv

def transform_csv(input_filename, output_filename):
    """
    Reads a CSV file and transforms it into a new CSV format.

    Args:
        input_filename (str): The name of the input CSV file.
        output_filename (str): The name of the output CSV file.
    """
    try:
        with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            fieldnames = ["query-id", "corpus-id", "score"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            # Write the header of the new CSV
            writer.writeheader()

            # Process each row
            for row in reader:
                new_row = {
                    "query-id": row['query_id'],
                    "corpus-id": row['key_id'],
                    "score": 1  # Always set the score to 1
                }
                writer.writerow(new_row)

        print(f"✅ Transformation complete. Data written to {output_filename}")

    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_filename}' not found.")
    except KeyError as e:
        print(f"❌ Error: Missing column in input file: {e}. Check your input file header.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# --- Configuration ---
INPUT_FILE = "/workspace/Mingkwan/RMSearch/beir_out/scifact/pair.csv"    # Replace with your actual input filename
OUTPUT_FILE = "/workspace/Mingkwan/RMSearch/beir_out/scifact/qrels.csv"  # The name for the new CSV file

# Run the transformation
transform_csv(INPUT_FILE, OUTPUT_FILE)