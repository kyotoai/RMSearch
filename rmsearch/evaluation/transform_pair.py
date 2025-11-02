import csv
from pathlib import Path
import argparse

def csv_to_tsv(input_csv_file, output_tsv_file):
    """
    Convert CSV format to TSV format, renaming columns from 
    query_id, key_id, score to query-id, corpus-id, score.
    
    Args:
        input_csv_file: Path to input CSV file
        output_tsv_file: Path to output TSV file
    """
    # Read CSV data
    rows = []
    with open(input_csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Write TSV file
    with open(output_tsv_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header with new column names
        writer.writerow(['query-id', 'corpus-id', 'score'])
        
        # Write data rows
        for row in rows:
            writer.writerow([
                row['query_id'],
                row['key_id'],
                1, #row['score']
            ])
    
    print(f"Conversion complete! TSV saved to {output_tsv_file}")
    print(f"Converted {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pair.csv into tsv file which is the direct input of ndcg.py.")
    parser.add_argument("--input-file", type=Path, required=True, help="Input csv file containing pair information.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output tsv file containing the key text.")
    args = parser.parse_args()
    # Example usage
    #input_file = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/csv_files/pair.csv"  # Change to your input file path
    #output_file = "/workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/new_emb_results_adj.tsv"  # Change to your output file path
    
    csv_to_tsv(args.input_file, args.output_file)