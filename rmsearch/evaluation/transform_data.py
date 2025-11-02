import json
import csv
import argparse
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file and return list of records"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def convert_queries(input_file, output_file):
    """
    Convert queries JSONL to CSV format
    Input: {"_id": "PLAIN-3", "text": "Breast Cancer...", "metadata": {...}}
    Output: id,text
    """
    print(f"Converting queries from {input_file} to {output_file}")
    
    queries = load_jsonl(input_file)
    
    # Create mapping from _id to numeric id
    id_mapping = {}
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text'])
        
        for idx, query in enumerate(queries):
            original_id = query['_id']
            text = str(query['text'])
            
            # Store mapping
            id_mapping[original_id] = idx
            
            writer.writerow([idx, text])
    
    print(f"✓ Converted {len(queries)} queries")
    return id_mapping

def convert_corpus(input_file, output_file):
    """
    Convert corpus JSONL to CSV format
    Input: {"_id": "MED-10", "title": "...", "text": "...", "metadata": {...}}
    Output: id,text
    """
    print(f"Converting corpus from {input_file} to {output_file}")
    
    corpus = load_jsonl(input_file)
    
    # Create mapping from _id to numeric id
    id_mapping = {}
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text'])
        
        for idx, doc in enumerate(corpus):
            original_id = doc['_id']
            # Combine title and text
            title = doc.get('title', '')
            text = doc.get('text', '')
            combined_text = f"{title} {text}".strip() if title else text
            
            # Store mapping
            id_mapping[original_id] = idx
            
            writer.writerow([idx, combined_text])
    
    print(f"✓ Converted {len(corpus)} documents")
    return id_mapping

def convert_qrels(input_file, output_file, query_mapping, corpus_mapping):
    """
    Convert qrels TSV to CSV format
    Input: query-id	corpus-id	score (TSV)
    Output: query_id,key_id,score (CSV)
    """
    print(f"Converting qrels from {input_file} to {output_file}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['query_id', 'key_id', 'score'])
            
            # Skip header
            next(f_in)
            
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                query_id_str = parts[0]
                corpus_id_str = parts[1]
                score = parts[2]
                
                # Map to numeric IDs
                if query_id_str in query_mapping and corpus_id_str in corpus_mapping:
                    query_id = query_mapping[query_id_str]
                    key_id = corpus_mapping[corpus_id_str]
                    writer.writerow([query_id, key_id, score])
                    count += 1
                else:
                    print(f"Warning: Missing mapping for {query_id_str} or {corpus_id_str}")
    
    print(f"✓ Converted {count} qrels entries")
    
def main():
    parser = argparse.ArgumentParser(description='Convert BEIR dataset files to CSV format')
    parser.add_argument('--queries', required=True, help='Input queries JSONL file')
    parser.add_argument('--corpus', required=True, help='Input corpus JSONL file')
    parser.add_argument('--qrels', required=True, help='Input qrels TSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BEIR Dataset to CSV Converter")
    print("=" * 60)
    
    # Convert queries
    queries_output = output_dir / 'query.csv'
    query_mapping = convert_queries(args.queries, queries_output)
    
    print()
    
    # Convert corpus
    corpus_output = output_dir / 'key.csv'
    corpus_mapping = convert_corpus(args.corpus, corpus_output)
    
    print()
    
    # Convert qrels
    qrels_output = output_dir / 'pair.csv'
    convert_qrels(args.qrels, qrels_output, query_mapping, corpus_mapping)
    
    print()
    print("=" * 60)
    print("Conversion Complete!")
    print(f"Output files saved to: {output_dir}")
    print(f"  - {queries_output.name}")
    print(f"  - {corpus_output.name}")
    print(f"  - {qrels_output.name}")
    print("=" * 60)

if __name__ == '__main__':
    main()
    
# python trasnform_data.py \
#     --queries /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/queries.jsonl \
#     --corpus /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/corpus.jsonl \
#     --qrels /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/qrels/test.tsv \
#     --output-dir /workspace/Mingkwan/RMSearch/rmsearch/evaluation/datasets/nfcorpus/csv_files 