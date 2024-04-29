import time
import os
import json
from collections import defaultdict
from bm25_pt import BM25  # Changed to use bm25_pt library
import torch
import pandas as pd
import numpy as np
import tqdm

# Load data from JSON file
def load_data(file_path, num_authors):
    data = []
    pbar = tqdm.auto.tqdm("load_data", colour="blue", leave=False, total=num_authors)
    with open(file_path, 'r', encoding='utf-8') as file:
        count = 0
        for line in file:
            data.append(json.loads(line))
            count += 1
            pbar.update(1)
            if count == num_authors:
                break
        # data = json.load(file)
    return data

def save_results_to_json(scores, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

def save_results_to_csv(scores, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)
    df = pd.DataFrame(scores)
    df.to_csv(full_path, index=False)

def sample_docs(docs, window_size=16):
    docs = np.random.choice(docs, window_size)
    return docs

def calculate_bm25_all_pairs(data_preprocessed, batch_size, topk):

    all_results = []
    authors = list(data_preprocessed.keys())
    # texts = ["\n".join(docs) for docs in data_preprocessed.values()]
    texts = ["\n".join(sample_docs(docs)) for docs in data_preprocessed.values()]
    topk = min(topk, len(authors))

    print("number of authors", len(authors))
    print("average document length by chars", sum(len(docs) for docs in texts) // len(authors))

    start_time = time.time()
    bm25 = BM25(device='cuda')
    bm25.index(texts)
    print("indexing done on queries")
    doc_scores = bm25.score_batch(texts, batch_size)
    print("bm25 done on targets")

    top_values, top_indices = torch.topk(doc_scores, topk, dim=1)
    top_values, top_indices = top_values.tolist(), top_indices.tolist()

    # for query_id in range(len(top_values)):
    #     data_structure = {
    #         "query_author_id": authors[query_id],
    #         "top_n_bm25_scores": top_values[query_id],
    #         "target_author_ids": [authors[target_id] for target_id in top_indices[query_id]],
    #     }
    #     all_results.append(data_structure)

    all_results = top_indices

    current_time = time.time()
    print(f"Total running time: {current_time - start_time} seconds")

    return all_results

def main():
    num_authors = 50000
    batch_size = 4
    topk = 512
    file_path = '/share/rush/authorship/data/all_pii/all_data_pii_removed.jsonl'  # Update with your actual path
    data = load_data(file_path, num_authors)
    data_preprocessed = {}
    document_counts = defaultdict(int)

    for entry in data:
        author_id = entry['author_id']
        # Count the number of documents (i.e., items in the 'syms' list) for this author
        document_counts[author_id] += len(entry['syms'])

    for entry in data:
        author_id = entry['author_id']
        document_counts[author_id] += len(entry['syms'])
        data_preprocessed[author_id] = entry['syms']

    scores = calculate_bm25_all_pairs(data_preprocessed, batch_size, topk)

    directory = './reddit'
    filename = f'bm25_result_{num_authors}-authors.csv'
    save_results_to_csv(scores, directory, filename)

if __name__ == "__main__":
    main()