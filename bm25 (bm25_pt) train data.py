import time
import os
import json
from itertools import combinations
from collections import defaultdict
from bm25_pt import BM25  # Changed to use bm25_pt library
import torch

start_time = time.time()

# Load data from JSON file
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Tokenize documents
def tokenize(documents):
    return [doc.lower().split() for doc in documents]

def save_results_to_json(scores, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

def calculate_bm25_all_pairs(data_preprocessed, n=5):

    all_results = []
    author_pairs = combinations(data_preprocessed.keys(), 2)

    for author1, author2 in author_pairs:

        a1 = author1
        a2 = author2
        print("author1, author2: ", author1, author2)

        # print(len(data_preprocessed[author1]))
        # print(data_preprocessed[author1][0:5])

        # print(len(data_preprocessed[author2]))
        # print(data_preprocessed[author2][0:5])

        # Initialize BM25 with the documents from both authors
        # combined_documents = data_preprocessed[author1] + data_preprocessed[author2]
        # combined_documents_text = [" ".join(doc) for doc in combined_documents]
        # print("combined_documents_text: ", combined_documents_text)

        query_corpus = data_preprocessed[author1]
        print("len(query_corpus): ", len(query_corpus))

        bm25 = BM25()
        bm25.index(query_corpus)

        # print("data_preprocessed[author1]: ", data_preprocessed[author1])
        target_corpus = data_preprocessed[author2]
        print("len(target_corpus): ", len(target_corpus))

        doc_scores = bm25.score_batch(target_corpus)
        # print("doc_scores: ", doc_scores)
        # print("doc_scores.shape: ", doc_scores.shape)
        # print("doc_scores[0].shape: ", doc_scores[0].shape)

        # Use torch.topk to get the top n values and their indices for each row
        top_values, top_indices = torch.topk(doc_scores, n, dim=1)

        top_values, top_indices = top_values.tolist(), top_indices.tolist()

        # print("Top n values for each row:\n", top_values)
        # print("Indices of top n values for each row:\n", top_indices)

        for i in range(len(top_values)):
            # print("i: ", i)

            data_structure = {
                "query_author_id": author1,
                "query_document_id": i,
                "top_n_bm25_scores": top_values[i],
                "target_author_ids": author2,
                "top_n_bm25_document_ids": top_indices[i]
            }

            all_results.append(data_structure)
            # print("all_results[-1]: ", all_results[-1])

        # for doc_index, query in enumerate(data_preprocessed[author1]):
        #     query_text = " ".join(query)
        #     doc_scores = bm25.score(query_text)

        #     # Convert Tensor to list if necessary
        #     # Assuming doc_scores is the Tensor you need to convert
        #     # This is a placeholder; you'll need to adapt it to your specific use case
        #     if isinstance(doc_scores, torch.Tensor):
        #         doc_scores = doc_scores.tolist()  # This line converts Tensor to list

        #     target_scores = doc_scores[len(data_preprocessed[author1]):]

        #     top_n_indices_scores = sorted(enumerate(target_scores), key=lambda x: x[1], reverse=True)[:n]
        #     # print("author1, doc_index, [score for _, score in top_n_indices_scores], author2, [index for index, _ in top_n_indices_scores]: ", author1, doc_index, [score for _, score in top_n_indices_scores], author2, [index for index, _ in top_n_indices_scores])
        #     # print("top_n_indices_scores: ", top_n_indices_scores)

        #     data_structure = {
        #         "query_author_id": author1,
        #         "query_document_id": doc_index,
        #         "top_n_bm25_scores": [score for _, score in top_n_indices_scores],
        #         "target_author_ids": author2,
        #         "top_n_bm25_document_ids": [index for index, _ in top_n_indices_scores]
        #     }

        #     all_results.append(data_structure)
        #     # print("all_results[-1]: ", all_results[-1])
        # # print("all_results: ", all_results)
    

        print("ready to save!")
        directory = './result(bm25_pt)(batch_score)'
        # filename = 'bm25_results.json'
        filename = "bm25_results (batch_score)" + "_author1 (" + str(a1) + ") author2 (" + str(a2) + ")"

        save_results_to_json(all_results, directory, filename)
        current_time = time.time()
        print("saved!")
        print(f"Total running time: {current_time - start_time} seconds")


    return all_results

def main():
    file_path = 'EnronEmails_pii/train.jsonl'  # Update with your actual path
    data = load_data(file_path)
    data_preprocessed = {}
    document_counts = defaultdict(int)

    for entry in data:
        author_id = entry['author_id']
        # Count the number of documents (i.e., items in the 'syms' list) for this author
        document_counts[author_id] += len(entry['syms'])

    # Print the number of documents for each author
    print("Number of documents for each author:")
    for author_id, count in document_counts.items():
        print(f"Author {author_id}: {count} documents")
    

    for entry in data:
        author_id = entry['author_id']
        document_counts[author_id] += len(entry['syms'])
        # print("entry['syms']: ", entry['syms'])
        # tokenized_syms = tokenize(entry['syms'])
        # data_preprocessed[author_id] = tokenized_syms
        data_preprocessed[author_id] = entry['syms']

    # print("len(data_preprocessed[1]): ", len(data_preprocessed[1]))
    # for i in range(len(data_preprocessed[0])):
    #     print("data_preprocessed[0][i][0:6]: ", data_preprocessed[0][i][0:6])

    scores = calculate_bm25_all_pairs(data_preprocessed, 3)
    directory = './result(bm25_pt)'
    filename = 'bm25_results.json'

    save_results_to_json(scores, directory, filename)

if __name__ == "__main__":
    main()







# import heapq
# import os
# from rank_bm25 import BM25Okapi
# import json
# from itertools import combinations, product
# from collections import defaultdict

# # Load data from JSON file
# def load_data(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#     return data

# # Tokenize documents
# def tokenize(documents):
#     return [doc.lower().split() for doc in documents]


# def save_results_to_json(scores, directory, filename):
#     """
#     Saves the given scores to a JSON file.

#     Parameters:
#     - scores: Data to be saved.
#     - directory: The directory where the file should be saved.
#     - filename: The name of the file.

#     The function will create the directory if it doesn't exist.
#     """
#     # Ensure the directory exists
#     if not os.path.exists(directory):
#         os.makedirs(directory, exist_ok=True)

#     # Construct the full path to the file
#     full_path = os.path.join(directory, filename)

#     # Write the data to the JSON file
#     with open(full_path, 'w', encoding='utf-8') as f:
#         json.dump(scores, f, ensure_ascii=False, indent=4)

# def calculate_bm25_all_pairs(data_preprocessed, n=5):
#     all_results = []
#     author_pairs = combinations(data_preprocessed.keys(), 2)

#     for author1, author2 in author_pairs:

#         print("author1, author2: ", author1, author2)

#         query_documents = data_preprocessed[author1]
#         target_documents = data_preprocessed[author2]

#         # Initialize BM25 with the documents from the second author
#         bm25 = BM25Okapi(target_documents)
        
#         # Generate indices for the target documents
#         target_doc_ids = list(range(len(target_documents)))
#         # print("target_doc_ids: ", target_doc_ids)

#         for doc_index, query in enumerate(query_documents):
            
#             # print("doc_index: ", doc_index)
#             # print("query: ", query)

#             # Calculate BM25 scores for the current document against all target documents using get_batch_scores
#             doc_scores = bm25.get_batch_scores(query, target_doc_ids)
#             # print("doc_scores: ", doc_scores)

#             # Pair each score with its document index in the target documents
#             doc_scores_with_index = list(enumerate(doc_scores))

#             # Find the top N scores
#             top_n = heapq.nlargest(n, doc_scores_with_index, key=lambda x: x[1])  # Notice we're now using x[1] to sort by score

#             # Preparing data structure for the current document
#             data_structure = {
#                 "query_author_id": author1,
#                 "query_document_id": doc_index,
#                 "top_n_bm25_scores": [score for _, score in top_n],
#                 "target_author_ids": author2,
#                 "top_n_bm25_document_ids": [index for index, _ in top_n]
#             }

#             # Add the current document's result to the all_results list
#             all_results.append(data_structure)
#             # if doc_index % 500 == 0:
#         if (author1 + author2) % 5 == 0:
#             save_results_to_json(all_results, './result/bm25_results.json', "bm25_results (train)" + "_author1 (" + str(author1) + ") author2 (" + str(author2) + ")")

#     # The rest of your code for saving results...
#     return all_results

# def main():
#     # Load and preprocess data
#     file_path = 'EnronEmails_pii/train.jsonl'  # Update with your actual path
#     data = load_data(file_path)
#     data_preprocessed = {}
#     document_counts = defaultdict(int)

#     for entry in data:
#         author_id = entry['author_id']
#         # Count the number of documents (i.e., items in the 'syms' list) for this author
#         document_counts[author_id] += len(entry['syms'])

#     # Print the number of documents for each author
#     print("Number of documents for each author:")
#     for author_id, count in document_counts.items():
#         print(f"Author {author_id}: {count} documents")

#     for entry in data:
#         author_id = entry['author_id']
#         tokenized_syms = tokenize(entry['syms'])
#         data_preprocessed[author_id] = tokenized_syms

#     # Calculate BM25 scores for all document pairs
#     scores = calculate_bm25_all_pairs(data_preprocessed, 3)

#     # Specify the directory and filename where you want to save the results
#     directory = './result'  # Ensure this is a valid path
#     filename = 'bm25_results.json'
#     full_path = os.path.join(directory, filename)

#     # Create the directory if it does not exist
#     os.makedirs(directory, exist_ok=True)

#     # Writing the results to a JSON file
#     with open(full_path, 'w') as f:
#         json.dump(scores, f, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
#     main()
