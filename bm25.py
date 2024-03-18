import math
from collections import Counter

# sample document
documents = [
    "the sky is blue",
    "the sun is bright",
    "the sun in the sky is bright",
    "we can see the shining sun, the bright sun"
]

# calculate TF and DF
df = Counter()
tf = []
for doc in documents:
    tokens = doc.split()
    tf.append(Counter(tokens))
    df.update(set(tokens))

# BM25
N = len(documents)  # number of documents
avgdl = sum(len(doc.split()) for doc in documents) / N  # average length of documents
k1 = 1.5
b = 0.75

# calculate IDF
def idf(term):
    return math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))

# calculate BM25 scores for a query 
def bm25(doc_index, query):
    doc_tf = tf[doc_index]
    score = 0.0
    dl = sum(doc_tf.values())
    for term in query.split():
        if term in doc_tf:
            term_tf = doc_tf[term]
            score += idf(term) * ((term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * dl / avgdl)))
    return score

# check BM25 score for keyword "sun bright" for each document
# query = "sun bright"
# query = "Sun Bright"
query = "sun"
scores = [bm25(i, query) for i in range(N)]
print(f"Scores of keyword ('{query}'): ", scores)

rank = [[i + 1, scores[i]]for i in range(len(scores))]
sorted_rank = sorted(rank, key=lambda x: x[1], reverse=True)
print("Rank: ", sorted_rank)