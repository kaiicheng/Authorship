from bm25_pt import BM25

bm25 = BM25()
corpus = [
    "A high weight in tf–idf is reached by a high term frequency",
    "(in the given document) and a low document frequency of the term",
    "in the whole collection of documents; the weights hence tend to filter",
    "out common terms. Since the ratio inside the idf's log function is always",
    "greater than or equal to 1, the value of idf (and tf–idf) is greater than or equal",
    "to 0. As a term appears in more documents, the ratio inside the logarithm approaches",
    "1, bringing the idf and tf–idf closer to 0.",
]
bm25.index(corpus)

queries = ["weights", "ratio logarithm"]
# doc_scores = bm25.score_batch(queries)
# print(doc_scores)

doc_scores = bm25.score(queries[0])
print(doc_scores)
