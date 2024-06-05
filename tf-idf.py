import math
from collections import Counter

# sample document
# documents = [
#     "the sky is blue",
#     "the sun is bright",
#     "the sun in the sky is bright",
#     "we can see the shining sun, the bright sun"
# ]

documents = ['this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document'
    ]

# calculate term frequency
def compute_tf(document):
    tf_document = Counter(document.split())
    num_words = len(document.split())
    for word in tf_document:
        tf_document[word] /= num_words
    return tf_document

# calculate idf (inverse document frequency)
def compute_idf(documents):
    N = len(documents)
    idf_dict = {}
    all_documents = []
    for document in documents:
        all_documents.extend(set(document.split()))
    all_documents_count = Counter(all_documents)
    for word, count in all_documents_count.items():
        idf_dict[word] = math.log(N / float(count))
    return idf_dict

# calculate tf-idf 
def compute_tf_idf(documents):
    tf_idf = []
    idf = compute_idf(documents)
    for document in documents:
        tf = compute_tf(document)
        tf_idf_document = {}
        for word, tf_value in tf.items():
            tf_idf_document[word] = tf_value * idf[word]
        tf_idf.append(tf_idf_document)
    return tf_idf

# call tf-idf function
tf_idf_scores = compute_tf_idf(documents)

for i in range(len(tf_idf_scores)):
    print(f"Document {i}:")
    print(tf_idf_scores[i])
