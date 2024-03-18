from sklearn.feature_extraction.text import TfidfVectorizer

# sample document
documents = [
    "the sky is blue",
    "the sun is bright",
    "the sun in the sky is bright",
    "we can see the shining sun, the bright sun"
   ]

# documents = ['this is the first document',
#         'this is the second second document',
#         'and the third one',
#         'is this the first document'
#     ]

# initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# use vectorizer to learn vocabulabries and idf vector
tfidf_matrix = vectorizer.fit_transform(documents)
print("tfidf_matrix: ", tfidf_matrix)

# tf-idf scores
feature_names = vectorizer.get_feature_names_out()
print("feature_names: ", feature_names)

# print out tf-idf vector
tf_idf_score = []
for doc_id, doc in enumerate(tfidf_matrix):
    print(f"Document {doc_id}:")
    df = tfidf_matrix[doc_id].T.todense()
    # print("df: ", df)
    df_list = df.tolist()
    # print("df_list: ", df_list)
    tfidf_scores = [(feature_names[i], df_list[i][0]) for i in range(len(feature_names))]
    tfidf_scores_sorted = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    
    for word, score in tfidf_scores_sorted:
        tf_idf_score.append([word, score])
        print(f" - {word}: {score}")

# for i in range(len(tf_idf_score)):
    # print("tf_idf_score[i]: ", tf_idf_score[i])
print("\n")