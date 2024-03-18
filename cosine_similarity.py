# pip3 install pydantic==2.0.3
# python -m spacy download en_core_web_sm

import spacy
# import cosine distance metric
from scipy.spatial.distance import cosine

nlp = spacy.load("en_core_web_sm")

# vectorise words "like", "love", "hate"
word_1 = nlp("like").vector
word_2 = nlp("love").vector
word_3 = nlp("hate").vector

# compare semantic similarities
dist_1_2 = cosine(word_1, word_2)
dist_2_3 = cosine(word_2, word_3)
dist_3_1 = cosine(word_3, word_1)

print("similarity between 'like' and 'love': ", dist_1_2)
print("similarity between 'love'and 'hate': ", dist_2_3)
print("similarity between 'hate' and 'like': ", dist_3_1)