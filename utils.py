import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
    

# Word utils

def preprocess(review: str) -> list:
    # 1. Clean text
    review = clean_review(review)
    #print(review)
    # 2. Split into individual words
    tokens = word_tokenize(review)
    #print("tok", tokens)
    
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return tokens

def vectorize_data(data, w2v) -> list:
    #keys = list(w2v.get_vector.keys())
    
    vectorization = dict()
    for sentence in data:
        for word in sentence:
            if word not in vectorization:
                if w2v.has_index_for(word):
                    vectorized_word = w2v.get_vector(word)
                    vectorization[word] = vectorized_word
                else:
                    vectorization[word] = (np.random.random(300)-0.5).tolist()

    #encode = lambda review: list(map(w2v.get_vector, filter(filter_unknown, review)))
    
    vectorized = []
    for d in data:
        vectorized.append(list(map(vectorization.get, d)))


    print('Vectorize sentences... (done)')
    return vectorized

REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z1-9\s]')

def clean_review(raw_review: str) -> str:

    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    # 2. Remove non-letters
    letters_only = REPLACE_WITH_SPACE.sub(" ", review_text)
    # 3. Convert to lower case
    lowercase_letters = letters_only.lower()
    return lowercase_letters


def lemmatize(tokens: list) -> list:
    #print("lemmatiziing:")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")) # These are meaningnful????
    # 1. Lemmatize
    #print("Massive list of stop words:", stop_words)
    tokens = list(map(lemmatizer.lemmatize, tokens))
    #print("Tokens:", tokens)
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    #print("lemmatized tokens:", lemmatized_tokens)
    # 2. Remove stop words
    #meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))

    return lemmatized_tokens