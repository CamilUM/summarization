import numpy as np
from src.nlp.token import flat
from src.stat.metric import tfidf
from src.select.filter import threshold, k_top
from src.select.split import split, sample
from src.model.heuristic import h

# Machine Learning
def learn(train, test):
    """
    - documents: [[[str]]].
    - return: [[int]] with indices.
    """
    return predict(test, fit(train))

def fit(documents):
    """
    - documents: [[[str]]].
    - return: {str: float} with TF-IDF model.
    """
    matrix, features = tfidf([flat(d) for d in documents])
    return dict(zip(features, np.sum(matrix, axis = 0)))

def predict(documents, model):
    return [summarize(d, model) for d in documents]

# Summarization
def summarize(document, row):
    """
    - document: [[str]].
    - row: {str: float} where {feature: TF-IDF}.
    - return: [int] with indices
    """
    scores = [score(s, row) for s in document]
    scores = h(scores)
    return threshold(scores, sum(scores)/len(scores))

def score(sentence, row):
    """
    Score one sentence. Calculate the mean of TF-IDF values of its words.

    - sentence: [str].
    - row: {str: float} where {feature: TF-IDF}.
    - return: float.
    """
    values = [row[w] for w in sentence if w in row]
    return 0 if values == [] else sum(values)/len(values)
