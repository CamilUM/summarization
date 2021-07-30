import numpy as np
from src.nlp.token import flat
from src.stat.metric import tfidf
from src.select.filter import threshold
from src.nlp.token import flat
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
    return [summarize(A(d, model)) for d in documents]

# Summarization
def A(document, row):
    """
    Construct A matrix to be used in SVD. Words that are not in the document have 0.

    - document: [[str]].
    - row: {str: float} where {feature: metric}.
    - return: [[float]] with A matrix for SVD with (words x sentences).
    """
    vocabulary = set(flat(document))
    document = [set(s) for s in document]
    return [[row[f] if f in s and f in row else 0 for s in document] for f in vocabulary]

def summarize(A):
    """
    - A: [[float]] with A matrix for SVD with (words x sentences).
    - return: [int] with indices.
    """
    # SVD
    _, S, V = np.linalg.svd(A, full_matrices = False)
    D = np.diag(S) @ V

    # Weights
    scores = [score(r) for r in D]
    return threshold(scores, sum(scores)/len(scores))

def score(row):
    """
    - row: [float] with weights for that topic in all sentences.
    - return: float.
    """
    return np.sum(row*row) ** 0.5
