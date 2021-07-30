import math
import numpy as np
from src.nlp.token import flat
from src.stat.metric import tfidf
from src.model.tfidf import score as tfidfscore
from src.select.filter import threshold
from time import time

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
    - row: {str: float} where {features: TF-IDF}.
    - return: [int] with indices
    """
    W = [tfidfscore(s, row) for s in document]
    w = [[similarity(s1, s2) for s2 in document] for s1 in document]

    # Avoid itself
    for i in range(len(w)):
        for j in range(len(w[i])):
            # Avoid itself
            if i == j:
                w[i][j] = 0

    # Filter below average vertices
    for i in range(len(w)):
        t = sum(w[i])/len(w[i])
        for j in range(len(w[i])):
            if w[i][j] < t:
                w[i][j] = 0

    # TextRank
    scores = rank(W, w)
    return threshold(scores, np.average(scores, weights = scores))

def rank(W, w, d = 1, threshold = 0.01):
    """
    - W: [float] with initial vertex ranks.
    - w: [[float]] with graph, edge weights.
    - d: float in [0, 1] with damping factor.
    - threshold: float with vertex difference between iterations.
    - return: np.array[float] with vertex ranks.
    """
    if d < 0: return W
    if d > 1: return W
    if threshold <= 0: return W
    if len(W) != len(w): return W

    while True:
        W1 = ws(W, w, d)
        if halt(W, W1, threshold):
            return W1
        W = W1

def ws(W, w, d):
    """
    - W: [float] with initial vertex ranks.
    - w: [[float]] with graph, edge weights.
    - d: float and 0 =< d <= 1 with damping factor.
    - return: np.array[float] with vertex ranks.
    """
    W1 = W.copy()
    for i in range(len(W)):
        W1[i] = ws_1(W1, w, d, i)
    return W1

def ws_1(W, w, d, i):
    """
    - W: [float] with initial vertex ranks.
    - w: [[float]] with graph, edge weights.
    - d: float and 0 =< d <= 1 with damping factor.
    - i: int with current vertex.
    - return: float with vertex rank.
    """
    weight = 0

    # Each incoming vertex
    for j in range(len(w)):
        # Avoid non existent edges
        if w[j][i] == 0: continue

        # Each outcoming vertex of the incoming j
        denominator = 0
        for k in range(len(w[i])):
            # Avoid non existent edges
            if w[j][k] == 0: continue

            # Add
            denominator += w[j][k]

        # Add
        if denominator != 0:
            weight += w[j][i] / denominator * W[j]

    return (1 - d) + d * weight

def similarity(sentence1, sentence2):
    """
    - sentence1: [str].
    - sentence2: [str].
    - return: float with TextRank sentence similarity.
    """
    if sentence1 == [] or sentence2 == []: return 0
    n = len(set(sentence1) & set(sentence2))
    d = math.log(len(sentence1)) + math.log(len(sentence2))
    return 0 if d == 0 else n/d

def halt(W, W1, threshold):
    """
    - W: [float] with previous vertex ranks.
    - W1: [float] with current vertex ranks.
    - threshold: float with vertex difference between iterations.
    - return: bool with whether TextRank should halt or not.
    """
    return any([abs(w1 - w) < threshold for w1, w in zip(W1, W)])
