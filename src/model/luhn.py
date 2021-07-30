import numpy as np
from src.nlp.token import flat
from src.stat.metric import tfidf
from src.select.filter import threshold
from src.select.split import split, sample
from src.model.heuristic import h

# Machine Learning
def learn(train, test):
    return predict(test, fit(train))

def fit(documents):
    """
    - documents: [[[str]]].
    - return: {str} with significant words.
    """
    matrix, features = tfidf([flat(d) for d in documents])
    stats  = np.mean(matrix, axis = 0)
    return significants(dict(zip(features, stats)), np.mean(stats))

def predict(documents, model):
    return [summarize(d, model) for d in documents]

# Summarization
def significants(stats, D):
    """
    Note: stats can be frequency, TF-IDF...

    - stats: {str: int|float} where {word: stat}.
    - D: int|float with inferior threshold.
    - return: {str} with significant words.
    """
    return {w for w in stats if D <= stats[w]}

def summarize(document, significants):
    """
    - document: [[str]].
    - significants: {str} with significant words.
    - return: [int] with indices.
    """
    scores = [score(s, significants) for s in document]
    scores = h(scores)
    return threshold(scores, sum(scores)/len(scores))

def score(sentence, significants):
    """
    - sentence: [str].
    - significants: {str} with significant words.
    - return: float.
    """
    cs = clusters(sentence, significants, 5)
    factors = [significance_factor(c, significants) for c in cs]
    return 0 if factors == [] else max(factors)

def clusters(sentence, significants, n):
    """
    Divide the sentence in clusters that have at least one significant word and
    less than n non-significant words.

    - sentence: [str].
    - significants: {str} with significant words.
    - n: int with the maximum non-significant allowed words per cluster.
    - return [[str]].
    """
    return [sentence]

def significance_factor(cluster, significants):
    """
    - cluster: [str] with words.
    - significants: {str} with significant words.
    - return: float with significance factor.
    """
    if cluster == []: return 0
    n = len([w for w in cluster if w in significants])
    d = len(cluster)
    return 0 if d == 0 else n*n/d
