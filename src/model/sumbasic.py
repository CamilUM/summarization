import numpy as np
from math import ceil
from src.nlp.token import flat
from src.stat.metric import tf
from scipy.stats import entropy
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
    - return: {str: float} with probabilities model.
    """
    matrix, features = tf([flat(d) for d in documents])
    ps = np.sum(matrix, axis = 0)
    ps = ps / np.sum(ps)
    return dict(zip(features, ps))

def predict(documents, model):
    matrix, features = tf([flat(d) for d in documents])
    ps = dict(zip(features, np.sum(matrix, axis = 0)))

    return [summarize(d, model, desired_length(d, ps)) for d in documents]

# Summarization
def summarize(document, probabilities, n):
    """
    - document: [[str]].
    - probabilities: {str: float} where {feature: probability}.
    - n: int with desired number of sentences.
    - return: [int] with indices.
    """
    if len(document) <= n: return range(len(document))

    # Avoid mutation
    probabilities = probabilities.copy()
    document = document.copy()

    # Sumbasic filtering
    bests = []
    for _ in range(n):
        scores = [score(s, probabilities) for s in document]
        scores = h(scores)
        best   = np.argmax(scores)
        probabilities = update(probabilities, document[best])
        bests.append(best)
        document.pop(best)

    return sorted(bests)

def score(sentence, probabilities):
    """
    Score one sentence. Calculate the mean of probabilities of its words.

    - sentence: [str].
    - probabilities: {str: float} where {feature: probability}.
    - return: float.
    """
    values = [probabilities[w] for w in sentence if w in probabilities]
    return 0 if values == [] else sum(values)/len(values)

def update(probabilities, sentence):
    new = probabilities
    for w in sentence:
        if w in new:
            new[w] = probabilities[w] ** 2

    return new

def desired_length(document, probabilities):
    plano    = flat(document)
    entropia = entropy([probabilities[w] for w in plano if w in probabilities])
    words    = np.exp(entropia)
    desired  = len(document) * words / len(plano)
    return ceil(desired)
