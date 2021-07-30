import numpy as np
from src.nlp.token import flat
from src.stat.metric import tf
from src.model.sumbasic import desired_length
from src.select.split import sample

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
    - return: ().
    """
    return ()

def predict(documents, model):
    matrix, features = tf([flat(d) for d in documents])
    ps = dict(zip(features, np.sum(matrix, axis = 0)))

    return [summarize(d, desired_length(d, ps)) for d in documents]

# Summarization
def summarize(document, n, seed = 0):
    """
    - document: [[str]].
    - row: {str: float} where {feature: TF-IDF}.
    - return: [int] with indices
    """
    return sample(document, n/len(document), seed)
