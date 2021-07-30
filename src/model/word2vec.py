import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from src.nlp.token import flat
from src.stat.metric import tf, centroid
from src.select.filter import k_top
from src.model.sumbasic import desired_length
from src.model.heuristic import h
from sklearn.model_selection import ParameterGrid
from src.select.extract import extract
from sklearn.model_selection import KFold
from src.stat.metric import rouge_1

# Machine Learning
def learn(Xtrain, Ytrain, Otrain, test):
    """
    - documents: [[[str]]].
    - return: [[int]] with indices.
    """
    params = {"dimensions": [100,120,140,160,180], "window": [3,5,7], "epochs": [2]}
    params = {"dimensions": [140], "window": [5], "epochs": [2]}
    return predict(test, fit(Xtrain, Ytrain, Otrain, params))

def fit(Xtrain, Ytrain, Otrain, params):
    """
    - documents: [[[str]]].
    - return: {str: [float]} where {word: vector} with |vector| = dimensions.
    """
    best = (None, {})
    for pg in ParameterGrid(params):
        # Crosvalidate
        result = crossvalidate(Xtrain, Ytrain, Otrain, pg)

        # Gets better?
        if best[0] is None:
            best = (result, pg)
        elif best[0] < result:
            best = (result, pg)

    # Print best model
    print(best[1])

    return {} if best[0] == None else subfit(Xtrain, best[1])

def crossvalidate(Xtrain, Ytrain, Otrain, pg):
    # Split train into train and validation
    results = []
    for tr, vl in KFold().split(Xtrain):
        # Splits
        Xsubtrain = [Xtrain[i] for i in tr]
        Xvalidate = [Xtrain[i] for i in vl]
        Yvalidate = [Ytrain[i] for i in vl]
        Ovalidate = [Otrain[i] for i in vl]
        
        # Fit
        model = subfit(Xsubtrain, pg)

        # Predict
        indices = predict(Xvalidate, model)
        summaries = [extract(d, i) for d,i in zip(Ovalidate, indices)]
        
        # Evaluation
        result = evaluate(summaries, Yvalidate)
        results.append(result)
    return 0 if results == [] else sum(results)/len(results)

def subfit(Xtrain, pg):
    return word2vec(Xtrain, pg["dimensions"], pg["window"], pg["epochs"])

def evaluate(summaries, references):
    results = [rouge_1(h, r) for h,r in zip(summaries, references)]
    return sum(results)/len(results)

def predict(documents, model):
    # Probabilities
    matrix, features = tf([flat(d) for d in documents])
    ps = dict(zip(features, np.sum(matrix, axis = 0)))

    # Summarize
    return [summarize(d, model, desired_length(d, ps)) for d in documents]

# Summarization
def summarize(document, we, n):
    """
    - documents: [[str]].
    - we: {str: [float]} where {word: vector}.
    - row: {str: [float]} where {feature: [TF-IDF]}.
    - return: [int] with indices.
    """
    C = centroid([we[w] for w in flat(document) if w in we])
    scores = [score(s, C, we) for s in document]
    return k_top(scores, n)

def score(sentence, C, we):
    """
    - sentence: [str].
    - C: [float] document centroid.
    - we: {str: [float]} where {word: vector}.
    """
    if C == []: return 0
    if not any(C): return 0
    c = centroid([we[w] for w in sentence if w in we])
    if c == []: return 0
    if not any(C): return 0
    return 1 - cosine(C, c)

def word2vec(documents, dimensions, window, epochs):
    """
    - documents: [[str]].
    - dimensions: int with vectors dimensions for word2vec.
    - return: {str: [float]} where {word: vector} with |vector| = dimensions.
    """
    sentences = [s for d in documents for s in d]
    model = Word2Vec(
        sentences = sentences,
        vector_size = dimensions,
        window = window,
        epochs = epochs,
        seed = 0,
        min_count = 1,
        sg = 0,
        hs = 1,
        workers = 1
    )

    return {k: model.wv[k] for k in model.wv.key_to_index}
