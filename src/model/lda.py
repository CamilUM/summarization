import numpy as np
from src.nlp.token import flat
from src.stat.metric import tf, lda
from src.model.word2vec import summarize
from src.model.sumbasic import desired_length
from src.select.extract import extract
from src.stat.metric import rouge_1, edit_distance, bleu
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from src.select.split import sample, split

# Machine Learning
def learn(Xtrain, Ytrain, Otrain, test):
    """
    - documents: [[[str]]].
    - return: [[int]] with indices.
    """
    params = {"n": [8,9,10,11,12,13,14,15,16], "iters": [1,2,3,4]}
    params = {"n": [15], "iters": [1]}
    return predict(test, fit(Xtrain, Ytrain, Otrain, params))

def fit(Xtrain, Ytrain, Otrain, params):
    """
    - documents: [[[str]]].
    - return: {str: float} with probabilities model
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
    matrix, features = tf([flat(d) for d in Xtrain])
    _, TW = lda(matrix, pg["n"], pg["iters"])
    return dict(zip(features, TW.transpose()))

def evaluate(summaries, references):
    results = [rouge_1(h, r) for h,r in zip(summaries, references)]
    return sum(results)/len(results)

def predict(documents, model):
    # Probabilities
    matrix, features = tf([flat(d) for d in documents])
    ps = dict(zip(features, np.sum(matrix, axis = 0)))
    
    return [summarize(d, model, desired_length(d, ps)) for d in documents]
