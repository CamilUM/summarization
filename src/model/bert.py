import torch
import numpy as np
from src.nlp.token import flat
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from src.stat.metric import centroid, tf
from src.select.filter import k_top
from src.model.sumbasic import desired_length
from src.model.heuristic import h

# Machine Learning
def learn(train, test):
    """
    - documents: [[[str]]].
    - return: [[int]] with indices.
    """
    return predict(test, *fit(train))

def fit(documents):
    """
    - documents: [[[str]]].
    - return: {str: float} with TF-IDF model.
    """
    return tokenizer(), bert()

def predict(documents, token, model):
    # Probabilities
    matrix, features = tf([flat(d) for d in documents])
    ps = np.sum(matrix, axis = 0)
    ps = ps / np.sum(ps)
    ps = dict(zip(features, ps))
    
    return [summarize(d, token, model, desired_length(d, ps)) for d in documents]

# Summarization
def summarize(document, token, model, n):
    if len(document) <= n: range(len(document))

    # Reunite
    sentences = [" ".join(s) for s in document]
    # Embeddings
    se = sentence_embeddings(sentences, token, model)
    # Document centroid
    C = centroid(se)
    scores = [score(C, e) for e in se]
    return k_top(scores, n)

def score(C, e):
    if C == []: return 0
    if e == []: return 0
    return 1-cosine(C, e)

def tokenizer():
    return AutoTokenizer.from_pretrained("./src/model/bert")

def bert():
    return AutoModel.from_pretrained("./src/model/bert/")

def sentence_embeddings(sentences, token, model):
    encoded = token(
        sentences,
        padding = True,
        truncation = True,
        max_length = 128,
        return_tensors = "pt"
    )
    with torch.no_grad():
        output = model(**encoded)
        se = output[0][:,0]
    return se.tolist()
