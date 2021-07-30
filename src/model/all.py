from time import time
from functools import partial
from src.input.pre import clean
from src.nlp.token import document
from src.select.extract import extract
from src.select.split import sample, split
from src.stat.grid import grid
import src.model.random
import src.model.tfidf
import src.model.luhn
import src.model.sumbasic
import src.model.lsa
import src.model.lda
import src.model.textrank
import src.model.word2vec
import src.model.bert

def learn(originals, references):
    # Preprocess
    #documents = [[clean(s) for s in o] for o in originals]
    documents = [[s for s in o] for o in originals]

    # Split
    samples = sample(originals)
    Xtrain, Xtest = split(documents, samples)
    Ytrain, Ytest = split(references, samples)
    Otrain, Otest = split(originals, samples)

    # Models
    summarizers = {
        "Random":   partial(src.model.random.learn, Xtrain, Xtest),
        "TF-IDF":   partial(src.model.tfidf.learn, Xtrain, Xtest),
        "Luhn":     partial(src.model.luhn.learn, Xtrain, Xtest),
        #"SumBasic": partial(src.model.sumbasic.learn, Xtrain, Xtest),
        #"LSA":      partial(src.model.lsa.learn, Xtrain, Xtest),
        #"LDA":      partial(src.model.lda.learn, Xtrain, Ytrain, Otrain, Xtest),
        #"TextRank": partial(src.model.textrank.learn, Xtrain, Xtest),
        #"Word2Vec": partial(src.model.word2vec.learn, Xtrain, Ytrain, Otrain, Xtest),
        #"BERT":     partial(src.model.bert.learn, Xtrain, Xtest)
    }

    # Metrics
    evaluators = {
        "ROUGE-1": src.stat.metric.rouge_1,
        "BLEU":    src.stat.metric.bleu,
        "Edit":    src.stat.metric.edit_distance
    }

    # Execute
    summaries = {}
    times     = {}
    for method, summarizer in summarizers.items():
        t0 = time()
        indices = summarizer()
        summaries[method] = [extract(d, i) for d, i in zip(Otest, indices)]
        times[method] = time() - t0

    # Comparing with human summaries
    results = {}
    for method, hypotheses in summaries.items():
        metrics = {}
        for evaluation, evaluator in evaluators.items():
            measures = []
            for h, r in zip(hypotheses, Ytest):
                measures.append(evaluator(h, r))
            metrics[evaluation] = measures
        results[method] = metrics

    # Comparing with original text
    for method, hypotheses in summaries.items():
        measures = []
        for h, o in zip(hypotheses, Otest):
            measures.append(len(o))
            #measures.append(len(h) / len(o))
        results[method]["Proportion"] = measures

    return results, times

def train(originals):
    """
    Train all data with all models.

    - originals: [[[str]]] with uncleaned tokenized documents.
    - return: {str: [[[str]]]} where {method: [[[tokenized summary]]]}.
    - return: {str: number} where {method: time}.
    """
    # Preprocess
    documents = [[clean(s) for s in o] for o in originals]

    # Models
    summarizers = {
        "TF-IDF":   partial(src.model.tfidf.learn, documents),
        "Luhn":     partial(src.model.luhn.learn, documents),
        #"SumBasic": partial(src.model.sumbasic.train, documents),
        #"LSA":      partial(src.model.lsa.train, documents),
        #"LDA":      partial(src.model.lda.train, documents),
        #"TextRank": partial(src.model.textrank.train, documents),
        #"Word2Vec": partial(src.model.word2vec.train, documents),
        #"BERT":     partial(src.model.bert.train, documents)
    }

    # Gridding - Experimental - Delete
    #summarizers = grid("Word2Vec", src.model.word2vec.train, documents, {"dimensions": [320, 344, 368, 392, 416, 440, 464, 488], "window": [4, 6, 8], "epochs": [32]})
    #summarizers = grid("LDA", src.model.lda.train, documents, {"n": [16, 24, 32, 40, 48, 56], "iters": [1, 2, 4, 6, 8]})
    
    # Execute
    summaries = {}
    times     = {}
    for method, summarizer in summarizers.items():
        t0 = time()
        indices = summarizer()
        summaries[method] = [extract(d, i) for d, i in zip(originals, indices)]
        times[method] = time() - t0

    return summaries, times

def evaluate(summaries, references, originals):
    """
    Evaluate all results with all metrics.

    - summaries: {str: [[[str]]]} where {method: [[[tokenized summary]]]}.
    - references: [[str]] with uncleaned and untokenied documents.
    - originals: [str] with uncleaned and untokenied documents.
    - return: {str: {str: [float]}} with {method: {metric: [measure]}}.
    """
    # Metrics
    evaluators = {
        "ROUGE-1": src.stat.metric.rouge_1,
        "BLEU":    src.stat.metric.bleu,
        "Edit":    src.stat.metric.edit_distance
    }

    # Comparing with human summaries
    results = {}
    for method, hypotheses in summaries.items():
        metrics = {}
        for evaluation, evaluator in evaluators.items():
            measures = []
            for h, r in zip(hypotheses, references):
                measures.append(evaluator(h, r))
            metrics[evaluation] = measures
        results[method] = metrics

    # Comparing with original text
    for method, hypotheses in summaries.items():
        measures = []
        for h, o in zip(hypotheses, originals):
            measures.append(len(h) / len(o))
        results[method]["Proportion"] = measures

    return results
