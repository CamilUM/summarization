import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from src.nlp.token import flat, text
from rouge_metric import PyRouge
from Levenshtein import distance

def tfidf(documents):
    """
    Calculate TF-IDF matrix where rows are documents and columns are features.

    - documents: [[str]] with flat documents.
    - return: [[float]] with TF-IDF matrix with (documents x features).
    - return: [str] with features.
    """
    if documents == []:
        return [], []

    vectorizer = TfidfVectorizer(tokenizer = lambda x: x, lowercase = False)
    matrix   = vectorizer.fit_transform(documents).toarray()
    features = vectorizer.get_feature_names()

    return matrix, features

def tf(documents):
    """
    Term Frequency of a tokenized sentence.

    - documents: [[str]] with flat documents.
    - return: [[float]] with TF matrix with (documents x features).
    - return: [str] with features.
    """
    if documents == []:
        return [], []

    vectorizer = CountVectorizer(tokenizer = lambda x: x, lowercase = False)
    matrix   = vectorizer.fit_transform(documents).toarray()
    features = vectorizer.get_feature_names()

    return matrix, features

def lda(tf, n, iters = 5, updates = 1):
    """
    Latent Dirichlet Allocation.

    - tf: [[int]] with Term Frequency matrix.
    - n: int with number of topics to extract.
    - iters: int with maximum iterations.
    - updates: int with maximum iterations
    to update document topic distribution in E-step.
    - return: [[float]] with (documents x topics)
    - return: [[float]] with (topics x words)
    """
    model = LatentDirichletAllocation(
        n_components = n,
        max_iter = iters,
        max_doc_update_iter = updates,
        random_state = 0
    )
    DT = model.fit_transform(tf)
    TW = model.components_
    return DT, TW

def kmeans(vectors, k, inits, iters):
    """
    - vectors: [[float]] with list of vectors
    - return: [int] with associated labels.
    - return: [[float]] with centroid of each label.
    """
    km = KMeans(
        n_clusters = k,
        n_init = inits,
        max_iter = iters,
        random_state = 0
    ).fit(vectors)

    return km.labels_.tolist(), km.cluster_centers_.tolist()


def centroid(vectors):
    """
    - vectors: [[float]] with the list of vectors.
    - return: [float] with centroid vector.
    """
    if vectors == []: return []
    if vectors[0] == []: return []
    return np.mean(vectors, axis = 0).tolist()

def rouge_1(hypothesis, references):
    """
    - hypothesis: [[str]] with machine-generated summary.
    - references: [[[[str]]] with human-generated summaries.
    - return: float with ROUGE-1.
    """
    rouge = PyRouge(rouge_n = 1, rouge_l = False)
    return rouge.evaluate_tokenized([hypothesis], [references])["rouge-1"]["f"]

def rouge_2(hypothesis, references):
    rouge = PyRouge(rouge_n = 2, rouge_l = False)
    return rouge.evaluate_tokenized([hypothesis], [references])["rouge-2"]["f"]

def rouge_l(hypothesis, references):
    rouge = PyRouge(rouge_n = (), rouge_l = True)
    return rouge.evaluate_tokenized([hypothesis], [references])["rouge-l"]["f"]

def bleu(hypothesis, references):
    """
    - hypothesis: str with machine-generated summary.
    - references: [str] with human-generated summaries.
    - return: float with BLEU.
    """
    hypothesis = flat(hypothesis)
    references = [flat(r) for r in references]
    return sentence_bleu(references, hypothesis, weights = [1])

def edit_distance(hypothesis, references):
    """
    - hypothesis: [[str]] with machine-generated summary.
    - references: [[[str]]] with human-generated summaries.
    - return: float with edit distance.
    """
    if references == []: return 0
    hypothesis = text(hypothesis)
    references = [text(r) for r in references]
    d = [distance(hypothesis, r) for r in references]
    return sum(d)/len(d)
