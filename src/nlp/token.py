from nltk import sent_tokenize
from nltk import word_tokenize
from spacy import load

NLP = load("es_core_news_sm", exclude = ["transformer", "tok2vec", "sentencizer", "senter", "attribute_ruler", "morphologizer", "lemmatizer", "textcat_multilabel", "textcat", "entity_ruler", "entity_linker", "ner", "parser", "tagger"])

def sentences(text):
    return sent_tokenize(text, language = "spanish")

def words(text):
    tokens = NLP(text)
    return [t.text for t in tokens]

def document(text):
    return [words(s) for s in sentences(text)]

def flat(document):
    return [w for s in document for w in s]

def text(document):
    return ' '.join(flat(document))
