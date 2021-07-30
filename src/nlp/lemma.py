from spacy import load
from spacy.tokens import Doc

class NoTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        return Doc(self.vocab, words = text)

NLP = load("es_core_news_sm", exclude = ["transformer", "sentencizer", "senter", "attribute_ruler", "textcat_multilabel", "textcat", "entity_ruler", "entity_linker", "ner", "parser", "tagger"])
NLP.tokenizer = NoTokenizer(NLP.vocab)

def lemmatize(text):
    tokens = NLP(text)
    return [t.lemma_ for t in tokens]
