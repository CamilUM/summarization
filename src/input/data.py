from src.nlp.token import document

class Data:
    def __init__(self, original, sums, category):
        self.original = original
        self.sums = sums
        self.category = category

    def tokenize(self):
        original = document(self.original)
        sums = [document(s) for s in self.sums]
        return Data(original, sums, self.category)
