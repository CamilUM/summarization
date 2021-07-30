from src.input.pre import fix
from src.nlp.token import sentences, words
from src.input.data import Data

class Raw:
    def __init__(self, head, body, sums, category):
        self.head = head
        self.body = body
        self.sums = sums
        self.category = category

    def fix(self):
        head = fix(self.head)
        body = [fix(s) for s in self.body]
        sums = [fix(t) for t in self.sums]
        category = fix(self.category)[:-1]
        return Raw(head, body, sums, category)

    def fix_mlsum(self):
        body = sentences(self.body)
        return Raw(self.head, body, self.sums, self.category)

    def data(self):
        return Data(self.original(), self.sums, self.category)

    def original(self):
        return self.head + " " + (" ".join(self.body))
