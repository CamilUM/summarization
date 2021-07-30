import unittest
import metric
from src.nlp.token import document

class TestMetric(unittest.TestCase):
    def test_tf_idf_features(self):
        """
        Verify that it does not unify lowercase, uppercase, accents.
        Because that will belong to cleaning prerprocess.
        """
        corpus = [
            document("España"),
            document("españa"),
            document("espana"),
            document("espáña")
        ]
        _, features = metric.tfidf(corpus)
        self.assertEqual(set(features), {"España", "españa", "espana", "espáña"})

    def test_tf_idf_tokenization(self):
        corpus = [document("hola a todos"), document("hola mundo")]
        matrix, features = metric.tfidf(corpus)
        size = [len(matrix), len(matrix[0])]

        self.assertEqual(size, [2, 4])
        self.assertEqual(set(features), {"hola", "a", "todos", "mundo"})

    def test_rouge_1(self):
        hypothesis = "Hola, mundo!"
        references = ["Hola, mundo!", "Hello, world!"]
        value = metric.rouge_1(hypothesis, references)
        
        self.assertTrue(0 <= value <= 1)

    def test_rouge_2(self):
        hypothesis = "Hola, mundo!"
        references = ["Hola, mundo!", "Hello, world!"]
        value = metric.rouge_2(hypothesis, references)
        
        self.assertTrue(0 <= value <= 1)

    def test_rouge_l(self):
        hypothesis = "Hola, mundo!"
        references = ["Hola, mundo!", "Hello, world!"]
        value = metric.rouge_l(hypothesis, references)
        
        self.assertTrue(0 <= value <= 1)

    def test_bleu(self):
        hypothesis = "Hola, mundo. Estamos listos."
        references = ["Hola, a todos.", "Hello, mundo. We are ready."]
        value = metric.bleu(hypothesis, references)
        
        self.assertTrue(0 <= value <= 1)

    def test_edit_distance(self):
        hypothesis = "Hola, mundo"
        references = ["Hola, a todos", "Hola"]
        value = metric.edit_distance(hypothesis, references)
        
        self.assertTrue(0 <= value <= 1)

if __name__ == "__main__":
    unittest.main()
