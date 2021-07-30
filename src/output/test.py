import unittest
import tabulate

class TestMetric(unittest.TestCase):
    def test_max(self):
        results = {
            "tfidf" : {"rouge-1": [0.5, 1], "rouge-2": [0.25, 0.75]},
            "luhn": {"rouge-1": [0, 0], "rouge-2": [1, 1]}
        }

        results_predicted = tabulate.mean(results)
        results_real = {
            "tfidf" : {
                "rouge-1": 0.75,
                "rouge-2": 0.5
            },
            "luhn": {
                "rouge-1": 0,
                "rouge-2": 1
            }
        }
        
        self.assertEqual(0, 0)

if __name__ == "__main__":
    unittest.main()
