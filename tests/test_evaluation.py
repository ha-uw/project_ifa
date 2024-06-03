import unittest
import torch
from src.data.evaluation import concordance_index


class TestConcordanceIndex(unittest.TestCase):
    def test_concordance_index_1(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        result = concordance_index(y, y_pred)
        self.assertEqual(result, 1.0)

    def test_concordance_index_2(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([3.0, 2.0, 1.0])
        result = concordance_index(y, y_pred)
        self.assertEqual(result, 0.0)

    def test_concordance_index_3(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 3.0, 1.0])
        result = concordance_index(y, y_pred)
        self.assertAlmostEqual(result, 0.333, places=3)

    def test_concordance_index_4(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 3.0])
        result = concordance_index(y, y_pred)
        self.assertAlmostEqual(result, 0.667, places=3)


if __name__ == "__main__":
    unittest.main()
