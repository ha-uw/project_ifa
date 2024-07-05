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

    def test_concordance_index_large_dataset(self):
        y = torch.rand(1000)
        y_pred = torch.rand(1000)
        result = concordance_index(y, y_pred)
        self.assertTrue(0 <= result <= 1)


class TestConcordanceIndexDetailed(unittest.TestCase):
    def test_concordance_index_perfect_match(self):
        # Test case where y and y_pred are perfectly matched
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = concordance_index(y, y_pred)
        self.assertEqual(result, 1.0, "Perfect match should return CI of 1.0")

    def test_concordance_index_inverse_match(self):
        # Test case where y and y_pred are inversely matched
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([4.0, 3.0, 2.0, 1.0])
        result = concordance_index(y, y_pred)
        self.assertEqual(result, 0.0, "Inverse match should return CI of 0.0")

    def test_concordance_index_partial_match(self):
        # Test case where y and y_pred are partially matched
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([2.0, 1.0, 4.0, 3.0])
        result = concordance_index(y, y_pred)
        self.assertTrue(
            0 < result < 1, "Partial match should return CI between 0 and 1"
        )


if __name__ == "__main__":
    unittest.main()
