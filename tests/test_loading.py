import unittest
from src.data.loading import TDCDataset
from src.methods.configs import ConfigLoader
import torch


class TestTDCDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TDCDataset(
            name="DAVIS",
            split="train",
            path="./data",
            mode_drug="cnn",
            mode_target="cnn",
        )

    def test_getitem(self):
        item = self.dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 3)
        drug, target, label = item
        self.assertIsInstance(drug, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
