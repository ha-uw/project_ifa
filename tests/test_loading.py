import unittest
from pathlib import Path
from src.data.loading import TDCDataset


class TestTDCDataset(unittest.TestCase):

    def test_initialization_with_valid_parameters(self):
        dataset = TDCDataset(name="davis", split="train", path="data")
        self.assertEqual(dataset.name, "davis")
        self.assertEqual(dataset.path, Path("data", "davis"))

    def test_initialization_with_invalid_split(self):
        with self.assertRaises(ValueError):
            TDCDataset(name="davis", split="invalid_split", path="data")


if __name__ == "__main__":
    unittest.main()
