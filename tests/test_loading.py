import unittest
from pathlib import Path
from src.data.loading import TDCDataset
from pandas import DataFrame


class TestTDCDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TDCDataset(name="davis", path="data", print_stats=False)

    def test_initialization_with_valid_parameters(self):
        self.assertEqual(self.dataset.name, "davis")
        self.assertEqual(self.dataset.path, Path("data", "davis"))

    def test_data_type(self):
        self.assertIsInstance(self.dataset.data, DataFrame)

    def test_dataframe_columns(self):
        expected_columns = ["Drug_ID", "Target", "Target_ID", "Y"]
        for column in expected_columns:
            self.assertIn(column, self.dataset.data.columns)

    def test_harmonize_affinities(self):
        self.dataset = TDCDataset(
            name="davis", path="data", print_stats=False, harmonize_affinities=True
        )


if __name__ == "__main__":
    unittest.main()
