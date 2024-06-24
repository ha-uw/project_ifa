import unittest
from unittest.mock import patch

from src.methods.deepdta import DeepDTA
from src.methods.graphdta import GraphDTA
from pathlib import Path


class TestIntegration(unittest.TestCase):
    def test_deepdta_runs(self):
        config_file = Path("configs", "deepdta.yaml")
        deepdta = DeepDTA(config_file, fast_dev_run=True)
        try:
            deepdta.train()
        except Exception as e:
            self.fail(f"deepdta function failed with an exception: {e}")

    def test_graphdta_runs(self):
        config_file = Path("configs", "graphdta.yaml")
        drug_encoder = "GCN"
        graphdta = GraphDTA(config_file, drug_encoder=drug_encoder, fast_dev_run=True)
        try:
            graphdta.train()
        except Exception as e:
            self.fail(f"graphdta function failed with an exception: {e}")


if __name__ == "__main__":
    unittest.main()
