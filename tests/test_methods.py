import unittest
from pathlib import Path
from src.methods.deepdta import DeepDTA
from src.methods.graphdta import GraphDTA
from src.methods.widedta import WideDTA

from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")


class TestMethods(unittest.TestCase):
    def test_deepdta(self):
        config_file = Path("configs", "DeepDTA", "test.yaml")
        deepdta = DeepDTA(config_file, fast_dev_run=True)

        try:
            deepdta.run_k_fold_validation(2)
        except Exception as e:
            self.fail(f"deepdta function failed with an exception: {e}")

    def test_graphdta(self):
        config_file = Path("configs", "GraphDTA", "test.yaml")
        drug_encoder = "GCN"
        graphdta = GraphDTA(config_file, drug_encoder=drug_encoder, fast_dev_run=True)

        try:
            graphdta.run_k_fold_validation(2)
        except Exception as e:
            self.fail(f"graphdta function failed with an exception: {e}")

    def test_widedta(self):
        config_file = Path("configs", "WideDTA", "test.yaml")
        widedta = WideDTA(config_file, fast_dev_run=True)

        try:
            widedta.run_k_fold_validation(2)
        except Exception as e:
            self.fail(f"widedta function failed with an exception: {e}")


if __name__ == "__main__":
    unittest.main()
