import unittest
import torch
from torch_geometric.data import Data
from src.modules.trainers import BaseDTATrainer


class MockEncoder:
    def forward(self, x):
        return x


class MockDecoder:
    def forward(self, x):
        return x


class TestBaseDTATrainer(unittest.TestCase):
    def setUp(self):
        self.drug_encoder = MockEncoder()
        self.target_encoder = MockEncoder()
        self.decoder = MockDecoder()
        self.trainer = BaseDTATrainer(
            self.drug_encoder, self.target_encoder, self.decoder
        )

    def test_initialization(self):
        self.assertEqual(self.trainer.drug_encoder, self.drug_encoder)
        self.assertEqual(self.trainer.target_encoder, self.target_encoder)
        self.assertEqual(self.trainer.decoder, self.decoder)


if __name__ == "__main__":
    unittest.main()
