import unittest
import torch
from src.modules import decoders


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.in_dim = 10
        self.hidden_dim = 20
        self.out_dim = 30
        self.dropout_rate = 0.1

    def test_initialization_without_decoder_layers(self):
        mlp = decoders.MLP(
            self.in_dim, self.hidden_dim, self.out_dim, self.dropout_rate, False
        )
        self.assertEqual(mlp.fc1.in_features, self.in_dim)
        self.assertEqual(mlp.fc1.out_features, self.hidden_dim)
        self.assertEqual(mlp.fc2.in_features, self.hidden_dim)
        self.assertEqual(mlp.fc2.out_features, self.out_dim)
        self.assertFalse(hasattr(mlp, "fc3"))
        self.assertFalse(hasattr(mlp, "fc4"))

    def test_initialization_with_decoder_layers(self):
        mlp = decoders.MLP(
            self.in_dim, self.hidden_dim, self.out_dim, self.dropout_rate, True
        )
        self.assertEqual(mlp.fc1.in_features, self.in_dim)
        self.assertEqual(mlp.fc1.out_features, self.hidden_dim)
        self.assertEqual(mlp.fc2.in_features, self.hidden_dim)
        self.assertEqual(mlp.fc2.out_features, self.hidden_dim)
        self.assertEqual(mlp.fc3.in_features, self.hidden_dim)
        self.assertEqual(mlp.fc3.out_features, self.out_dim)
        self.assertEqual(mlp.fc4.in_features, self.out_dim)
        self.assertEqual(mlp.fc4.out_features, 1)

    def test_forward_without_decoder_layers(self):
        mlp = decoders.MLP(
            self.in_dim, self.hidden_dim, self.out_dim, self.dropout_rate, False
        )
        input_tensor = torch.randn(1, self.in_dim)
        output = mlp(input_tensor)
        self.assertEqual(output.size(), (1, self.out_dim))

    def test_forward_with_decoder_layers(self):
        mlp = decoders.MLP(
            self.in_dim, self.hidden_dim, self.out_dim, self.dropout_rate, True
        )
        input_tensor = torch.randn(1, self.in_dim)
        output = mlp(input_tensor)
        self.assertEqual(output.size(), (1, 1))


if __name__ == "__main__":
    unittest.main()
