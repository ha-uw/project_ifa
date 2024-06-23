import unittest
import torch
from src.modules import decoders


class TestMLP(unittest.TestCase):
    def setUp(self):
        # Initialize MLP with different configurations for comprehensive testing
        self.mlp_default = decoders.MLP(in_dim=10, hidden_dim=20)
        self.mlp_dropout = decoders.MLP(in_dim=10, hidden_dim=20, dropout_rate=0.5)
        self.mlp_single_layer = decoders.MLP(in_dim=10, hidden_dim=20, num_fc_layers=1)
        self.mlp_four_layers = decoders.MLP(in_dim=10, hidden_dim=20, num_fc_layers=4)

    def test_forward_shape(self):
        # Test to ensure the output shape is correct for different configurations
        input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
        output_default = self.mlp_default(input_tensor)
        output_dropout = self.mlp_dropout(input_tensor)
        output_single_layer = self.mlp_single_layer(input_tensor)
        output_four_layers = self.mlp_four_layers(input_tensor)

        self.assertEqual(output_default.shape, (5, 1))
        self.assertEqual(output_dropout.shape, (5, 1))
        self.assertEqual(output_single_layer.shape, (5, 1))
        self.assertEqual(output_four_layers.shape, (5, 1))

    def test_dropout_effect(self):
        # Test to ensure dropout is being applied by comparing the variance of outputs
        input_tensor = torch.randn(
            100, 10
        )  # Larger batch size to ensure statistical significance
        output_default = self.mlp_default(input_tensor)
        output_dropout = self.mlp_dropout(input_tensor)

        var_default = torch.var(output_default).item()
        var_dropout = torch.var(output_dropout).item()

        self.assertGreater(var_dropout, var_default)

    def test_weight_initialization(self):
        # Test to ensure weights of the last layer in a 4-layer MLP are normally initialized
        last_layer_weights = self.mlp_four_layers.fc_layers[-1].weight.data
        # Mean and std of normal distribution
        mean = torch.mean(last_layer_weights).item()
        std = torch.std(last_layer_weights).item()

        # These checks are somewhat lenient due to the randomness, but they ensure the weights are not uniform
        self.assertTrue(-0.5 < mean < 0.5)
        self.assertTrue(0.5 < std < 1.5)

    def test_num_fc_layers(self):
        # Test to ensure the correct number of layers are created
        self.assertEqual(len(self.mlp_default.fc_layers), 2)
        self.assertEqual(len(self.mlp_single_layer.fc_layers), 1)
        self.assertEqual(len(self.mlp_four_layers.fc_layers), 4)


if __name__ == "__main__":
    unittest.main()
