import unittest
import torch
from torch_geometric.data import Data
from src.modules import encoders


class TestEncoders(unittest.TestCase):

    def setUp(self):
        self.data = Data(
            x=torch.rand((10, 78)),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            batch=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )

    def test_CNN(self):
        model = encoders.CNN(100, 10, 10, 32, [3, 4, 5], num_conv_layers=3)
        output = model.forward(torch.randint(0, 100, (10, 10)))
        self.assertEqual(output.size(), (10, 96))

        # Test with different input size
        output = model.forward(torch.randint(0, 100, (20, 10)))
        self.assertEqual(output.size(), (20, 96))

    def test_GAT(self):
        model = encoders.GAT()
        output = model.forward(self.data)
        self.assertEqual(output.size(), (1, 128))

        # Test with different input size
        data = Data(
            x=torch.rand((20, 78)),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            batch=torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        )
        output = model.forward(data)
        self.assertEqual(output.size(), (1, 128))

    def test_GAT_GCN(self):
        model = encoders.GAT_GCN()
        output = model.forward(self.data)
        self.assertEqual(output.size(), (1, 128))

        # Test with different input size
        data = Data(
            x=torch.rand((20, 78)),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            batch=torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        )
        output = model.forward(data)
        self.assertEqual(output.size(), (1, 128))

    def test_GCN(self):
        model = encoders.GCN()
        output = model.forward(self.data)
        self.assertEqual(output.size(), (1, 128))

        # Test with different input size
        data = Data(
            x=torch.rand((20, 78)),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            batch=torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        )
        output = model.forward(data)
        self.assertEqual(output.size(), (1, 128))

    def test_GIN(self):
        model = encoders.GIN()
        output = model.forward(self.data)
        self.assertEqual(output.size(), (1, 128))

        # Test with different input size
        data = Data(
            x=torch.rand((20, 78)),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            batch=torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        )
        output = model.forward(data)
        self.assertEqual(output.size(), (1, 128))


if __name__ == "__main__":
    unittest.main()
