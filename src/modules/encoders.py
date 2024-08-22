"""
encoders.py

This module contains the implementation of various encoders used in the project. 
Encoders are components of a machine learning model that transform the high-dimensional 
input data into a lower-dimensional representation suitable for processing by the model.

Sources:
https://github.com/thinng/GraphDTA
"""

# Imports ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


# Encoders --------------------------------------------------------------
class CNN(nn.Module):
    """
    CNN Module

    This module implements a Convolutional Neural Network (CNN) for text classification.

    Args:
        num_embeddings (int): The number of unique embeddings in the input.
        embedding_dim (int): The dimensionality of the embedding vectors.
        sequence_length (int): The length of the input sequences.
        num_filters (int): The number of filters in each convolutional layer.
        kernel_size (list or int): The size of the convolutional kernels. If a list, it should have a length equal to num_conv_layers.
        num_conv_layers (int): The number of convolutional layers.

    Attributes:
        embedding (nn.Embedding): The embedding layer.
        conv_layers (nn.ModuleList): The list of convolutional layers.
        global_max_pool (nn.AdaptiveMaxPool1d): The global max pooling layer.

    Methods:
        forward(x): Performs forward pass through the CNN.

    Raises:
        ValueError: If the number of kernel sizes is not equal to the number of convolutional layers.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        sequence_length,
        num_filters,
        kernel_size: list | int,
        num_conv_layers: int,
    ):
        super(CNN, self).__init__()
        if isinstance(kernel_size, list):
            if len(kernel_size) != num_conv_layers:
                raise ValueError(
                    "Number of Kernel sizes needs to be equal to the number of layers."
                )
        else:
            kernel_size = [kernel_size for _ in range(num_conv_layers)]
        self.num_conv_layers = num_conv_layers

        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv_layers = nn.ModuleList()

        # First Layer
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=sequence_length,
                out_channels=num_filters,
                kernel_size=kernel_size[0],
            )
        )

        for i in range(1, num_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=num_filters * i,
                    out_channels=num_filters * (i + 1),
                    kernel_size=kernel_size[i],
                )
            )

        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x = self.embedding(x)

        for conv in self.conv_layers:
            x = F.relu(conv(x))

        x = self.global_max_pool(x)
        x = x.squeeze(2)
        return x


class WideCNN(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_embeddings,
        embedding_dim=128,
        num_filters=32,
        kernel_size=2,
    ):
        super(WideCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv1 = nn.Conv1d(
            in_channels=sequence_length,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )

        # self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x) -> tuple:
        x = self.embedding(x)
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = x.squeeze(2)

        return x


# Graph -----------------------------------------------------------------
class GAT(nn.Module):
    """
    Graph Attention Network (GAT) module.
    """

    def __init__(
        self,
        input_dim=78,
        num_heads=10,
        output_dim=128,
        dropout_rate=0.2,
    ):
        super(GAT, self).__init__()

        # Graph layers
        self.conv1 = GATConv(
            input_dim, input_dim, heads=num_heads, dropout=dropout_rate
        )
        self.conv2 = GATConv(input_dim * num_heads, output_dim, dropout=dropout_rate)
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph data processing
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc(x)
        x = F.relu(x)

        return x


class GAT_GCN(nn.Module):
    """
    GAT_GCN is a PyTorch module that implements a Graph Attention Network (GAT)
    combined with a Graph Convolutional Network (GCN) for graph data processing.
    """

    def __init__(
        self,
        input_dim=78,
        num_heads=32,
        output_dim=128,
        dropout_rate=0.2,
    ):
        super(GAT_GCN, self).__init__()

        self.conv1 = GATConv(
            input_dim, input_dim, heads=num_heads, dropout=dropout_rate
        )
        self.conv2 = GCNConv(input_dim * num_heads, input_dim * num_heads)
        self.fc = nn.Linear(input_dim * num_heads * 2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph data processing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) module.
    """

    def __init__(
        self,
        input_dim=78,
        output_dim=128,
        dropout_rate=0.2,
    ):
        super(GCN, self).__init__()

        # GCN layers for graph data (representing molecules)
        self.conv1 = GCNConv(input_dim, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim * 2)
        self.conv3 = GCNConv(input_dim * 2, input_dim * 4)

        # nn.Linear layers for processing graph data after GCN convolutions
        self.fc1 = nn.Linear(input_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph data processing with GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)

        # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)

        return x


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) Convolutional Neural Network.
    """

    def __init__(
        self,
        input_dim=78,
        num_filters=32,
        output_dim=128,
        dropout_rate=0.2,
    ):
        super().__init__()

        # GIN layers
        self.conv1 = GINConv(
            nn=nn.Sequential(
                nn.Linear(input_dim, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, num_filters),
            )
        )
        self.bn1 = nn.BatchNorm1d(num_filters)

        self.conv2 = GINConv(
            nn=nn.Sequential(
                nn.Linear(num_filters, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, num_filters),
            )
        )
        self.bn2 = nn.BatchNorm1d(num_filters)

        self.conv3 = GINConv(
            nn=nn.Sequential(
                nn.Linear(num_filters, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, num_filters),
            )
        )
        self.bn3 = nn.BatchNorm1d(num_filters)

        self.conv4 = GINConv(
            nn=nn.Sequential(
                nn.Linear(num_filters, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, num_filters),
            )
        )
        self.bn4 = nn.BatchNorm1d(num_filters)

        self.conv5 = GINConv(
            nn=nn.Sequential(
                nn.Linear(num_filters, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, num_filters),
            )
        )
        self.bn5 = nn.BatchNorm1d(num_filters)

        self.fc1 = nn.Linear(num_filters, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GIN layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x
