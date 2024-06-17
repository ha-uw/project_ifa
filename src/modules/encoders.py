"""
encoders.py

This module contains the implementation of various encoders used in the project. 
Encoders are components of a machine learning model that transform the high-dimensional 
input data into a lower-dimensional representation suitable for processing by the model.

Author: Raul Oliveira
Date: 01/06/2024
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
    Convolutional Neural Network (CNN) encoder.
    This encoder uses an embedding layer followed by three convolutional layers and a global max pooling layer.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        sequence_length,
        num_filters,
        filter_length,
    ):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv1 = nn.Conv1d(
            in_channels=sequence_length,
            out_channels=num_filters,
            kernel_size=filter_length[0],
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=filter_length[1],
        )
        self.conv3 = nn.Conv1d(
            in_channels=num_filters * 2,
            out_channels=num_filters * 3,
            kernel_size=filter_length[2],
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x)
        x = x.squeeze(2)
        return x


# class CNN(nn.Module):
#     """
#     Convolutional Neural Network (CNN) encoder with dynamic convolutional layers.
#     This encoder uses an embedding layer followed by a dynamic number of convolutional layers and a global max pooling layer.
#     """

#     def __init__(
#         self,
#         num_embeddings,
#         embedding_dim,
#         sequence_length,
#         num_filters,
#         filter_lengths,
#         num_conv_layers,
#     ):
#         super(CNN, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)

#         # Initialize convolutional layers dynamically based on num_conv_layers
#         self.conv_layers = nn.ModuleList()
#         for i in range(num_conv_layers):
#             in_channels = sequence_length if i == 0 else num_filters * 2 ** (i - 1)
#             out_channels = num_filters * 2**i
#             kernel_size = filter_lengths[i]
#             conv_layer = nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#             )
#             self.conv_layers.append(conv_layer)

#         self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

#     def forward(self, x: torch.Tensor | list[torch.Tensor]):
#         if isinstance(x, list):
#             # If x is a list of tensors, process each tensor individually and concatenate the results
#             outputs = []
#             for xn in x:
#                 xn = self.embedding(xn)
#                 for conv in self.conv_layers:
#                     xn = self.global_max_pool(F.relu(conv(xn)))
#                     # Dynamically calculate the number of features
#                     num_features = xn.size(1) * xn.size(2)
#                     xn = xn.view(-1, num_features)
#                 outputs.append(xn)

#             # Concatenate along the first dimension
#             x = torch.cat(outputs, dim=1)

#         else:
#             # Process a single tensor
#             x = self.embedding(x)
#             for conv in self.conv_layers:
#                 x = F.relu(conv(x))
#             x = self.global_max_pool(x)
#             x = x.squeeze(2)

#         return x


class WideCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ###protein
        self.pconv1 = nn.Conv1d(
            in_channels=6729, out_channels=16, kernel_size=2, stride=1, padding=1
        )
        self.pconv2 = nn.Conv1d(16, 32, 2, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(2, 2)
        ###ligands
        self.lconv1 = nn.Conv1d(
            in_channels=10, out_channels=16, kernel_size=2, stride=1, padding=1
        )
        self.lconv2 = nn.Conv1d(16, 32, 2, stride=1, padding=1)
        #####motif
        self.mconv1 = nn.Conv1d(
            in_channels=1076, out_channels=16, kernel_size=2, stride=1, padding=1
        )
        self.mconv2 = nn.Conv1d(16, 32, 2, stride=1, padding=1)

        ###
        self.dropout = nn.Dropout(0.3)
        self.FC1 = nn.Linear(5120, 512)
        self.FC2 = nn.Linear(512, 10)
        self.FC3 = nn.Linear(10, 1)

    def forward(self, x1, x2, x3):
        x1 = self.maxpool(F.relu(self.pconv1(x1)))
        x1 = self.maxpool(F.relu(self.pconv2(x1)))

        x2 = self.maxpool(F.relu(self.lconv1(x2)))
        x2 = self.maxpool(F.relu(self.lconv2(x2)))

        x3 = self.maxpool(F.relu(self.mconv1(x3)))
        x3 = self.maxpool(F.relu(self.mconv2(x3)))

        x1 = x1.view(-1, 149 * 32)
        x2 = x2.view(-1, 3 * 32)
        x3 = x3.view(-1, 8 * 32)

        x = torch.cat([x1, x2, x3], 1)
        x = F.relu(self.FC1(x))
        x = self.dropout(x)
        x = F.relu(self.FC2(x))
        x = self.dropout(x)
        x = self.FC3(x)
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
