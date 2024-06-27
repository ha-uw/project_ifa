"""
decoders.py

This module contains the implementation of various decoders used in the project. 
Decoders are components of a machine learning model that transform the output of 
an encoder back into the original high-dimensional space or into a different useful format.

Author: Raul Oliveira
Date: 01/06/2024
"""

# Imports ---------------------------------------------------------------
import torch
from torch import nn
from torch.nn import functional as F


# Decoders --------------------------------------------------------------
class MLP(nn.Module):
    """
    A generalized MLP (Multi-Layer Perceptron) model that dynamically adjusts the number of fully connected (fc) layers based on the input parameter.

    Args:
        in_dim (int): The dimension of the input feature.
        hidden_dim (int): The dimension of hidden layers.
        num_fc_layers (int): The number of fully connected layers. Defaults to 2 for a simpler MLP structure.
        dropout_rate (float): The dropout rate during training.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout_rate=0.1,
        num_fc_layers=3,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.num_fc_layers = num_fc_layers
        self.dropout = nn.Dropout(dropout_rate)

        if num_fc_layers < 3:
            raise ValueError("Minimum of 3 FC layers.")

        # First layer
        self.fc_layers.append(nn.Linear(in_dim, hidden_dim))

        # Intermediate layers
        for _ in range(1, num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_layers.append(nn.Linear(hidden_dim, out_dim))

        # Output/ Prediction
        self.fc_layers.append(nn.Linear(out_dim, 1))

        if num_fc_layers == 4:
            torch.nn.init.normal_(self.fc_layers[-1].weight)

    def forward(self, x):
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < self.num_fc_layers - 1:
                # Apply ReLU and dropout to all but the last 2 layers
                x = F.relu(x)
                x = self.dropout(x)

        return x


# class MLP(nn.Module):
#     """
#     A generalized MLP (Multi-Layer Perceptron) model that dynamically adjusts the number of fully connected (fc) layers based on the input parameter.

#     Args:
#         in_dim (int): The dimension of the input feature.
#         hidden_dim (int): The dimension of hidden layers.
#         num_fc_layers (int): The number of fully connected layers. Defaults to 2 for a simpler MLP structure.
#         dropout_rate (float): The dropout rate during training.
#     """

#     def __init__(
#         self,
#         in_dim,
#         hidden_dim,
#         dropout_rate=0.1,
#         num_fc_layers=2,
#     ):
#         super(MLP, self).__init__()
#         self.fc_layers = nn.ModuleList()
#         self.num_fc_layers = num_fc_layers
#         self.dropout = nn.Dropout(dropout_rate)

#         # First layer
#         self.fc_layers.append(nn.Linear(in_dim, hidden_dim))

#         # Intermediate layers
#         for _ in range(1, num_fc_layers - 1):
#             self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

#         # Output layer hardcoded to 1
#         if num_fc_layers > 1:
#             self.fc_layers.append(nn.Linear(hidden_dim, 1))
#         else:
#             self.fc_layers[-1] = nn.Linear(in_dim, 1)

#         if num_fc_layers == 4:
#             torch.nn.init.normal_(self.fc_layers[-1].weight)

#     def forward(self, x):
#         for i, fc in enumerate(self.fc_layers):
#             x = fc(x)
#             if i < self.num_fc_layers - 1:
#                 # Apply ReLU and dropout to all but the last layer
#                 x = F.relu(x)
#                 x = self.dropout(x)

#         return x


# -----------------------------------------------------------------------------------------
# # Custom MLP
# fc1 = nn.Linear(64 * 3, self.config.Decoder.in_dim)
# fc2 = nn.Linear(self.config.Decoder.in_dim, self.config.Decoder.in_dim)
# fc3 = nn.Linear(self.config.Decoder.in_dim, self.config.Decoder.hidden_dim)
# fc_out = nn.Linear(self.config.Decoder.hidden_dim, 1)
# decoder.fc_layers = nn.ModuleList([fc1, fc2, fc3, fc_out])


# class MLP(nn.Module):
#     """
#     A generalized MLP (Multi-Layer Perceptron) model that can act as either a 2-layer MLP or a 4-layer MLP based on the include_decoder_layers parameter.

#     Args:
#         in_dim (int): the dimension of input feature.
#         hidden_dim (int): the dimension of hidden layers.
#         out_dim (int): the dimension of output layer.
#         dropout_rate (float): the dropout rate during training.
#         include_decoder_layers (bool): whether or not to include the additional layers that are part of the MLP
#     """

#     def __init__(
#         self,
#         in_dim,
#         hidden_dim,
#         out_dim,
#         dropout_rate=0.1,
#         include_decoder_layers=False,
#     ):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.include_decoder_layers = include_decoder_layers

#         if self.include_decoder_layers:
#             self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#             self.fc3 = nn.Linear(hidden_dim, out_dim)
#             self.fc4 = nn.Linear(out_dim, 1)
#             torch.nn.init.normal_(self.fc4.weight)
#             self.dropout = nn.Dropout(dropout_rate)
#         else:
#             self.fc2 = nn.Linear(hidden_dim, 1)
#             self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)

#         if self.include_decoder_layers:
#             x = F.relu(x)
#             x = self.dropout(x)
#             x = self.fc3(x)
#             x = F.relu(x)
#             x = self.fc4(x)

#         return x
