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
    A generalized MLP (Multi-Layer Perceptron) model that can act as either a 2-layer MLP or a 4-layer MLP based on the include_decoder_layers parameter.

    Args:
        in_dim (int): the dimension of input feature.
        hidden_dim (int): the dimension of hidden layers.
        out_dim (int): the dimension of output layer.
        dropout_rate (float): the dropout rate during training.
        include_decoder_layers (bool): whether or not to include the additional layers that are part of the MLP
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout_rate=0.1,
        include_decoder_layers=False,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.include_decoder_layers = include_decoder_layers

        if self.include_decoder_layers:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, out_dim)
            self.fc4 = nn.Linear(out_dim, 1)
            torch.nn.init.normal_(self.fc4.weight)
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.fc2 = nn.Linear(hidden_dim, out_dim)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if self.include_decoder_layers:
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)

        return x
