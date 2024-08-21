"""
decoders.py

This module contains the implementation of various decoders used in the project. 
Decoders are components of a machine learning model that transform the output of 
an encoder back into the original high-dimensional space or into a different useful format.
"""

# Imports ---------------------------------------------------------------
import torch
from torch import nn
from torch.nn import functional as F


# Code ------------------------------------------------------------------
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.
    Args:
        in_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden layers.
        out_dim (int): Dimensionality of the output.
        dropout_rate (float, optional): Dropout rate to apply between layers. Defaults to 0.1.
        num_fc_layers (int, optional): Number of fully connected layers. Must be at least 3. Defaults to 3.
    Attributes:
        fc_layers (nn.ModuleList): List of fully connected layers.
        num_fc_layers (int): Number of fully connected layers.
        dropout (nn.Dropout): Dropout layer.
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
