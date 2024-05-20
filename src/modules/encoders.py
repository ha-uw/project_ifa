import torch
import torch.nn as nn
import torch.nn.functional as F

# Former DeepDTA


# Encoder
class CNNEncoder(nn.Module):
    """ """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        sequence_length,
        num_kernels,
        kernel_length,
    ):
        super(CNNEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv1 = nn.Conv1d(
            in_channels=sequence_length,
            out_channels=num_kernels,
            kernel_size=kernel_length[0],
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_kernels,
            out_channels=num_kernels * 2,
            kernel_size=kernel_length[1],
        )
        self.conv3 = nn.Conv1d(
            in_channels=num_kernels * 2,
            out_channels=num_kernels * 3,
            kernel_size=kernel_length[2],
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


# Decoder
class MLPDecoder(nn.Module):
    """
    A generalized MLP model that can act as either a 2-layer MLPDecoder or a 4-layer MLPDecoder based on the include_decoder_layers parameter.

    Args:
        in_dim (int): the dimension of input feature.
        hidden_dim (int): the dimension of hidden layers.
        out_dim (int): the dimension of output layer.
        dropout_rate (float): the dropout rate during training.
        include_decoder_layers (bool): whether or not to include the additional layers that are part of the MLPDecoder
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout_rate=0.1,
        include_decoder_layers=False,
    ):
        super(MLPDecoder, self).__init__()
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
