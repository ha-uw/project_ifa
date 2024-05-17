import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# Model parameters
NUM_FEATURES_XD = 78
NUM_FEATURES_XT = 25
num_filters = 32
EMBED_DIM = 128
OUTPUT_DIM = 128
DROPOUT = 0.2
NUM_HEADS = 10
KERNEL_SIZE = 8

# GAT_GCN
conv_xt1_in_channels = 1000


class GATNet(nn.Module):
    """
    GATNet class implements a Graph Attention Network (GAT) for graph-based and sequence-based data fusion.

    Args:
        num_features_xd (int): Number of features for the graph data (default: 78).
        n_output (int): Number of output units (default: 1).
        num_features_xt (int): Number of features for the protein sequence data (default: 25).
        num_filters (int): Number of filters for the protein sequence convolutional layer (default: 32).
        embed_dim (int): Dimension of the embedding for the protein sequence data (default: 128).
        output_dim (int): Dimension of the output features (default: 128).
        dropout (float): Dropout rate (default: 0.2).

    Attributes:
        gcn1 (nn.GATConv): Graph Convolutional Layer 1.
        gcn2 (nn.GATConv): Graph Convolutional Layer 2.
        fc_g1 (nn.Linear): Fully connected layer for graph data.
        embedding_xt (nn.Embedding): Embedding layer for protein sequence data.
        conv_xt1 (nn.Conv1d): Convolutional layer for protein sequence data.
        fc_xt (nn.Linear): Fully connected layer for protein sequence data.
        fc1 (nn.Linear): Fully connected layer 1 for feature combination.
        fc2 (nn.Linear): Fully connected layer 2 for feature combination.
        out (nn.Linear): Output layer.

    """

    def __init__(
        self,
        num_features_xd=78,
        n_output=1,
        num_features_xt=25,
        num_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.2,
    ):
        super(GATNet, self).__init__()

        # Graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # Protein sequence processing layers
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(
            in_channels=embed_dim, out_channels=num_filters, kernel_size=8
        )
        self.fc_xt = nn.Linear(32 * 121, output_dim)

        # Combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # Graph data processing
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc_g1(x)
        x = F.relu(x)

        # Protein processing
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = F.relu(conv_xt)

        # Flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt(conv_xt)

        # Combine features and final layers
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)
        xc = self.fc2(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)
        out = self.out(xc)

        return out


class GAT_GCN(nn.Module):
    """ """

    def __init__(
        self,
        n_output=1,
        num_features_xd=78,
        num_features_xt=25,
        num_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.2,
    ):
        super(GAT_GCN, self).__init__()

        self.conv1 = GATConv(
            num_features_xd, num_features_xd, heads=10, dropout=dropout
        )
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)

        # Protein sequence processing layers
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(
            in_channels=1000,
            out_channels=num_filters,
            kernel_size=8,  # TODO check why this is hardcoded to 1000
        )
        self.fc_xt = nn.Linear(32 * 121, output_dim)

        # Combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # Graph data processing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        x = self.fc_g1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc_g2(x)

        # Process protein data (inherited from nn.Module)
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        #  TODO figure out why there's no relu here

        # Flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt(conv_xt)

        # Combine features and final layers
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)
        xc = self.fc2(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)
        out = self.out(xc)

        return out


class GCNNet(nn.Module):
    """ """

    def __init__(
        self,
        n_output=1,
        num_filters=32,
        embed_dim=128,
        num_features_xd=78,
        num_features_xt=25,
        output_dim=128,
        dropout=0.2,
    ):
        super(GCNNet, self).__init__()
        self.n_output = n_output

        # GCN layers for graph data (representing molecules)
        self.conv1 = GCNConv(num_features_xd, num_features_xd, dropout=dropout)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2, dropout=dropout)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4, dropout=dropout)

        # nn.Linear layers for processing graph data after GCN convolutions
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)

        # Protein sequence processing layers
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(
            in_channels=1000,
            out_channels=num_filters,
            kernel_size=8,  # TODO check why this is hardcoded to 1000
        )
        self.fc_xt = nn.Linear(32 * 121, output_dim)

        # Combined layers for final prediction
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # Graph data processing with GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)

        # Flatten
        x = self.fc_g1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc_g2(x)
        x = F.dropout(x)

        # Protein processing
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)

        # Flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt(xt)

        # Combine features and final layers
        xc = torch.cat((x, xt), 1)  # Concatenate graph and protein sequence features
        xc = self.fc1(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)
        xc = self.fc2(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)

        out = self.out(xc)  # Final output layer

        return out


class GINConvNet(nn.Module):
    def __init__(
        self,
        n_output=1,
        num_features_xd=78,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.2,
    ):
        super().__init__()
        self.n_output = n_output

        # GIN layers
        self.conv1 = GINConv(
            nn=nn.Sequential(
                nn.Linear(num_features_xd, 32), F.relu(), nn.Linear(32, 32)
            )
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = GINConv(
            nn=nn.Sequential(nn.Linear(32, 32), F.relu(), nn.Linear(32, 32))
        )
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = GINConv(
            nn=nn.Sequential(nn.Linear(32, 32), F.relu(), nn.Linear(32, 32))
        )
        self.bn3 = nn.BatchNorm1d(32)

        self.conv4 = GINConv(
            nn=nn.Sequential(nn.Linear(32, 32), F.relu(), nn.Linear(32, 32))
        )
        self.bn4 = nn.BatchNorm1d(32)

        self.conv5 = GINConv(
            nn=nn.Sequential(nn.Linear(32, 32), F.relu(), nn.Linear(32, 32))
        )
        self.bn5 = nn.BatchNorm1d(32)

        self.fc1_xd = nn.Linear(32, output_dim)

        # Protein sequence embedding
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(1000, n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

        # Combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

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
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Protein processing
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)

        # Flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt(xt)

        xc = torch.cat((x, xt), dim=1)
        xc = self.fc1(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)
        xc = self.fc2(xc)
        xc = F.relu(xc)
        xc = F.dropout(xc)

        out = self.out(xc)

        return out
