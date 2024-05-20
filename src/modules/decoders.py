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
    """ """

    def __init__(
        self,
        num_features_xd=78,
        n_output=1,
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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph data processing
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc_g1(x)
        x = F.relu(x)

        return x


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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

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

        return x


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

        return x


class GINConvNet(nn.Module):
    def __init__(
        self,
        n_output=1,
        num_features_xd=78,
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
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x
