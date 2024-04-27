import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

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
        self.gcn1 = nn.GATConv(
            num_features_xd, num_features_xd, heads=10, dropout=dropout
        )
        self.gcn2 = nn.GATConv(num_features_xd * 10, output_dim, dropout=dropout)
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

        self.conv1 = nn.GATConv(
            num_features_xd, num_features_xd, heads=10, dropout=dropout
        )
        self.conv2 = nn.GCNConv(num_features_xd * 10, num_features_xd * 10)
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


# TODO
class GCNNet(nn.Module):
    """
    Inherits from nn.Module for protein sequence processing.
    Defines GCN convolutional layers for graph data representing molecules.
    """

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
        super(GCNNet, self).__init__(
            num_features_xd, num_features_xt, embed_dim, output_dim, dropout
        )
        self.n_output = n_output  # Number of output features (e.g., binding affinity)

        # GCN layers for graph data (representing molecules)
        self.conv1 = nn.GCNConv(num_features_xd, num_features_xd)  # First GCN layer
        self.conv2 = nn.GCNConv(
            num_features_xd, num_features_xd * 2
        )  # Second GCN layer, doubles feature dim
        self.conv3 = nn.GCNConv(
            num_features_xd * 2, num_features_xd * 4
        )  # Third GCN layer, doubles feature dim again

        # nn.Linear layers for processing graph data after GCN convolutions
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)

        # Combined layers for final prediction
        self.fc1 = nn.Linear(
            2 * output_dim, 1024
        )  # Combined feature dimension after concatenating with protein sequence
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # Graph data processing with GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x, training=self.training)  # Apply dropout for regularization

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x, training=self.training)
        x = global_add_pool(
            x, batch
        )  # Global add pooling to aggregate node features in the graph

        # Process protein data (inherited from nn.Module)
        xt = self.process_protein_sequence(target)

        # Combine features and final layers
        xc = torch.cat((x, xt), 1)  # Concatenate graph and protein sequence features
        xc = self.fc1(xc)
        xc = F.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = F.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)  # Final output layer

        return out


class GINConvNet(nn.Module):
    """
    Inherits from nn.Module for protein sequence processing.
    Defines nn.GINConv layers for graph data representing molecules.
    """

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
        super(nn.GINConvNet, self).__init__(
            num_features_xd, num_features_xt, embed_dim, output_dim, dropout
        )
        self.n_output = n_output  # Number of output features (e.g., binding affinity)

        # Hidden layer dimension for nn.GINConv layers
        self.hidden_dim = 32

        # nn.GINConv layers with nn.Sequential nn.Linear layers and batch normalization
        self.conv1 = nn.GINConv(
            nn.Sequential(
                nn.Linear(num_features_xd, self.hidden_dim),
                nn.Relu(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        )
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_dim)

        self.conv2 = nn.GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Relu(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        )
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_dim)

        self.conv3 = nn.GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Relu(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        )
        self.bn3 = torch.nn.BatchNorm1d(self.hidden_dim)

        self.conv4 = nn.GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Relu(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        )
        self.bn4 = torch.nn.BatchNorm1d(self.hidden_dim)

        self.conv5 = nn.GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Relu(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        )
        self.bn5 = torch.nn.BatchNorm1d(self.hidden_dim)

        # Final nn.Linear layer for processing nn.GINConv output
        self.fc1_xd = nn.Linear(self.hidden_dim, output_dim)

        # Combined layers for final prediction
        self.fc1 = nn.Linear(
            2 * output_dim, 1024
        )  # Combined feature dimension after concatenation
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # nn.GINConv layers with nn.Relu activation and dropout
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout(x, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout(x, training=self.training)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = self.dropout(x, training=self.training)

        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = self.dropout(x, training=self.training)

        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)  # Use global add pool for nn.GINConv
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # Inherit protein sequence processing from BaseDrugTargetModel
        xt = self.process_protein_sequence(target)

        # Combine features
        xc = torch.cat((x, xt), dim=1)  # Concatenate drug and protein features

        # Final layers for prediction
        xc = F.relu(self.fc1(xc))
        xc = F.dropout(xc, p=0.5, training=self.training)
        xc = self.fc2(xc)
        xc = F.dropout(xc, p=0.5, training=self.training)
        output = self.out(xc)

        return output
