import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# from kale.evaluate.metrics import concord_index


# Encoder
class CNNEncoder(nn.Module):
    """
    The DeepDTA's CNN encoder module, which comprises three 1D-convolutional layers and one max-pooling layer.
    The module is applied to encoding drug/target sequence information, and the input should be transformed information
    with integer/label encoding. The original paper is `"DeepDTA: deep drug–target binding affinity prediction"
    <https://academic.oup.com/bioinformatics/article/34/17/i821/5093245>`_ .

    Args:
        num_embeddings (int): Number of embedding labels/categories, depends on the types of encoding sequence.
        embedding_dim (int): Dimension of embedding labels/categories.
        sequence_length (int): Max length of input sequence.
        num_kernels (int): Number of kernels (filters).
        kernel_length (int): Length of kernel (filter).
    """

    def __init__(
        self, num_embeddings, embedding_dim, sequence_length, num_kernels, kernel_length
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
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.include_decoder_layers:
            x = self.dropout(F.relu(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

        return x


# Trainer
class BaseDTATrainer(pl.LightningModule):
    """
    Base class for all drug target encoder-decoder architecture models, which is based on pytorch lightning wrapper,
    for more details about pytorch lightning, please check https://github.com/PyTorchLightning/pytorch-lightning.
    If you inherit from this class, a forward pass function must be implemented.

    Args:
        drug_encoder: drug information encoder.
        target_encoder: target information encoder.
        decoder: drug-target representations decoder.
        lr: learning rate. (default: 0.001)
        ci_metric: calculate the Concordance Index (CI) metric, and the operation is time-consuming for large-scale
        dataset. (default: :obj:`False`)
    """

    def __init__(
        self, drug_encoder, target_encoder, decoder, lr=0.001, ci_metric=False, **kwargs
    ):
        super(BaseDTATrainer, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.ci_metric = ci_metric
        self.drug_encoder = drug_encoder
        self.target_encoder = target_encoder
        self.decoder = decoder

    def configure_optimizers(self):
        """
        Config adam as default optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x_drug, x_target):
        """
        Same as :meth:`torch.nn.Module.forward()`
        """
        raise NotImplementedError("Forward pass needs to be defined.")

    def training_step(self, train_batch, batch_idx):
        """
        Compute and return the training loss on one step
        """
        x_drug, x_target, y = train_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        if self.ci_metric:
            ci = concord_index(y, y_pred)
            self.logger.log_metrics({"train_step_ci": ci}, self.global_step)
        self.logger.log_metrics({"train_step_loss": loss}, self.global_step)
        self.log("train_step_loss_2", loss, prog_bar=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        """
        Compute and return the validation loss on one step
        """
        x_drug, x_target, y = valid_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Compute and return the test loss on one step
        """
        x_drug, x_target, y = test_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        if self.ci_metric:
            ci = concord_index(y, y_pred)
            self.log("test_ci", ci, on_epoch=True, on_step=False)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        return loss


class DeepDTATrainer(BaseDTATrainer):
    """
    An implementation of DeepDTA model based on BaseDTATrainer.
    Args:
        drug_encoder: drug CNN encoder.
        target_encoder: target CNN encoder.
        decoder: drug-target MLP decoder.
        lr: learning rate.
    """

    def __init__(
        self, drug_encoder, target_encoder, decoder, lr=0.001, ci_metric=False, **kwargs
    ):
        super().__init__(drug_encoder, target_encoder, decoder, lr, ci_metric, **kwargs)

    def forward(self, x_drug, x_target):
        """
        Forward propagation in DeepDTA architecture.

        Args:
            x_drug: drug sequence encoding.
            x_target: target protein sequence encoding.
        """
        drug_emb = self.drug_encoder(x_drug)
        target_emb = self.target_encoder(x_target)
        comb_emb = torch.cat((drug_emb, target_emb), dim=1)
        output = self.decoder(comb_emb)
        return output

    def validation_step(self, valid_batch, batch_idx):
        x_drug, x_target, y = valid_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        if self.ci_metric:
            ci = concord_index(y, y_pred)
            self.log("valid_ci", ci, on_epoch=True, on_step=False)
        self.log("valid_loss", loss, on_epoch=True, on_step=False)
        return loss
