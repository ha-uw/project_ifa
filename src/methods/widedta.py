import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .configs import ConfigLoader
from src.data.loading import TDCDataset
from src.data.preprocessing import MotifFetcher
from src.data.processing import (
    to_deepsmiles,
    seq_to_words,
    make_words_set,
    one_hot_words,
)

from src.modules.encoders import WideCNN
from src.modules.decoders import MLP
from src.modules.trainers import BaseDTATrainer
from src.data.evaluation import concordance_index


# =============================== Code ==================================
class WideDTA:
    def __init__(self, config_file: str, fast_dev_run=False) -> None:
        self._widedta = _WideDTA(config_file, fast_dev_run)

    def make_model(self):
        self._widedta.make_model()

    def train(self):
        self._widedta.train()

    def test(self):
        raise NotImplementedError("This will be implemented soon.")


class _WideDTADataHandler(TDCDataset):
    words_sets = {
        "Drug": {},
        "Target": {},
        "Motif": {},
    }
    drug_max_len: int
    target_max_len: int
    motif_max_len: int

    def __init__(
        self,
        dataset_name: str,
        split="train",
        path="./data",
        label_to_log=True,
        drug_transform=None,
        target_transform=None,
    ):
        super().__init__(
            name=dataset_name,
            split=split,
            path=path,
            label_to_log=label_to_log,
            drug_transform=drug_transform,
            target_transform=target_transform,
        )

        self._preprocess_data()

    def _preprocess_data(self):
        try:
            self._load_motifs()
            self._smiles_to_deep()
            self._convert_seqs_to_words()
            self._make_words_set()
            self.data.reset_index(drop=True, inplace=True)
        except Exception as e:
            print("Error preprocessing data:", e)

        return

    def _load_motifs(self):
        mf = MotifFetcher()
        motifs = mf.get_motifs(self.data, self.path, self.name)
        self.data = pd.merge(self.data, motifs, on=["Target_ID"], how="left").dropna()

    def _smiles_to_deep(self):
        self.data["Drug"] = self.data["Drug"].apply(lambda x: to_deepsmiles(x))

    def _convert_seqs_to_words(self):
        self.data["Drug"] = self.data["Drug"].apply(
            lambda x: seq_to_words(x, word_len=8)
        )

        self.data["Target"] = self.data["Target"].apply(
            lambda x: seq_to_words(x, word_len=3)
        )

        self.data["Motif"] = self.data["Motif"].apply(
            lambda x: seq_to_words(x, word_len=3)
        )

        self.drug_max_len = self.data["Drug"].str.len().max()
        self.target_max_len = self.data["Target"].str.len().max()
        self.motif_max_len = self.data["Motif"].str.len().max()

    def _make_words_set(self):
        self.words_sets["Drug"] = make_words_set(self.data["Drug"])
        self.words_sets["Target"] = make_words_set(self.data["Target"])
        self.words_sets["Motif"] = make_words_set(self.data["Motif"])

    def _fetch_data(self, index):
        drug, target, motif, label = (
            self.data["Drug"][index],
            self.data["Target"][index],
            self.data["Motif"][index],
            self.data["Y"][index],
        )

        drug = one_hot_words(
            drug,
            allowable_set=self.words_sets["Drug"],
            length=self.drug_max_len,
        )
        drug = torch.LongTensor(drug)

        target = one_hot_words(
            target,
            allowable_set=self.words_sets["Target"],
            length=self.target_max_len,
        )
        target = torch.LongTensor(target)

        motif = one_hot_words(
            motif,
            allowable_set=self.words_sets["Motif"],
            length=self.motif_max_len,
        )
        motif = torch.LongTensor(motif)

        label = torch.FloatTensor([label])

        return drug, target, motif, label

    def __getitem__(self, index):
        drug, target, motif, label = self._fetch_data(index)

        return drug, target, motif, label


class _WideDTATrainer(pl.LightningModule):
    """ """

    def __init__(
        self,
        drug_encoder,
        target_encoder,
        motif_encoder,
        decoder,
        lr=0.003,
        ci_metric=False,
        **kwargs
    ):

        super(_WideDTATrainer, self).__init__()
        # self.save_hyperparameters()
        self.lr = lr
        self.ci_metric = ci_metric
        self.drug_encoder = drug_encoder
        self.target_encoder = target_encoder
        self.motif_encoder = motif_encoder
        self.decoder = decoder

    def configure_optimizers(self):
        """
        Config adam as default optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x_drug, x_target, x_motif):
        """
        Forward propagation in WideDTA architecture.

        Args:
            x_drug: drug sequence encoding.
            x_target: target protein sequence encoding.
        """
        drug = self.drug_encoder(x_drug)
        target = self.target_encoder(x_target)
        motif = self.motif_encoder(x_motif)
        comb_emb = torch.cat((drug, target, motif), dim=1)

        output = self.decoder(comb_emb)

        return output

    def training_step(self, train_batch):
        """
        Compute and return the training loss on one step
        """
        x_drug, x_target, x_motif, y = train_batch
        y_pred = self(x_drug, x_target, x_motif)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        if self.ci_metric:
            ci = concordance_index(y, y_pred)
            self.logger.log_metrics({"train_step_ci": ci}, self.global_step)
        self.logger.log_metrics({"train_step_loss": loss}, self.global_step)
        self.log("train_step_loss_2", loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, valid_batch):
        x_drug, x_target, x_motif, y = valid_batch
        y_pred = self(x_drug, x_target, x_motif)
        loss = F.mse_loss(y_pred, y.view(-1, 1))

        if self.ci_metric:
            ci = concordance_index(y, y_pred)
            self.log("valid_ci", ci, on_epoch=True, on_step=False)
        self.log("valid_loss", loss, on_epoch=True, on_step=False)

        return loss

    def test_step(self, test_batch):
        """
        Compute and return the test loss on one step
        """
        x_drug, x_target, x_motif, y = test_batch
        y_pred = self(x_drug, x_target, x_motif)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        if self.ci_metric:
            ci = concordance_index(y, y_pred)
            self.log("test_ci", ci, on_epoch=True, on_step=False)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss


class _WideDTA:
    config: ConfigLoader

    def __init__(self, config_file: str, fast_dev_run=False) -> None:
        self._load_configs(config_file)
        self.fast_dev_run = fast_dev_run

    def _load_configs(self, config_path: str):
        cl = ConfigLoader()
        cl.load_config(config_path=config_path)

        self.config = cl

    def make_model(self):
        drug_encoder = WideCNN(
            sequence_length=self.config.Encoder.Drug.sequence_length,
            num_filters=self.config.Encoder.Drug.num_filters,
            kernel_size=self.config.Encoder.Drug.kernel_size,
        )

        target_encoder = WideCNN(
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_filters=self.config.Encoder.Target.num_filters,
            kernel_size=self.config.Encoder.Target.kernel_size,
        )

        motif_encoder = WideCNN(
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_filters=self.config.Encoder.Target.num_filters,
            kernel_size=self.config.Encoder.Target.kernel_size,
        )

        decoder = MLP(
            in_dim=self.config.Decoder.in_dim,
            hidden_dim=self.config.Decoder.hidden_dim,
            dropout_rate=self.config.Decoder.dropout_rate,
            num_fc_layers=3,
        )

        # Custom MLP
        fc1 = nn.Linear(self.config.Decoder.in_dim, self.config.Decoder.hidden_dim)
        fc2 = nn.Linear(self.config.Decoder.hidden_dim, 10)
        fc3 = nn.Linear(10, 1)
        decoder.fc_layers = nn.ModuleList([fc1, fc2, fc3])

        model = _WideDTATrainer(
            drug_encoder=drug_encoder,
            target_encoder=target_encoder,
            motif_encoder=motif_encoder,
            decoder=decoder,
            lr=self.config.Trainer.learning_rate,
            ci_metric=self.config.Trainer.ci_metric,
        )

        print(model)
        self.model = model

    def train(self):
        tb_logger = TensorBoardLogger(
            "outputs",
            name=self.config.Dataset.name,
        )

        pl.seed_everything(seed=self.config.General.random_seed, workers=True)

        # ---- set dataset ----
        train_dataset = _WideDTADataHandler(
            dataset_name=self.config.Dataset.name,
            split="train",
            path=self.config.Dataset.path,
        )
        valid_dataset = _WideDTADataHandler(
            dataset_name=self.config.Dataset.name,
            split="valid",
            path=self.config.Dataset.path,
        )
        test_dataset = _WideDTADataHandler(
            dataset_name=self.config.Dataset.name,
            split="test",
            path=self.config.Dataset.path,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=self.config.Trainer.train_batch_size,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            shuffle=False,
            batch_size=self.config.Trainer.test_batch_size,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.config.Trainer.test_batch_size,
        )

        # ---- set model ----
        if not hasattr(self, "model") or self.model is None:
            self.make_model()

        # ---- training and evaluation ----
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}", monitor="valid_loss", mode="min"
        )
        early_stop_callback = EarlyStopping(
            monitor="valid_loss",
            min_delta=self.config.General.min_delta,
            patience=self.config.General.patience_epochs,
            verbose=False,
            mode="min",
        )

        callbacks = [checkpoint_callback]

        if self.config.General.early_stop:
            callbacks.append(early_stop_callback)

        trainer = pl.Trainer(
            max_epochs=self.config.Trainer.max_epochs,
            accelerator="auto",
            devices="auto",
            logger=tb_logger,
            callbacks=callbacks,
            fast_dev_run=self.fast_dev_run,
        )

        trainer.fit(self.model, train_loader, valid_loader)
        trainer.test(
            self.model,
            test_loader,
        )
