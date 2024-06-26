import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path
import json

from .configs import ConfigLoader
from data.loading import TDCDataset
from data.preprocessing import MotifFetcher
from data.processing import (
    to_deepsmiles,
    seq_to_words,
    make_words_dict,
    one_hot_words,
)

from modules.encoders import WideCNN
from modules.decoders import MLP
from data.evaluation import concordance_index


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
    MAX_DRUG_LEN = 85
    MAX_TARGET_LEN = 1000
    MAX_MOTIF_LEN = 500
    RAW_DATA: pd.DataFrame

    word_to_int = {"Drug": {}, "Target": {}, "Motif": {}}

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

        self.RAW_DATA = self.data.copy()
        self._preprocess_data()

    def __len__(self):
        return len(self.data)

    def _preprocess_data(self):
        try:
            self._filter_data()
            self._load_motifs()
            self._smiles_to_deep()
            self._convert_seqs_to_words()
            self._load_words_dict()
            self.data.reset_index(drop=True, inplace=True)
        except Exception as e:
            print("Error preprocessing data:", e)
            raise Exception(e)

    def _filter_data(self):
        pass

    def _load_motifs(self):
        mf = MotifFetcher()
        motifs = mf.get_motifs(self.data, self.path, self.name)

        # Merget dfs and drop NaN
        self.data = pd.merge(
            self.RAW_DATA, motifs, on=["Target_ID"], how="left"
        ).dropna()

        # Reorder columns
        self.data = self.data[["Drug_ID", "Drug", "Target_ID", "Target", "Motif", "Y"]]

    def _smiles_to_deep(self):
        self.data["Drug"] = self.data["Drug"].apply(lambda x: to_deepsmiles(x))

    def _convert_seqs_to_words(self):
        self.data["Drug"] = self.data["Drug"].apply(
            lambda x: seq_to_words(x, word_len=8, max_length=self.MAX_DRUG_LEN)
        )

        self.data["Target"] = self.data["Target"].apply(
            lambda x: seq_to_words(x, word_len=3, max_length=self.MAX_TARGET_LEN)
        )

        self.data["Motif"] = self.data["Motif"].apply(
            lambda x: seq_to_words(x, word_len=3, max_length=self.MAX_MOTIF_LEN)
        )

    def _make_words_dics(self):
        self.word_to_int["Drug"] = make_words_dict(self.data["Drug"])
        self.word_to_int["Target"] = make_words_dict(self.data["Target"])
        self.word_to_int["Motif"] = make_words_dict(self.data["Motif"])

    def _load_words_dict(self):
        file_path = Path(self.path, self.name, f"{self.name}_words.json")
        if file_path.is_file():
            with open(file_path) as f:
                self.word_to_int = json.load(f)
            print("Words dictionary loaded from file.")
        else:
            self._make_words_dics()
            with open(file_path, "w") as f:
                f.write(json.dumps(self.word_to_int))

    def _fetch_data(self, index):
        drug, target, motif, label = (
            self.data["Drug"][index],
            self.data["Target"][index],
            self.data["Motif"][index],
            self.data["Y"][index],
        )

        drug = one_hot_words(
            drug,
            word_to_int=self.word_to_int["Drug"],
            length=self.MAX_DRUG_LEN,
        )
        drug = torch.LongTensor(drug)

        target = one_hot_words(
            target,
            word_to_int=self.word_to_int["Target"],
            length=self.MAX_TARGET_LEN,
        )
        target = torch.LongTensor(target)

        motif = one_hot_words(
            motif,
            word_to_int=self.word_to_int["Motif"],
            length=self.MAX_MOTIF_LEN,
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
        **kwargs,
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
            num_embeddings=self.config.Encoder.Drug.num_embeddings,
            sequence_length=self.config.Encoder.Drug.sequence_length,
            num_filters=self.config.Encoder.num_filters,
            kernel_size=self.config.Encoder.kernel_size,
        )

        target_encoder = WideCNN(
            num_embeddings=self.config.Encoder.Target.num_embeddings,
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_filters=self.config.Encoder.num_filters,
            kernel_size=self.config.Encoder.kernel_size,
        )

        motif_encoder = WideCNN(
            num_embeddings=self.config.Encoder.Motif.num_embeddings,
            sequence_length=self.config.Encoder.Motif.sequence_length,
            num_filters=self.config.Encoder.num_filters,
            kernel_size=self.config.Encoder.kernel_size,
        )

        decoder = MLP(
            in_dim=self.config.Decoder.in_dim,
            hidden_dim=self.config.Decoder.hidden_dim,
            dropout_rate=self.config.Decoder.dropout_rate,
            num_fc_layers=4,
        )

        # Custom MLP
        fc1 = nn.Linear(192, self.config.Decoder.in_dim)
        fc2 = nn.Linear(self.config.Decoder.in_dim, self.config.Decoder.in_dim)
        fc3 = nn.Linear(self.config.Decoder.in_dim, self.config.Decoder.hidden_dim)
        fc4 = nn.Linear(self.config.Decoder.hidden_dim, 1)
        decoder.fc_layers = nn.ModuleList([fc1, fc2, fc3, fc4])

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
            name=f"{self.config.Dataset.name}_WideDTA",
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

        self.TRAIN_WORDSET_LEN = 125

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
