import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

from .configs import ConfigLoader

from data.processing import tokenize_smiles, tokenize_target
from data.evaluation import concordance_index
from data.loading import TDCDataset
from modules.encoders import CNN
from modules.decoders import MLP
from modules.trainers import BaseDTATrainer


# =============================== Code ==================================
class DeepDTA:
    def __init__(self, config_file: str, fast_dev_run=False) -> None:
        self._deepdta = _DeepDTA(config_file, fast_dev_run)

    def make_model(self):
        self._deepdta.make_model()

    def train(self):
        self._deepdta.train()

    def test(self):
        raise NotImplementedError("This will be implemented soon.")


class DeepDTADataHandler(TDCDataset):
    def __init__(
        self,
        dataset_name: str,
        split="train",
        path="data",
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

    def _deepdta(self, drug, target):
        drug = torch.LongTensor(tokenize_smiles(drug))
        target = torch.LongTensor(tokenize_target(target))

        return drug, target

    def _fetch_data(self, index):
        drug, target, label = (
            self.data["Drug"][index],
            self.data["Target"][index],
            self.data["Y"][index],
        )

        label = torch.FloatTensor([label])
        drug, target = self._deepdta(drug, target)

        return drug, target, label

    def __getitem__(self, index):
        drug, target, label = self._fetch_data(index)
        return drug, target, label


class _DeepDTATrainer(BaseDTATrainer):
    """ """

    def __init__(self, drug_encoder, target_encoder, decoder, lr, ci_metric, **kwargs):
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
            ci = concordance_index(y, y_pred)
            self.log("valid_ci", ci, on_epoch=True, on_step=False)
        self.log("valid_loss", loss, on_epoch=True, on_step=False)

        return loss


class _DeepDTA:
    config: ConfigLoader

    def __init__(self, config_file: str, fast_dev_run=False) -> None:
        self._load_configs(config_file)
        self.fast_dev_run = fast_dev_run

    def _load_configs(self, config_path: str):
        cl = ConfigLoader()
        cl.load_config(config_path=config_path)

        self.config = cl

    def make_model(self):
        drug_encoder = CNN(
            num_embeddings=self.config.Encoder.Drug.num_embeddings,
            embedding_dim=self.config.Encoder.Drug.embedding_dim,
            sequence_length=self.config.Encoder.Drug.sequence_length,
            num_filters=self.config.Encoder.Drug.num_filters,
            kernel_size=self.config.Encoder.Drug.kernel_size,
            num_conv_layers=self.config.Encoder.Drug.num_conv_layers,
        )

        target_encoder = CNN(
            num_embeddings=self.config.Encoder.Target.num_embeddings,
            embedding_dim=self.config.Encoder.Target.embedding_dim,
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_filters=self.config.Encoder.Target.num_filters,
            kernel_size=self.config.Encoder.Target.kernel_size,
            num_conv_layers=self.config.Encoder.Target.num_conv_layers,
        )

        decoder = MLP(
            in_dim=self.config.Decoder.in_dim,
            hidden_dim=self.config.Decoder.hidden_dim,
            out_dim=self.config.Decoder.out_dim,
            dropout_rate=self.config.Decoder.dropout_rate,
            num_fc_layers=self.config.Decoder.num_fc_layers,
        )

        model = _DeepDTATrainer(
            drug_encoder=drug_encoder,
            target_encoder=target_encoder,
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
        train_dataset = DeepDTADataHandler(
            dataset_name=self.config.Dataset.name,
            split="train",
            path=self.config.Dataset.path,
        )
        valid_dataset = DeepDTADataHandler(
            dataset_name=self.config.Dataset.name,
            split="valid",
            path=self.config.Dataset.path,
        )
        test_dataset = DeepDTADataHandler(
            dataset_name=self.config.Dataset.name,
            split="test",
            path=self.config.Dataset.path,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=self.config.Trainer.train_batch_size,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            shuffle=False,
            batch_size=self.config.Trainer.test_batch_size,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.config.Trainer.test_batch_size,
            pin_memory=True,
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
