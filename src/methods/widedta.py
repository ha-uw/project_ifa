import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .configs import ConfigLoader
from data.loading import TDCDataset
from modules.encoders import WideCNN
from modules.decoders import MLP
from modules.trainers import BaseDTATrainer
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


class _WideDTATrainer(BaseDTATrainer):
    """ """

    def __init__(
        self,
        drug_encoder,
        target_encoder,
        motif_encoder,
        decoder,
        lr,
        ci_metric,
        **kwargs
    ):
        super().__init__(drug_encoder, target_encoder, decoder, lr, ci_metric, **kwargs)

        self.motif_encoder = motif_encoder

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

    def validation_step(self, valid_batch, batch_idx):
        x_drug, x_target, y = valid_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))

        if self.ci_metric:
            ci = concordance_index(y, y_pred)
            self.log("valid_ci", ci, on_epoch=True, on_step=False)
        self.log("valid_loss", loss, on_epoch=True, on_step=False)

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
            embedding_dim=self.config.Encoder.Drug.embedding_dim,
            sequence_length=self.config.Encoder.Drug.sequence_length,
            num_filters=self.config.Encoder.Drug.num_filters,
            filter_length=self.config.Encoder.Drug.filter_length,
        )

        target_encoder = WideCNN(
            num_embeddings=self.config.Encoder.Target.num_embeddings,
            embedding_dim=self.config.Encoder.Target.embedding_dim,
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_filters=self.config.Encoder.Target.num_filters,
            filter_length=self.config.Encoder.Target.filter_length,
        )

        motif_encoder = WideCNN(
            num_embeddings=self.config.Encoder.Target.num_embeddings,
            embedding_dim=self.config.Encoder.Target.embedding_dim,
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_filters=self.config.Encoder.Target.num_filters,
            filter_length=self.config.Encoder.Target.filter_length,
        )

        decoder = MLP(
            in_dim=self.config.Decoder.in_dim,
            hidden_dim=self.config.Decoder.hidden_dim,
            out_dim=self.config.Decoder.out_dim,
            dropout_rate=self.config.Decoder.dropout_rate,
            num_fc_layers=3,
        )

        model = _WideDTATrainer(
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
        train_dataset = TDCDataset(
            name=self.config.Dataset.name, split="train", path=self.config.Dataset.path
        )
        valid_dataset = TDCDataset(
            name=self.config.Dataset.name, split="valid", path=self.config.Dataset.path
        )
        test_dataset = TDCDataset(
            name=self.config.Dataset.name, split="test", path=self.config.Dataset.path
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=self.config.Trainer.train_batch_size,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            shuffle=True,
            batch_size=self.config.Trainer.test_batch_size,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=True,
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
