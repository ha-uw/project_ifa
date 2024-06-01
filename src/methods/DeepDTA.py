import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from data.loading import TDCDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from .configs import ConfigLoader

from modules.encoders import CNN
from modules.decoders import MLP
from modules.trainers import DeepDTATrainer


# ============================== Configs ================================
from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")


# =============================== Code ==================================
class DeedDTA:
    config: ConfigLoader

    def __init__(self, config_file: str) -> None:
        self._load_configs(config_file)

    def _load_configs(self, config_path: str):
        cl = ConfigLoader()
        cl.load_config(config_path=config_path)

        self.config = cl

    def _make_model(self):
        drug_encoder = CNN(
            num_embeddings=self.config.Encoder.Drug.num_embeddings,
            embedding_dim=self.config.Encoder.Drug.embedding_dim,
            sequence_length=self.config.Encoder.Drug.sequence_length,
            num_kernels=self.config.Encoder.Drug.num_filters,
            kernel_length=self.config.Encoder.Drug.filter_length,
        )

        target_encoder = CNN(
            num_embeddings=self.config.Encoder.Target.num_embeddings,
            embedding_dim=self.config.Encoder.Target.embedding_dim,
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_kernels=self.config.Encoder.Target.num_filters,
            kernel_length=self.config.Encoder.Target.filter_length,
        )

        decoder = MLP(
            in_dim=self.config.Decoder.in_dim,
            hidden_dim=self.config.Decoder.hidden_dim,
            out_dim=self.config.Decoder.out_dim,
            dropout_rate=self.config.Decoder.dropout_rate,
            include_decoder_layers=self.config.Decoder.include_decoder_layers,
        )

        model = DeepDTATrainer(
            drug_encoder=drug_encoder,
            target_encoder=target_encoder,
            decoder=decoder,
            lr=self.config.Trainer.learning_rate,
            ci_metric=self.config.Trainer.ci_metric,
        )

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
        self._make_model()
        print(self.model)

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
            fast_dev_run=self.config.General.fast_dev_run,
        )

        trainer.fit(self.model, train_loader, valid_loader)
        trainer.test(test_loader)
