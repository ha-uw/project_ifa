import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data.lightning import LightningDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .configs import ConfigLoader
from data.loading import TDCDataset
from modules.decoders import MLP
from modules.encoders import CNN, GAT_GCN, GAT, GCN, GIN
from modules.trainers import BaseDTATrainer
from data.evaluation import concordance_index


class GraphDTA:
    def __init__(self, config_file: str, drug_encoder: str, fast_dev_run=False) -> None:
        self._graphdta = _GraphDTA(
            config_file, drug_encoder=drug_encoder, fast_dev_run=fast_dev_run
        )

    def make_model(self):
        self._graphdta.make_model()

    def train(self):
        self._graphdta.train()

    def predict(self):
        raise NotImplementedError


class _GraphDTATrainer(BaseDTATrainer):
    """"""

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


class _GraphDTA:
    config: ConfigLoader

    def __init__(self, config_file: str, drug_encoder: str, fast_dev_run=False) -> None:
        self._load_configs(config_file)
        self.fast_dev_run = fast_dev_run
        self.drug_encoder = drug_encoder

    def _load_configs(self, config_path: str):
        cl = ConfigLoader()
        cl.load_config(config_path=config_path)

        self.config = cl

    def make_model(self):
        match self.drug_encoder:
            case "GAT":
                drug_encoder = GAT(
                    input_dim=self.config.Encoder.GAT.input_dim,
                    num_heads=self.config.Encoder.GAT.num_heads,
                    output_dim=self.config.Encoder.GAT.output_dim,
                )
            case "GCN":
                drug_encoder = GCN(
                    input_dim=self.config.Encoder.GCN.input_dim,
                    output_dim=self.config.Encoder.GCN.output_dim,
                )
            case "GIN":
                drug_encoder = GIN(
                    input_dim=self.config.Encoder.GIN.input_dim,
                    num_filters=self.config.Encoder.GIN.num_filters,
                    output_dim=self.config.Encoder.GIN.output_dim,
                )
            case "GAT_GCN":
                drug_encoder = GAT_GCN(
                    input_dim=self.config.Encoder.GAT_GCN.input_dim,
                    num_heads=self.config.Encoder.GAT_GCN.num_heads,
                    output_dim=self.config.Encoder.GAT_GCN.output_dim,
                )
            case _:
                raise Exception("Invalid drug encoder.")

        target_encoder = CNN(
            embedding_dim=self.config.Encoder.Target.embedding_dim,
            num_embeddings=self.config.Encoder.Target.num_embeddings,
            filter_length=self.config.Encoder.Target.filter_length,
            num_filters=self.config.Encoder.Target.num_filters,
            sequence_length=self.config.Encoder.Target.sequence_length,
        )
        decoder = MLP(
            dropout_rate=self.config.Decoder.dropout_rate,
            hidden_dim=self.config.Decoder.hidden_dim,
            in_dim=self.config.Decoder.in_dim,
            include_decoder_layers=self.config.Decoder.include_decoder_layers,
            out_dim=self.config.Decoder.out_dim,
        )
        model = _GraphDTATrainer(
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
            name=f"GraphDTA_{self.drug_encoder}_{self.config.Dataset.name}",
        )

        pl.seed_everything(seed=self.config.General.random_seed, workers=True)

        # ---- set dataset ----
        train_dataset = TDCDataset(
            name=self.config.Dataset.name,
            split="train",
            path=self.config.Dataset.path,
            mode_drug="gcn",
        )
        valid_dataset = TDCDataset(
            name=self.config.Dataset.name,
            split="valid",
            path=self.config.Dataset.path,
            mode_drug="gcn",
        )
        test_dataset = TDCDataset(
            name=self.config.Dataset.name,
            split="test",
            path=self.config.Dataset.path,
            mode_drug="gcn",
        )

        pl_dataset = LightningDataset(
            train_dataset,
            valid_dataset,
            test_dataset,
            batch_size=self.config.Trainer.train_batch_size,
        )

        train_loader = pl_dataset.train_dataloader()
        valid_loader = pl_dataset.val_dataloader()
        test_loader = pl_dataset.test_dataloader()

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
