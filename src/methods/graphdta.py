"""
graphdta.py

Module for implementing the GraphDTA model and its training pipeline.
"""

import torch
import json
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data.lightning import LightningDataset
from torch_geometric import data as pyg_data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import KFold
from pathlib import Path

from .configs import ConfigLoader
from src.data.loading import TDCDataset
from src.data.processing import smile_to_graph, tokenise_target
from src.modules.decoders import MLP
from src.modules.encoders import CNN, GAT_GCN, GAT, GCN, GIN
from src.modules.trainers import DTATrainer


# =============================== Code ==================================
class GraphDTA:
    def __init__(
        self, config_file: str, drug_encoder: str = "GCN", fast_dev_run=False
    ) -> None:
        self._graphdta = _GraphDTA(
            config_file, drug_encoder=drug_encoder, fast_dev_run=fast_dev_run
        )

    def run_k_fold_validation(self, n_splits=5):
        self._graphdta.run_k_fold_validation(n_splits=n_splits)

    def resume_training(self, version, last_completed_fold):
        self._graphdta.resume_training(
            version=version, last_completed_fold=last_completed_fold
        )


class GraphDTADataHandler(Dataset):
    def __init__(self, dataset: Dataset, target_max_len=1200):
        self.dataset = dataset
        self.path = dataset.path
        self.dataset_name = dataset.name
        self.data = dataset.data.copy(deep=True)
        self.target_max_len = target_max_len

    def _process_data(self, drug, target, label):
        c_size, features, edge_index = smile_to_graph(drug)
        drug = pyg_data.Data(
            x=torch.Tensor(features),
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            y=torch.Tensor([label]),
        )
        drug.__setitem__("c_size", torch.LongTensor([c_size]))
        target = torch.LongTensor(tokenise_target(target, self.target_max_len))

        return drug, target

    def __len__(self):
        return len(self.data)

    def slice_data(self, indices: list):
        self.data = self.data.iloc[indices]
        print("Fold size:", self.data.shape[0])
        self._filter_data()

    def _filter_data(self):
        original_size = self.data.shape[0]
        self.data.drop_duplicates(subset=["Drug", "Target"], inplace=True)
        self.data.fillna("")
        self.data.reset_index(drop=True, inplace=True)

        n_rows_out = original_size - self.data.shape[0]

        print(
            f"Rows filtered out: {n_rows_out} ({(n_rows_out * 100)/original_size :.2f}%)"
        )

    def _fetch_data(self, index):
        drug, target, label = (
            self.data["Drug"][index],
            self.data["Target"][index],
            self.data["Y"][index],
        )

        label = torch.FloatTensor([label])
        drug, target = self._process_data(drug, target, label)

        return drug, target, label

    def __getitem__(self, index):
        drug, target, label = self._fetch_data(index)

        return drug, target, label


class GraphDTATrainer(DTATrainer):
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
            kernel_size=self.config.Encoder.Target.kernel_size,
            num_filters=self.config.Encoder.Target.num_filters,
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_conv_layers=self.config.Encoder.Target.num_conv_layers,
        )
        decoder = MLP(
            dropout_rate=self.config.Decoder.dropout_rate,
            hidden_dim=self.config.Decoder.hidden_dim,
            out_dim=self.config.Decoder.out_dim,
            in_dim=self.config.Decoder.in_dim,
            num_fc_layers=self.config.Decoder.num_fc_layers,
        )
        model = GraphDTATrainer(
            drug_encoder=drug_encoder,
            target_encoder=target_encoder,
            decoder=decoder,
            lr=self.config.Trainer.learning_rate,
            ci_metric=self.config.Trainer.ci_metric,
        )

        print(model)
        self.model = model

    def _make_data_loaders(self, train_dataset, val_dataset):
        pl_dataset = LightningDataset(
            train_dataset,
            val_dataset,
            batch_size=self.config.Trainer.train_batch_size,
        )

        train_loader = pl_dataset.train_dataloader()
        val_loader = pl_dataset.val_dataloader()

        return train_loader, val_loader

    def _get_logger(self, fold_n):
        output_dir = Path("outputs", f"GraphDTA_{self.drug_encoder}")
        dataset_dir = Path(output_dir, self.config.Dataset.name)

        dataset_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, "_version"):
            version_dirs = [d for d in dataset_dir.glob("version_*/") if d.is_dir()]
            version_names = [d.name for d in version_dirs]

            if version_names:
                latest_version = max(version_names, key=lambda x: int(x.split("_")[1]))
                next_version_num = int(latest_version.split("_")[1]) + 1
            else:
                next_version_num = 0

            self._version = str(next_version_num)

        tb_logger = TensorBoardLogger(
            save_dir=output_dir,
            name=self.config.Dataset.name,
            version=Path(self._version, f"fold_{fold_n}"),
        )

        return tb_logger

    def train(self, train_loader, valid_loader, fold_n=0):
        self.make_model()

        # ---- training and evaluation ----
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch:02d}-{step}-{valid_loss:.4f}",
            monitor="valid_loss",
            mode="min",
        )
        early_stop_callback = EarlyStopping(
            monitor="valid_loss",
            min_delta=self.config.General.min_delta,
            patience=self.config.General.patience_epochs,
            verbose=False,
            mode="min",
        )

        tb_logger = self._get_logger(fold_n=fold_n)

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

        # Run training
        trainer.fit(self.model, train_loader, valid_loader)

    def _load_or_create_folds(self, kfold, dataset, folds_file):
        if folds_file.exists():
            print("Loading folds from file.")
            with folds_file.open("r") as file:
                folds_data = json.load(file)
        else:
            folds_data = [
                {"train": train_idx.tolist(), "val": val_idx.tolist()}
                for train_idx, val_idx in kfold.split(dataset)
            ]
            with folds_file.open("w") as file:
                json.dump(folds_data, file)

        return [(np.array(fold["train"]), np.array(fold["val"])) for fold in folds_data]

    def _prepare_datasets(self, dataset, train_idx, val_idx):
        train_dataset = GraphDTADataHandler(
            dataset=dataset, target_max_len=self.config.Encoder.Target.sequence_length
        )
        train_dataset.slice_data(train_idx)
        val_dataset = GraphDTADataHandler(
            dataset=dataset, target_max_len=self.config.Encoder.Target.sequence_length
        )
        val_dataset.slice_data(val_idx)

        return train_dataset, val_dataset

    def run_k_fold_validation(self, n_splits=5, start_from_fold=0):
        kfold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.General.random_seed,
        )

        dataset = TDCDataset(
            name=self.config.Dataset.name,
            path=self.config.Dataset.path,
            label_to_log=self.config.Dataset.label_to_log,
            print_stats=True,
            harmonize_affinities=self.config.Dataset.harmonize_affinities,
        )

        folds_file = Path(dataset.path) / f"{dataset.name}_folds.json"
        folds_indices = self._load_or_create_folds(kfold, dataset, folds_file)

        for fold_n, (train_idx, val_idx) in enumerate(folds_indices):
            if fold_n < start_from_fold:
                continue
            train_dataset, val_dataset = self._prepare_datasets(
                dataset, train_idx, val_idx
            )
            train_loader, valid_loader = self._make_data_loaders(
                train_dataset=train_dataset, val_dataset=val_dataset
            )
            self.train(
                train_loader=train_loader, valid_loader=valid_loader, fold_n=fold_n
            )

    def resume_training(self, version: str | int, last_completed_fold: str | int):
        self._version = str(version)
        self.run_k_fold_validation(n_splits=5, start_from_fold=last_completed_fold + 1)
