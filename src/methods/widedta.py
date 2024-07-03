import torch
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold

from pathlib import Path
import json

from .configs import ConfigLoader
from data.loading import TDCDataset
from data.preprocessing import MotifFetcher
from data.processing import (
    to_deepsmiles,
    seq_to_words,
    make_words_dict,
    encode_word,
)

from modules.encoders import WideCNN
from modules.decoders import MLP
from data.evaluation import concordance_index


# =============================== Code ==================================
class WideDTA:
    def __init__(self, config_file: str, fast_dev_run=False) -> None:
        self._widedta = _WideDTA(config_file, fast_dev_run)

    # def make_model(self):
    #     self._widedta.make_model(
    #         num_embeddings_drug=num_embeddings_drug,
    #         num_embeddings_target=num_embeddings_target,
    #         num_embeddings_motif=num_embeddings_motif,
    #     )

    def train(self):
        self.make_model()
        self._widedta.train()

    def run_k_fold_validation(self, n_splits=5):
        self._widedta.run_k_fold_validation(n_splits=n_splits)

    def resume_training(self, version, last_completed_fold):
        self._widedta.resume_training(
            version=version, last_completed_fold=last_completed_fold
        )

    def test(self):
        raise NotImplementedError("This will be implemented soon.")


class _WideDTADataHandler(Dataset):
    MAX_DRUG_LEN = 85
    MAX_TARGET_LEN = 1000
    MAX_MOTIF_LEN = 500

    word_to_int = {"Drug": {}, "Target": {}, "Motif": {}}

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.path = dataset.path
        self.dataset_name = dataset.name
        self.data = dataset.data.copy(deep=True)

    def __len__(self):
        return len(self.data)

    def slice_data(self, indices: list):
        self.data = self.data.iloc[indices]
        print("Fold size:", self.data.shape[0])
        self._preprocess_data()

    def _preprocess_data(self):
        try:
            hifens = "-" * 50
            print("Starting Preprocessing.", hifens)
            self._filter_data()
            self._load_motifs()
            self._smiles_to_deep()
            self._convert_seqs_to_words()
            self._load_words_dict()
            self.data.reset_index(drop=True, inplace=True)
            print("Preprocessing finished.", hifens)
        except Exception as e:
            print("Error preprocessing data:", e)
            raise Exception(e)

    def _filter_data(self):
        original_size = self.data.shape[0]
        self.data.drop_duplicates(subset=["Drug", "Target"], inplace=True)
        self.data.fillna("")

        n_rows_out = original_size - self.data.shape[0]

        print(
            f"Rows filtered out: {n_rows_out} ({(n_rows_out * 100)/original_size :.2f}%)"
        )

    def _load_motifs(self):
        mf = MotifFetcher()
        motifs = mf.get_motifs(self.data, self.path, self.dataset_name)

        # Merget dfs and drop NaN
        self.data = pd.merge(self.data, motifs, on=["Target_ID"], how="left").fillna("")

        # Reorder columns
        self.data = self.data[["Drug_ID", "Drug", "Target_ID", "Target", "Motif", "Y"]]

    def _smiles_to_deep(self):
        print("Converting smiles to DeepSmiles.")
        self.data["Drug"] = self.data["Drug"].apply(lambda x: to_deepsmiles(x))

    def _convert_seqs_to_words(self):
        print("Converting sequences to words.")

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
        file_path = Path(self.path, f"{self.dataset_name}_words.json")
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

        drug = encode_word(
            drug,
            word_to_int=self.word_to_int["Drug"],
            length=self.MAX_DRUG_LEN,
        )
        drug = torch.LongTensor(drug)

        target = encode_word(
            target,
            word_to_int=self.word_to_int["Target"],
            length=self.MAX_TARGET_LEN,
        )
        target = torch.LongTensor(target)

        motif = encode_word(
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

        super(_WideDTATrainer, self).__init__(**kwargs)
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
            self.log("train_ci", ci, on_epoch=True, on_step=True, prog_bar=True)
            self.logger.log_metrics({"train_step_ci": ci}, self.global_step)
        self.log("train_loss", ci, on_epoch=True, on_step=True, prog_bar=True)
        self.logger.log_metrics({"train_step_loss": loss}, self.global_step)

        return loss

    def validation_step(self, valid_batch):
        x_drug, x_target, x_motif, y = valid_batch
        y_pred = self(x_drug, x_target, x_motif)
        loss = F.mse_loss(y_pred, y.view(-1, 1))

        if self.ci_metric:
            ci = concordance_index(y, y_pred)
            self.log("valid_ci", ci, on_epoch=True, on_step=False, prog_bar=True)
            self.logger.log_metrics({"valid_step_ci": ci}, self.global_step)
        self.log("valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.logger.log_metrics({"valid_step_loss": loss}, self.global_step)

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
        self._seed_random()
        self.fast_dev_run = fast_dev_run

    def _load_configs(self, config_path: str):
        cl = ConfigLoader()
        cl.load_config(config_path=config_path)

        self.config = cl

    def _seed_random(self):
        pl.seed_everything(seed=self.config.General.random_seed, workers=True)

    def load_data(self):
        self.data = _WideDTADataHandler(
            dataset_name=self.config.Dataset.name,
            path=self.config.Dataset.path,
            label_to_log=self.config.Dataset.label_to_log,
        )

    def _make_model(
        self, num_embeddings_drug, num_embeddings_target, num_embeddings_motif
    ):
        drug_encoder = WideCNN(
            num_embeddings=num_embeddings_drug,
            sequence_length=self.config.Encoder.Drug.sequence_length,
            num_filters=self.config.Encoder.num_filters,
            kernel_size=self.config.Encoder.kernel_size,
        )

        target_encoder = WideCNN(
            num_embeddings=num_embeddings_target,
            sequence_length=self.config.Encoder.Target.sequence_length,
            num_filters=self.config.Encoder.num_filters,
            kernel_size=self.config.Encoder.kernel_size,
        )

        motif_encoder = WideCNN(
            num_embeddings=num_embeddings_motif,
            sequence_length=self.config.Encoder.Motif.sequence_length,
            num_filters=self.config.Encoder.num_filters,
            kernel_size=self.config.Encoder.kernel_size,
        )

        decoder = MLP(
            in_dim=self.config.Decoder.in_dim,
            hidden_dim=self.config.Decoder.hidden_dim,
            out_dim=self.config.Decoder.out_dim,
            dropout_rate=self.config.Decoder.dropout_rate,
            num_fc_layers=self.config.Decoder.num_fc_layers,
        )

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

    def _make_data_loaders(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.Trainer.train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.Trainer.train_batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    def _get_logger(self, fold_n):
        output_dir = Path("outputs", "WideDTA")
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
        # ---- set model ----
        num_embeddings_drug = len(train_loader.dataset.word_to_int["Drug"])
        num_embeddings_target = len(train_loader.dataset.word_to_int["Target"])
        num_embeddings_motif = len(train_loader.dataset.word_to_int["Motif"])

        self._make_model(
            num_embeddings_drug=num_embeddings_drug,
            num_embeddings_target=num_embeddings_target,
            num_embeddings_motif=num_embeddings_motif,
        )

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

        # Logger
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

        trainer.fit(self.model, train_loader, valid_loader)
        print(checkpoint_callback.best_model_path)

    # ==========================================================================
    # def run_k_fold_validation(self, n_splits=5):
    #     kfold = KFold(n_splits=n_splits, shuffle=True)

    #     dataset = TDCDataset(
    #         name=self.config.Dataset.name,
    #         path=self.config.Dataset.path,
    #         label_to_log=self.config.Dataset.label_to_log,
    #         print_stats=True,
    #         harmonize_affinities=self.config.Dataset.harmonize_affinities,
    #     )

    #     for fold_n, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    #         train_dataset = _WideDTADataHandler(dataset=dataset)
    #         train_dataset.slice_data(train_idx)

    #         val_dataset = _WideDTADataHandler(dataset=dataset)
    #         val_dataset.slice_data(val_idx)

    #         train_loader, valid_loader = self._make_data_loaders(
    #             train_dataset=train_dataset, val_dataset=val_dataset
    #         )

    #         self.train(
    #             train_loader=train_loader, valid_loader=valid_loader, fold_n=fold_n
    #         )

    # def run_k_fold_validation(self, n_splits=5):
    #     kfold = KFold(n_splits=n_splits, shuffle=True)

    #     dataset = TDCDataset(
    #         name=self.config.Dataset.name,
    #         path=self.config.Dataset.path,
    #         label_to_log=self.config.Dataset.label_to_log,
    #         print_stats=True,
    #         harmonize_affinities=self.config.Dataset.harmonize_affinities,
    #     )

    #     folds_file = Path(dataset.path) / "folds.json"

    #     if folds_file.exists():
    #         with open(folds_file, "r") as file:
    #             folds_data = json.load(file)
    #             folds_indices = [
    #                 (np.array(fold["train"]), np.array(fold["val"]))
    #                 for fold in folds_data
    #             ]
    #     else:
    #         folds_indices = list(kfold.split(dataset))
    #         folds_data = [
    #             {"train": train_idx.tolist(), "val": val_idx.tolist()}
    #             for train_idx, val_idx in folds_indices
    #         ]
    #         with open(folds_file, "w") as file:
    #             json.dump(folds_data, file)

    #     for fold_n, (train_idx, val_idx) in enumerate(folds_indices):
    #         train_dataset = _WideDTADataHandler(dataset=dataset)
    #         train_dataset.slice_data(train_idx)

    #         val_dataset = _WideDTADataHandler(dataset=dataset)
    #         val_dataset.slice_data(val_idx)

    #         train_loader, valid_loader = self._make_data_loaders(
    #             train_dataset=train_dataset, val_dataset=val_dataset
    #         )

    #         self.train(
    #             train_loader=train_loader, valid_loader=valid_loader, fold_n=fold_n
    #         )

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
        train_dataset = _WideDTADataHandler(dataset=dataset)
        train_dataset.slice_data(train_idx)
        val_dataset = _WideDTADataHandler(dataset=dataset)
        val_dataset.slice_data(val_idx)

        return train_dataset, val_dataset

    def run_k_fold_validation(self, n_splits=5, start_from_fold=0):
        kfold = KFold(n_splits=n_splits, shuffle=True)
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
