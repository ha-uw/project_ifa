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
def load_configs(config_path: str):
    cl = ConfigLoader()
    cl.load_config(config_path=config_path)

    return cl


def get_model(config):
    drug_encoder = CNN(
        num_embeddings=config.Encoder.Drug.num_embeddings,
        embedding_dim=config.Encoder.Drug.embedding_dim,
        sequence_length=config.Encoder.Drug.sequence_length,
        num_kernels=config.Encoder.Drug.num_filters,
        kernel_length=config.Encoder.Drug.filter_length,
    )

    target_encoder = CNN(
        num_embeddings=config.Encoder.Target.num_embeddings,
        embedding_dim=config.Encoder.Target.embedding_dim,
        sequence_length=config.Encoder.Target.sequence_length,
        num_kernels=config.Encoder.Target.num_filters,
        kernel_length=config.Encoder.Target.filter_length,
    )

    decoder = MLP(
        in_dim=config.Decoder.in_dim,
        hidden_dim=config.Decoder.hidden_dim,
        out_dim=config.Decoder.out_dim,
        dropout_rate=config.Decoder.dropout_rate,
        include_decoder_layers=config.Decoder.include_decoder_layers,
    )

    model = DeepDTATrainer(
        drug_encoder=drug_encoder,
        target_encoder=target_encoder,
        decoder=decoder,
        lr=config.Trainer.learning_rate,
        ci_metric=config.Trainer.ci_metric,
    )

    return model


def main(config_file):
    config = load_configs(config_file)

    tb_logger = TensorBoardLogger(
        "outputs",
        name=config.Dataset.name,
    )

    pl.seed_everything(seed=config.General.random_seed, workers=True)

    # ---- set dataset ----
    train_dataset = TDCDataset(
        name=config.Dataset.name, split="train", path=config.Dataset.path
    )
    valid_dataset = TDCDataset(
        name=config.Dataset.name, split="valid", path=config.Dataset.path
    )
    test_dataset = TDCDataset(
        name=config.Dataset.name, split="test", path=config.Dataset.path
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=config.Trainer.train_batch_size,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=True,
        batch_size=config.Trainer.test_batch_size,
    )
    test_loader = DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=config.Trainer.test_batch_size
    )

    # ---- set model ----
    model = get_model(config)
    print(model)

    # ---- training and evaluation ----
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{valid_loss:.4f}", monitor="valid_loss", mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=config.General.min_delta,
        patience=config.General.patience_epochs,
        verbose=False,
        mode="min",
    )

    callbacks = [checkpoint_callback]

    if config.General.early_stop:
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        max_epochs=config.Trainer.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=tb_logger,
        callbacks=callbacks,
        fast_dev_run=config.General.fast_dev_run,
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
