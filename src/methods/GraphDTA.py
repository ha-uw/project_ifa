import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data.lightning import LightningDataset
from data.loading import TDCDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import set_float32_matmul_precision


from modules.decoders import GAT_GCN, GATNet, GCNNet, GINConvNet
from modules.encoders import CNNEncoder, MLPDecoder
from modules.trainers import GraphDTATrainer


# ============================== Configs ================================
set_float32_matmul_precision("medium")

dataset_path = "./data"
dataset_name = "DAVIS"

ci_metric = False

# ------------ sorver --------------
seed = 42

BATCH_SIZE = 512

min_epochs = 0
max_epochs = 1000

lr = 0.0005

# Model parameters
NUM_FEATURES_XD = 78
NUM_FEATURES_XT = 25
SEQUENCE_LENGTH_XT = 1200
NUM_FILTERS = 32
EMBED_DIM = 128
OUTPUT_DIM = 128
DROPOUT = 0.2
NUM_HEADS = 10
KERNEL_SIZE = [8, 8, 8]

# Decoder
HIDDEN_DIM = 1024
IN_DIM = 224
DECOD_OUT_DIM = 1

# ------------ early stop ------------------
early_stop = False
min_delta = 0.001
patience = 20

fast_dev_run = False

# =============================== Code ==================================


def get_model():
    model = GraphDTATrainer(
        drug_encoder=GAT_GCN(),
        target_encoder=CNNEncoder(
            embedding_dim=EMBED_DIM,
            num_embeddings=NUM_FEATURES_XT,
            kernel_length=KERNEL_SIZE,
            num_kernels=NUM_FILTERS,
            sequence_length=SEQUENCE_LENGTH_XT,
        ),
        decoder=MLPDecoder(
            dropout_rate=DROPOUT,
            hidden_dim=HIDDEN_DIM,
            in_dim=IN_DIM,
            include_decoder_layers=False,
            out_dim=DECOD_OUT_DIM,
        ),
        lr=lr,
        ci_metric=ci_metric,
    )

    return model


def main():
    tb_logger = TensorBoardLogger(
        "outputs",
        name=dataset_name + "_GraphDTA",
    )

    pl.seed_everything(seed=seed, workers=True)

    # ---- set dataset ----
    train_dataset = TDCDataset(
        name=dataset_name, split="train", path=dataset_path, mode_drug="gcn"
    )
    valid_dataset = TDCDataset(
        name=dataset_name, split="valid", path=dataset_path, mode_drug="gcn"
    )
    test_dataset = TDCDataset(
        name=dataset_name, split="test", path=dataset_path, mode_drug="gcn"
    )

    pl_dataset = LightningDataset(
        train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE
    )

    train_loader = pl_dataset.train_dataloader()
    valid_loader = pl_dataset.val_dataloader()
    test_loader = pl_dataset.test_dataloader()

    # ---- set model ----
    model = get_model()
    print(model)

    # ---- training and evaluation ----
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{valid_loss:.4f}", monitor="valid_loss", mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=min_delta,
        patience=patience,
        verbose=False,
        mode="min",
    )

    callbacks = [checkpoint_callback]

    if early_stop:
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=tb_logger,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    main()
