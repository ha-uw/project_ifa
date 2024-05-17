import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from src.data.tdc_datasets import BindingDBDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import set_float32_matmul_precision


from src.models.GraphDTA import GAT_GCN, GATNet, GCNNet, GINConvNet
from src.models.deepDTA import CNNEncoder, MLPDecoder
from src.models.GraphDTA_trainer import GraphDTATrainer


# ============================== Configs ================================
set_float32_matmul_precision("medium")

dataset_path = "./data"
dataset_name = "DAVIS"

ci_metric = False

# ------------ sorver --------------
seed = 42

train_batch_size = 512
test_batch_size = 512

min_epochs = 0
max_epochs = 1000

lr = 0.0005

# Model parameters
NUM_FEATURES_XD = 78
NUM_FEATURES_XT = 25
SEQUENCE_LENGTH_XT = 1000
NUM_FILTERS = 32
EMBED_DIM = 128
OUTPUT_DIM = 128
DROPOUT = 0.2
NUM_HEADS = 10
KERNEL_SIZE = [8, 8, 8]

# Decoder
HIDDEN_DIM = 1024
IN_DIM = 256
OUT_DIM = 1

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
            out_dim=OUT_DIM,
        ),
        lr=lr,
        ci_metric=ci_metric,
    )

    return model


print(get_model())


def main():
    tb_logger = TensorBoardLogger(
        "outputs",
        name=dataset_name + "_GraphDTA",
    )

    pl.seed_everything(seed=seed, workers=True)

    # ---- set dataset ----
    train_dataset = BindingDBDataset(
        name=dataset_name, split="train", path=dataset_path
    )
    valid_dataset = BindingDBDataset(
        name=dataset_name, split="valid", path=dataset_path
    )
    test_dataset = BindingDBDataset(name=dataset_name, split="test", path=dataset_path)

    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=train_batch_size
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, shuffle=True, batch_size=test_batch_size
    )
    test_loader = DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=test_batch_size
    )

    # ---- set model ----
    model = get_model()

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
