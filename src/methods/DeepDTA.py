import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from data.loading import TDCDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import set_float32_matmul_precision


from modules.encoders import CNNEncoder, MLPDecoder
from modules.trainers import DeepDTATrainer


# ============================== Configs ================================
set_float32_matmul_precision("medium")

dataset_path = "./data"
dataset_name = "KIBA"

# ---- encoder hyper-parameter ----
drug_dim = 128  # cfg.MODEL.DRUG_DIM
target_dim = 128  # cfg.MODEL.TARGET_DIM

# vocab size
num_drug_embeddings = 64  # cfg.MODEL.NUM_SMILE_CHAR
num_target_embeddings = 25  # cfg.MODEL.NUM_ATOM_CHAR

# input length
drug_length = 85  # cfg.MODEL.DRUG_LENGTH
target_length = 1200  # cfg.MODEL.TARGET_LENGTH

num_filters = 32  # cfg.MODEL.NUM_FILTERS

drug_filter_length = [4, 6, 8]  # cfg.MODEL.DRUG_FILTER_LENGTH

target_filter_length = [4, 8, 12]  # cfg.MODEL.TARGET_FILTER_LENGTH

ci_metric = True

output_dim = 96  # num_filters * 3

# ---- decoder hyper-parameter ----
decoder_in_dim = 192  # cfg.MODEL.MLP_IN_DIM
decoder_hidden_dim = 1024  # cfg.MODEL.MLP_HIDDEN_DIM
decoder_out_dim = 512  # cfg.MODEL.MLP_OUT_DIM
dropout_rate = 0.1  # cfg.MODEL.MLP_DROPOUT_RATE

# ------------ sorver --------------
seed = 42  # _C.SOLVER.SEED

train_batch_size = 256  # _C.SOLVER.TRAIN_BATCH_SIZE
test_batch_size = 256  # _C.SOLVER.TEST_BATCH_SIZE

min_epochs = 0  # _C.SOLVER.MIN_EPOCHS
max_epochs = 100  # _C.SOLVER.MAX_EPOCHS

lr = 0.001  # cfg.SOLVER.LR

# ------------ early stop ------------------
early_stop = False
min_delta = 0.001
patience = 20

fast_dev_run = False

# =============================== Code ==================================


def get_model():

    drug_encoder = CNNEncoder(
        num_embeddings=num_drug_embeddings,
        embedding_dim=drug_dim,
        sequence_length=drug_length,
        num_kernels=num_filters,
        kernel_length=drug_filter_length,
    )

    target_encoder = CNNEncoder(
        num_embeddings=num_target_embeddings,
        embedding_dim=target_dim,
        sequence_length=target_length,
        num_kernels=num_filters,
        kernel_length=target_filter_length,
    )

    decoder = MLPDecoder(
        in_dim=decoder_in_dim,
        hidden_dim=decoder_hidden_dim,
        out_dim=decoder_out_dim,
        dropout_rate=dropout_rate,
        include_decoder_layers=True,
    )

    model = DeepDTATrainer(
        drug_encoder=drug_encoder,
        target_encoder=target_encoder,
        decoder=decoder,
        lr=lr,
        ci_metric=ci_metric,
    )

    return model


print(get_model())


def main():
    tb_logger = TensorBoardLogger(
        "outputs",
        name=dataset_name,
    )

    pl.seed_everything(seed=seed, workers=True)

    # ---- set dataset ----
    train_dataset = TDCDataset(name=dataset_name, split="train", path=dataset_path)
    valid_dataset = TDCDataset(name=dataset_name, split="valid", path=dataset_path)
    test_dataset = TDCDataset(name=dataset_name, split="test", path=dataset_path)

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
