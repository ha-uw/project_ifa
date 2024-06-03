import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from data.evaluation import concordance_index


# Trainer ----------------------------------------------------------------------
class BaseDTATrainer(pl.LightningModule):
    """ """

    def __init__(
        self, drug_encoder, target_encoder, decoder, lr=0.001, ci_metric=False, **kwargs
    ):
        super(BaseDTATrainer, self).__init__()
        # self.save_hyperparameters()
        self.lr = lr
        self.ci_metric = ci_metric
        self.drug_encoder = drug_encoder
        self.target_encoder = target_encoder
        self.decoder = decoder

    def configure_optimizers(self):
        """
        Config adam as default optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x_drug, x_target):
        """
        Same as :meth:`torch.nn.Module.forward()`
        """
        raise NotImplementedError("Forward pass needs to be defined.")

    def training_step(self, train_batch, batch_idx):
        """
        Compute and return the training loss on one step
        """
        x_drug, x_target, y = train_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        if self.ci_metric:
            ci = concordance_index(y, y_pred)
            self.logger.log_metrics({"train_step_ci": ci}, self.global_step)
        self.logger.log_metrics({"train_step_loss": loss}, self.global_step)
        self.log("train_step_loss_2", loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        """
        Compute and return the validation loss on one step
        """
        x_drug, x_target, y = valid_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Compute and return the test loss on one step
        """
        x_drug, x_target, y = test_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        if self.ci_metric:
            ci = concordance_index(y, y_pred)
            self.log("test_ci", ci, on_epoch=True, on_step=False)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss
