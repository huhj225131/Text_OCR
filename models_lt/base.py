import argparse

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100

class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        

        self.data_config = self.model.data_config
        self.mapping = self.data_config["mapping"]
        self.input_dims = self.data_config["input_dims"]

        self.optimizer = self.hparams.get("optimizer", OPTIMIZER)

        self.lr = self.hparams.get("lr", LR)

        loss = self.hparams.get("loss", LOSS)

        if loss not in ("transformer",):
            self.loss_fn = getattr(torch.nn.functional, loss)
        self.one_cycle_max_lr = self.hparams.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.hparams.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy(task="multiclass", num_classes=len(self.mapping))
        self.val_acc = Accuracy(task="multiclass", num_classes=len(self.mapping))
        self.test_acc = Accuracy(task="multiclass", num_classes=len(self.mapping))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_acc(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.optimizer)
        return optimizer_cls(self.parameters(), lr=self.lr)
