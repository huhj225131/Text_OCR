import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from models_lt import BaseLitModel
from models import MLP
from datasets import EMNIST
from data_modules import EMNISTDataModule
import argparse

parser = argparse.ArgumentParser()
parser = EMNIST.add_to_argparse(parser)
parser = EMNISTDataModule.add_to_argparse(parser)
parser = MLP.add_to_argparse(parser)
parser = BaseLitModel.add_to_argparse(parser)
args = parser.parse_args()

data_module = EMNISTDataModule(dataset=EMNIST, args=args)
model = MLP(data_config=data_module.data_config,args=args)
lit_model = BaseLitModel(model=model, args = args)

wandb_logger = WandbLogger(project="Simple_character_ocr",
                           entity="hungluu_test"
                           ,log_model=True)

trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
trainer.fit(lit_model, datamodule= data_module)
trainer.test(datamodule=data_module, ckpt_path="best")

