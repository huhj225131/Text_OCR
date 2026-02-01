import  pytorch_lightning as L
import argparse
from torch.utils.data import DataLoader

BATCH_SIZE = 32
NUM_WORKERS = 2

class EMNISTDataModule(L.LightningDataModule):
    def __init__(self, dataset, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size  = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.dataset = dataset(args)
        self.data_config = self.dataset.data_config

    def train_dataloader(self):
        return DataLoader(self.dataset.trainset,
                        batch_size=self.batch_size,
                        shuffle=True, 
                        num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
        return parser

    

    