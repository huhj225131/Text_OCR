import argparse
from pathlib import Path
from typing import Any, Dict
import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np

DOWNLOADED_DATA_DIRNAME = Path(__file__).resolve().parents[1] / "data"
TRAIN_RATIO = 0.8
class EMNIST: 

    def __init__(self, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
        self.save_dir = self.args.get("save_dir", DOWNLOADED_DATA_DIRNAME)
        self.train_ratio = self.args.get("train_ratio", TRAIN_RATIO)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.EMNIST(root=self.save_dir, split='byclass', train=True,
                                       download=True, transform=transform)
        targets = dataset.targets.numpy()
        indices = np.arange(len(dataset))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=(1 - self.train_ratio),
            stratify=targets,
            random_state=30
        )
        self.trainset = Subset(dataset, train_indices)
        self.valset = Subset(dataset, val_indices)
       

        
        self.testset = torchvision.datasets.EMNIST(root= self.save_dir, split='byclass', train=False,
                                            download=True, transform=transform)
        
        sample, _ = self.trainset[0]
        original_dataset = self.trainset.dataset
        self.data_config = {
        "input_dims": tuple(sample.shape),  
        "mapping": list(range(len(original_dataset.classes))), 
        "output_dims": (len(original_dataset.classes),)
    }

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--save_dir", type=str, default= DOWNLOADED_DATA_DIRNAME)
        parser.add_argument("--train_ratio", type=float, default = TRAIN_RATIO)
        return parser