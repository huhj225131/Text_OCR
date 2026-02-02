import argparse
from pathlib import Path
from typing import Any, Dict
import torch, torchvision
from torch.utils.data import Dataset,ConcatDataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import io
from PIL import Image


class IAMLineDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.dataset = pd.read_parquet(parquet_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_bytes= self.dataset.iloc[index]["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes))
        if self.transform:
            image = self.transform(img)
        label = self.dataset.iloc[index]["text"]
        return image, label
    
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_RATIO = 0.8

class IMA_Line_Dataset:
    def __init__(self, args: argparse.Namespace = None):
        self.args  = vars(args) if args is not None else {}
        self.data_dir = Path(self.args.get("data_dir", DATA_DIR))
        self.train_ratio = self.args.get("train_ratio", TRAIN_RATIO)

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((480, 960)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load IAM-line datasets
        line_dir = self.data_dir / "IAM-line" / "data"
        line_datasets = []
        for parquet_file in  sorted(line_dir.glob("*.parquet")):
            line_datasets.append(IAMLineDataset(parquet_file, transform=transform))
        
        # Load IAM-sentences datasets 
        sentence_datasets = []
        sentence_dir = self.data_dir / "IAM-sentences" / "data"
        for parquet_file in sorted(sentence_dir.glob("train-*.parquet")):
            sentence_datasets.append(IAMLineDataset(parquet_file, transform=transform))
        
        # Gộp TẤT CẢ datasets lại
        all_datasets = line_datasets + sentence_datasets
        full_dataset = ConcatDataset(all_datasets)
        
        # Chia train/val/test
        indices = np.arange(len(full_dataset))
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size=(1 - self.train_ratio), 
            random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=0.5, 
            random_state=42
        )
        
        self.trainset = Subset(full_dataset, train_indices)
        self.valset = Subset(full_dataset, val_indices)
        self.testset = Subset(full_dataset, test_indices)
        
        # Data config
        sample_img, _ = self.trainset[0]
        self.data_config = {
            "input_dims": tuple(sample_img.shape),
        }
    

    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--data_dir", type=str, default=DATA_DIR)
        parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
        return parser

        

