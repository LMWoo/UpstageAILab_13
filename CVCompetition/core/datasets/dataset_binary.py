import os
from pathlib import Path
from typing import Tuple

import cv2
import albumentations as A
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotate,
    ColorJitter, RandomBrightnessContrast, CLAHE,
    GaussianBlur, CoarseDropout, Resize, Normalize
)
from PIL import Image
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split

from core.datasets.datasets import ImageDataset, AugraphyImageDataset

class BinaryImageDataset(Dataset):
    def __init__(self, data, path, transform=None, is_test=False):
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        self.df = df

        self.path = path
        self.transform = transform
        self.is_test = is_test

        self.samples = []
        if is_test:
            pass
        else:
            for name, target in self.df.values:
                self.samples.append((name, target))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["ID"]
        img_path = os.path.join(self.path, name)
        img = np.array(Image.open(img_path))

        if self.transform:
            img = self.transform(image=img)["image"]
        
        if self.is_test:
            return img, name
        else:
            label = int(row["target"])
            return img, label

class BinaryDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.data.num_workers
        self.batch_size = cfg.data.batch_size
        self.img_size = cfg.data.img_size
        self.data_path = cfg.data.data_path

        
        train_tf = A.Compose([
            # 1) 긴 변을 self.img_size 로 맞추고 비율 유지
            A.SmallestMaxSize(max_size=self.img_size),

            # 2) 다양한 공간/기하 변형
            A.RandomResizedCrop(size=(self.img_size, self.img_size),
                                scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                            rotate_limit=30, border_mode=cv2.BORDER_CONSTANT,
                            value=(255, 255, 255), p=0.7),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=160, border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.2),

            # 3) 모폴로지 증강
            A.OneOf([
                A.Morphological(scale=(1, 3), operation="dilation", p=1.0),
                A.Morphological(scale=(2, 4), operation="erosion", p=1.0),
            ], p=0.3),

            # 4) Blur / Noise
            A.OneOf([
                A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1.0),
                A.Blur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # 여긴 여전히 작동함

            # 5) 컬러/명암 증강
            A.ColorJitter(brightness=0.1, contrast=0.07,
                        saturation=0.07, hue=0.07, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),

            A.ImageCompression(quality_lower=30, quality_upper=60, p=0.3),
            
            # 6) 리사이즈 → Normalize → Tensor
            A.Resize(self.img_size, self.img_size),  # always_apply 제거
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # 3. 검증 / 테스트: 깨끗한 리사이즈 + 정규화만
        val_tf = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = val_tf

        self.train_idx = None
        self.val_idx = None
        self.train_df = None
        self.val_df = None

        self.dataset_cls = BinaryImageDataset

        self.full_data_name = "train"

        binary_df_path = os.path.join(self.data_path, "train_binary.csv")

        if not os.path.exists(binary_df_path):
            print(f"File not found: {binary_df_path}")
            raise FileNotFoundError(f"train_binary.csv does not exist in {self.data_path}. Exiting.")
        
        print(self.data_path)
        self.binary_df = pd.read_csv(os.path.join(self.data_path, "train_binary.csv"))
    
    def set_split_idx(self, train_idx, val_idx):
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            train_idx, val_idx = train_test_split(
                self.binary_df.index,
                test_size=0.2,
                stratify=self.binary_df["target"],
                random_state=42,
            )

            self.train_ds = self.dataset_cls(
                self.binary_df.iloc[train_idx].reset_index(drop=True),
                os.path.join(self.data_path, self.full_data_name),
                transform=self.train_tf

            )
            self.val_ds = self.dataset_cls(
                self.binary_df.iloc[val_idx].reset_index(drop=True),
                os.path.join(self.data_path, self.full_data_name),
                transform=self.val_tf
            )

            print("Train length:", len(self.train_ds))
            print("Val length:", len(self.val_ds))

        if stage in ("test", "predict", None):
            df = pd.read_csv(os.path.join(self.data_path, "sample_submission.csv"))

            if hasattr(self.cfg.data, "submission_ids"):
                df = df[df["ID"].isin(self.cfg.data.submission_ids)].reset_index(drop=True)

            self.test_ds = self.dataset_cls(
                df, os.path.join(self.data_path, "test"), transform=self.test_tf, is_test=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )