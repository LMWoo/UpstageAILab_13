import os
from pathlib import Path
from typing import Tuple

import cv2
import albumentations as A
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split

from core.datasets.datasets import ImageDataset, AugraphyImageDataset

def apply_morphology(image, mode="dilate", k_range=(1, 3)):
    k = np.random.randint(k_range[0], k_range[1] + 1)
    kernel = np.ones((k, k), np.uint8)
    if mode == "dilate":
        return cv2.dilate(image, kernel, iterations=1)
    elif mode == "erode":
        return cv2.erode(image, kernel, iterations=1)
    return image

class DatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.data.num_workers
        self.batch_size = cfg.data.batch_size
        self.img_size = cfg.data.img_size
        self.data_path = cfg.data.data_path

        # 1. 일반 이미지 (NoAugraphy)용 강한 augmentation
        aug_no_augraphy = A.Compose([
            # 1) 긴 변을 self.img_size 로 맞추고 비율 유지
            A.SmallestMaxSize(max_size=self.img_size),

            # 2) 다양한 공간/기하 변형
            A.RandomResizedCrop(width=self.img_size, height=self.img_size,
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
                A.Lambda(image=lambda x, **kwargs: apply_morphology(x, mode="dilate", k_range=(1, 3))),
                A.Lambda(image=lambda x, **kwargs: apply_morphology(x, mode="erode",  k_range=(2, 4))),
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

        # 2. Augraphy 이미지 전용 (기하 변형 위주)
        aug_augraphy = A.Compose([
            # 1) 긴 변을 self.img_size 로 맞추고 비율 유지
            A.SmallestMaxSize(max_size=self.img_size),

            # 2) 다양한 공간/기하 변형
            A.RandomResizedCrop(width=self.img_size, height=self.img_size,
                                scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),

            A.RandomRotate90(p=0.5),
            A.Rotate(limit=160, border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.2),

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),

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

        if self.cfg.trainer.use_augraphy:
            self.train_tf = {"Augraphy" : aug_augraphy, "NoAugraphy": aug_no_augraphy}
        else:
            self.train_tf = aug_no_augraphy

        if self.cfg.trainer.use_augraphy:
            self.val_tf = {"Augraphy" : val_tf, "NoAugraphy": val_tf}
        else:
            self.val_tf = val_tf

        self.test_tf = val_tf


        self.train_idx = None
        self.val_idx = None
        self.train_df = None
        self.val_df = None

        if self.cfg.trainer.use_augraphy:
            print("Using Augraphy")
            seed = 2025
            self.dataset_cls = AugraphyImageDataset
            self.full_data_name = f"train_augraphy_{seed}"
            self.full_df = pd.read_csv(os.path.join(self.data_path, self.full_data_name + ".csv"))
        else:
            self.dataset_cls = ImageDataset
            self.full_data_name = "train"
            self.full_df = pd.read_csv(os.path.join(self.data_path, self.full_data_name + ".csv"))

        self.meta_df = pd.read_csv(os.path.join(self.data_path, "meta.csv"))
        self.orig_df = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        self.origin_dataset = self.dataset_cls(self.orig_df, os.path.join(self.data_path, self.full_data_name), None)
    
    def set_split_idx(self, train_idx, val_idx):
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
             # for kfold validation train
            if self.train_idx is not None and self.val_idx is not None:
                # self.train_df = self.full_df.iloc[self.train_idx].reset_index(drop=True)
                # self.val_df = self.full_df.iloc[self.val_idx].reset_index(drop=True)

                # train_df : origin train idx + aug train + idx 
                # val_df : origin val idx 

                train_df_orig = self.orig_df.iloc[self.train_idx].reset_index(drop=True)
                val_df = self.orig_df.iloc[self.val_idx].reset_index(drop=True)

                train_ids = set(train_df_orig["ID"])
                aug_df = self.full_df[
                    self.full_df["ID"].str.startswith("aug_") &
                    self.full_df["ID"].str[4:].isin(train_ids)
                ].reset_index(drop=True)

                self.train_df = pd.concat([train_df_orig, aug_df]).reset_index(drop=True)
                self.val_df = val_df

                assert self.val_df["ID"].str.startswith("aug_").sum() == 0
            else:
                train_idx, val_idx = train_test_split(
                    self.orig_df.index, test_size=0.2, stratify=self.orig_df["target"], random_state=42
                )
                
                # train_df : origin train idx + aug train + idx 
                # val_df : origin val idx 

                train_df_orig = self.orig_df.iloc[train_idx].reset_index(drop=True)
                val_df = self.orig_df.iloc[val_idx].reset_index(drop=True)

                train_ids = set(train_df_orig["ID"])
                aug_df = self.full_df[
                    self.full_df["ID"].str.startswith("aug_") &
                    self.full_df["ID"].str[4:].isin(train_ids)
                ].reset_index(drop=True)

                self.train_df = pd.concat([train_df_orig, aug_df]).reset_index(drop=True)
                self.val_df = val_df

                assert self.val_df["ID"].str.startswith("aug_").sum() == 0
                
                self.set_split_idx(train_idx, val_idx)

            self.train_ds = self.dataset_cls(
                self.train_df, os.path.join(self.data_path, self.full_data_name), transform=self.train_tf
            )
            self.val_ds = self.dataset_cls(
                self.val_df, os.path.join(self.data_path, self.full_data_name), transform=self.val_tf
            )

        if stage in ("test", "predict", None):
            df = pd.read_csv(os.path.join(self.data_path, "sample_submission.csv"))
            self.test_ds = self.dataset_cls(
                df, os.path.join(self.data_path, "test"), transform=self.test_tf, is_test=True
            )

    def set_train_dataset(self, new_df):
        self.train_ds = self.dataset_cls(
            new_df, 
            os.path.join(self.data_path, self.full_data_name),
            transform=self.train_tf,
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