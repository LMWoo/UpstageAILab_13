import os
from pathlib import Path
from typing import Tuple, Dict

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from albumentations.pytorch import ToTensorV2
from PIL import Image
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F  
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.distributions.beta import Beta
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
import torchmetrics.classification
from transformers import get_cosine_schedule_with_warmup
import wandb
import timm
from timm.data.mixup import Mixup

from core.models.convnext import BinaryConvNeXt, ConvNeXt
from core.models.resnet50 import Resnet50
from core.models.resnet18 import BinaryResnet18, Resnet18
from core.models.vit import BinaryViT, ViT
from core.models.swinTransformer import SwinTransformer
from core.models.efficientnet import EfficientNet
from core.models.convnextArcFace import ConvNeXtArcFace
from core.losses.focalloss import FocalLoss
from core.losses.softtarget_focalloss import SoftTargetFocalLoss
from core.callbacks.ema import EMA

class BinaryTrainerModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if "resnet18" in cfg.model.model.model_name:
            if cfg.model.model.is_binary:
                self.model = BinaryResnet18(cfg)
            else:
                self.model = Resnet18(cfg)
        elif "convnext" in cfg.model.model.model_name:
            if cfg.model.model.is_binary:
                self.model = BinaryConvNeXt(cfg)
            else:
                self.model = ConvNeXt(cfg)
        elif "vit" in cfg.model.model.model_name or "deit" in cfg.model.model.model_name:
            if cfg.model.model.is_binary:
                self.model = BinaryViT(cfg)
            else:
                self.model = ViT(cfg)

        if cfg.loss.loss_name == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss.loss_name == "bceLoss":
            self.criterion = nn.BCEWithLogitsLoss()
        
        if cfg.model.model.is_binary:
            self.criterion = nn.BCEWithLogitsLoss()

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc   = torchmetrics.classification.BinaryAccuracy()
        self.train_f1  = torchmetrics.classification.BinaryF1Score()
        self.val_f1    = torchmetrics.classification.BinaryF1Score()

        self.ema = None
    
    @staticmethod
    def _map_to_binary(y: torch.Tensor) -> torch.Tensor:
        """3 → 0,   7 → 1  (값이 이미 0/1이면 그대로)"""
        if y.ndim == 2:
            y = y.argmax(1)
        return (y == 7).float() if y.max() > 1 else y.float() # for bce loss
    
    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        if self.cfg.trainer.use_ema == True:
            self.ema = EMA(self.model, decay=0.995)
            if hasattr(self.ema, "ema_model"):
                self.ema.ema_model.to(self.device)
                self.ema.ema_model.eval()

    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_target = self._map_to_binary(y)

        logits = self(x)
        loss   = self.criterion(logits, y_target)

        prob  = torch.sigmoid(logits)
        preds = (prob > 0.5).long()
        y_int = y_target.long()  
        
        if stage == "train":
            self.train_acc.update(preds, y_int)
            self.train_f1.update(preds, y_int)
        else:
            self.val_acc.update(preds, y_int)
            self.val_f1.update(preds, y_int)
        
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), on_step=False, on_epoch=True)
        return {"logits": logits, "targets": y_target, "loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema.state_dict())
            out = self._shared_step(batch, "val")
            self.model.load_state_dict(backup_state)
            return out
        return self._shared_step(batch, "val")
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            logits = self.ema.ema_model(x)
        else:
            logits = self(x)
        
        prob = torch.sigmoid(logits)
        preds = (prob > 0.5).long()
        return preds
    
    def on_train_epoch_end(self):
        if self.current_epoch == self.cfg.model.model.mid_freeze_epochs:
            print(f"Epoch {self.current_epoch+1}: Start Feature Extractor mid freeze and stages.3, stages.2 model fine-tuning")
            self.model.unfreeze_mid()

        if self.current_epoch == self.cfg.model.model.all_freeze_epochs:
            print(f"Epoch {self.current_epoch+1}: Start Feature Extractor mid freeze and Full model fine-tuning")
            self.model.unfreeze_all()

        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def _log_epoch_metrics(self, stage: str):
        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric = getattr(self, f"{stage}_f1")
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)

        wandb.log({f"Accuracy/{stage}": acc})
        wandb.log({f"F1score/{stage}": f1})
        
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        wandb.log({f"LR/{stage}": current_lr})

        acc_metric.reset()
        f1_metric.reset()

    def on_save_checkpoint(self, checkpoint):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("ema_state") and self.cfg.trainer.use_ema == True:
            decay = float(getattr(self.cfg.trainer, "ema_decay", 0.995))
            self.ema = EMA(self.model, decay=decay)
            self.ema.ema_model.load_state_dict(checkpoint["ema_state"])
            self.ema.ema_model.eval()

    def configure_optimizers(self):
        optimizer_name = str(self.cfg.optimizer._target_)
        print(f"=========== {optimizer_name} ==============")
        if "AdamW" in optimizer_name:
            print("======== AdamW =========")
            opt = AdamW(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

            scheduler = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=self.cfg.scheduler.warmup_steps,
                num_training_steps=self.cfg.scheduler.total_steps
            )
            return {
                "optimizer":   opt,
                "lr_scheduler": {
                    "scheduler":  scheduler,
                    "interval":   "step",   # ← 매 step마다 step()
                    "frequency":  1,
                    "name":       "cosine_warmup",
                },
            }
        else:
            opt = Adam(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

        return opt