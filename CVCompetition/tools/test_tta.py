import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from albumentations.pytorch import ToTensorV2
import albumentations as A

from core.datasets.dataset import DatasetModule
from core.trainer.trainer import TrainerModule
from core.trainer.HNMTrainer import HardNegativeMiningTrainerModule


def _find_latest_ckpt() -> str | None:
    from pathlib import Path
    ckpts = list(Path(".").rglob("lightning_logs/pred_mid_unfreeze/checkpoints/best-*.ckpt"))
    return str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]) if ckpts else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.get("seed", 42), workers=True)

    ckpt_path: str | None = cfg.get("test", {}).get("ckpt_path", None)  # hydra override 가능
    ckpt_path = ckpt_path or _find_latest_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint (.ckpt) found. Train the model first or specify test.ckpt_path=<path>.")

    print(f"Using checkpoint: {ckpt_path}")

    dm = DatasetModule(cfg)
    if cfg.trainer.hnm.use_hnm == True:
        model = HardNegativeMiningTrainerModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    else:
        model = TrainerModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()

    trainer = Trainer(accelerator="auto", devices="auto", precision="bf16-mixed" if cfg.get("bf16", False) else 32)
    # preds = trainer.predict(model, datamodule=dm)    
       
    # preds = torch.cat(preds, dim=0)                     
    # preds = torch.argmax(preds, dim=1).cpu().numpy()    

    resize_norm_tf = A.Compose([
        A.Resize(cfg.data.img_size, cfg.data.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    tta_tf_list = [
        # 1) 원본
        A.Compose(resize_norm_tf.transforms),

        # 2) 좌우 플립
        A.Compose([A.HorizontalFlip(p=1.0)] + resize_norm_tf.transforms),

        # 3) 상하 플립
        A.Compose([A.VerticalFlip(p=1.0)] + resize_norm_tf.transforms),

        # 4) 90° CW
        A.Compose([A.Affine(rotate=90, p=1.0)] + resize_norm_tf.transforms),

        # 5) 90° CCW
        A.Compose([A.Affine(rotate=-90, p=1.0)] + resize_norm_tf.transforms),

        # 6) 175° CW
        A.Compose([A.Affine(rotate=175, p=1.0)] + resize_norm_tf.transforms),

        # 7) 175° CCW
        A.Compose([A.Affine(rotate=-175, p=1.0)] + resize_norm_tf.transforms),
    ]

    probs_all = []           # T × N × C  (T: TTA 회수, N: 이미지 수, C: 클래스 수)

    for tta_tf in tta_tf_list:
        dm.test_tf = tta_tf          # ★ 변환 교체
        preds = trainer.predict(model, datamodule=dm)   # list[Tensor]
        preds = torch.cat(preds, dim=0)                 # (N, C) – 이미 softmax
        probs_all.append(preds)


    # (T, N, C) → 평균 → (N, C)
    probs_mean = torch.stack(probs_all, dim=0).mean(dim=0)
    final_preds = probs_mean.argmax(dim=1).cpu().numpy()

    submission = pd.read_csv(os.path.join(cfg.data.data_path, "sample_submission.csv"))
    submission["target"] = final_preds
    submission.to_csv("pred.csv", index=False)


if __name__ == "__main__":
    main()
