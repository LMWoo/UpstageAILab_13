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

from core.datasets.dataset_binary import BinaryDatasetModule
from core.trainer.trainer import TrainerModule
from core.trainer.BinaryTrainer import BinaryTrainerModule
from core.utils.utils import project_path


def _find_latest_ckpt() -> str | None:
    from pathlib import Path
    ckpts = list(Path(".").rglob("best-*.ckpt"))
    return str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]) if ckpts else None


@hydra.main(version_base=None, config_path="../configs/fine_classifier", config_name="config")
def main(cfg: DictConfig) -> None:
    # seed_everything(cfg.get("seed", 42), workers=True)

    # ckpt_path: str | None = cfg.get("test", {}).get("ckpt_path", None)  # hydra override 가능
    # ckpt_path = ckpt_path or _find_latest_ckpt()
    # if ckpt_path is None:
    #     raise FileNotFoundError("No checkpoint (.ckpt) found. Train the model first or specify test.ckpt_path=<path>.")

    # print(f"Using checkpoint: {ckpt_path}")

    # dm = DatasetModule(cfg)
    # if cfg.trainer.hnm.use_hnm == True:
    #     model = BinaryTrainerModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    # else:
    #     model = BinaryTrainerModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    # model.eval()

    # trainer = Trainer(accelerator="auto", devices="auto", precision="bf16-mixed" if cfg.get("bf16", False) else 32)
    # preds = trainer.predict(model, datamodule=dm)       
    # preds = torch.cat(preds, dim=0)                     
    # preds = torch.argmax(preds, dim=1).cpu().numpy()    

    # submission = pd.read_csv(os.path.join(cfg.data.data_path, "sample_submission.csv"))
    # submission["target"] = preds
    # submission.to_csv("pred.csv", index=False)

    seed_everything(cfg.get("seed", 42), workers=True)

    ckpt_path = cfg.get("test", {}).get("ckpt_path", None) or _find_latest_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint (.ckpt) found.")

    print(f"Using checkpoint: {ckpt_path}")

    # 1. 원래 제출 파일 로드
    submission_path = os.path.join(cfg.data.data_path, "best_submission.csv")
    submission = pd.read_csv(submission_path)

    # 2. 3 또는 7인 row만 추출
    mask = submission["target"].isin([3, 7])
    subset = submission[mask].copy()

    # 3. 해당 ID에 대한 이미지 경로/데이터셋 구성
    # cfg 수정 (inference 시 3,7에 해당하는 것만 로딩하는 DatasetModule 필요)
    cfg.data.submission_ids = subset["ID"].tolist()
    dm = BinaryDatasetModule(cfg)

    # 4. 모델 로드 및 예측
    model = BinaryTrainerModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()
    trainer = Trainer(accelerator="auto", devices="auto", precision="bf16-mixed" if cfg.get("bf16", False) else 32)
    preds = trainer.predict(model, datamodule=dm)
    binary_preds = torch.cat(preds).cpu().numpy() # 0, 1

    # 5. 예측 결과 → 원래 클래스(3, 7)로 다시 매핑
    mapped = [3 if p == 0 else 7 for p in binary_preds]
    
    submission.loc[mask, "target"] = mapped

    # 6. 저장
    submission.to_csv(os.path.join(project_path(), "submission_binary.csv"), index=False)

if __name__ == "__main__":
    main()
