import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import hydra
import wandb
import torch
import numpy as np

from core.datasets.dataset import DatasetModule
from core.datasets.dataset_binary import BinaryDatasetModule
from core.trainer.trainer import TrainerModule
from core.trainer.HNMTrainer import HardNegativeMiningTrainerModule
from core.trainer.BinaryTrainer import BinaryTrainerModule
from core.callbacks.hardNegativeMining import HNMCallback
from core.callbacks.errorAnalysis import ErrorAnalysisCallback
from core.callbacks.errorBinaryAnalysis import ErrorBinaryAnalysisCallback
from core.callbacks.tsne import TSNECallback
from core.callbacks.perClassLoss import PerClassLossCallback
from core.callbacks.confusionMatrix import ConfusionMatrixCallback
from core.utils.utils import auto_increment_run_suffix, get_latest_run, project_path, make_error_run_dir

@hydra.main(version_base=None, config_path="../configs/fine_classifier", config_name="config")
def main(cfg: DictConfig):
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    project_name = "CV4_Competition_Binary37"

    try:
        run_name = get_latest_run(project_name, cfg.experiment_name)
    except Exception as e:
        print(f"[W&B WARNING] Failed to get previous runs: {e}")
        run_name = f"{cfg.experiment_name.replace('_', '-')}-000"
    next_run_name = auto_increment_run_suffix(run_name)
    wandb.init(
        project=project_name,
        id=next_run_name,
        notes="content-based binary classification model",
        tags=["content-based", "binary classification"],
        config={
            "experiment_name": cfg.experiment_name,
            "model_name": cfg.model.model.model_name,
            "mid_freeze_epochs": cfg.model.model.mid_freeze_epochs,
            "freeze_epochs": cfg.model.model.all_freeze_epochs,
            "batch_size": cfg.data.batch_size
        }
    )

    seed_everything(cfg.get("seed", 42), workers=True)

    data_module = BinaryDatasetModule(cfg)
    data_module.setup("fit")

    OmegaConf.set_struct(cfg, False)

    model = BinaryTrainerModule(cfg)

    ckpt_cb = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best-binary-{epoch:02d}-{val_f1:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_cb = EarlyStopping(
        monitor=cfg.callback.monitor,
        mode=cfg.callback.mode,
        patience=cfg.callback.patience,
        min_delta=0.0005,
        verbose=True,
    )

    error_root_dir = make_error_run_dir("binary_version")

    class_names = ["confirmation_of_admission_and_discharge", "medical_outpatient_certificate"]

    error_cb = ErrorBinaryAnalysisCallback(
        cfg = cfg,
        num_classes=cfg.model.model.num_classes,
        class_names=class_names,
        save_dir=os.path.join(project_path(), error_root_dir, cfg.trainer.error.analysis.save_dir),
        top_k=cfg.trainer.error.analysis.top_k,
    )

    tsne_cb = TSNECallback(
        num_classes=cfg.model.model.num_classes,
        class_names=class_names,
        save_dir=os.path.join(project_path(), error_root_dir, cfg.trainer.error.tsne.save_dir),
        every_n_epoch=cfg.trainer.error.tsne.every_n_epoch,
    )

    pcl_cb = PerClassLossCallback(
        num_classes=cfg.model.model.num_classes,
        class_names=class_names,
        save_dir=os.path.join(project_path(), error_root_dir, cfg.trainer.error.perclassloss.save_dir),
    )

    cm_cb = ConfusionMatrixCallback(
        num_classes=cfg.model.model.num_classes,
        class_names=class_names,
        save_dir=os.path.join(project_path(), error_root_dir, cfg.trainer.error.confusion_matrix.save_dir),
    )

    callbacks = [ckpt_cb, lr_monitor, early_stop_cb, error_cb, tsne_cb]

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg.get("bf16", False) else 32,
        callbacks=callbacks,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
    )

    # 8. 학습
    trainer.fit(model, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    main()