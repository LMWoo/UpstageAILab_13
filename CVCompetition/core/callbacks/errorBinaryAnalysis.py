import os
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_lightning import Callback

class ErrorBinaryAnalysisCallback(Callback):
    def __init__(self, 
                 cfg,
                 num_classes: int, 
                 class_names: List[str] | None = None,
                 save_dir="error_binary_analysis", top_k=10):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [str(i) for i in range(num_classes)]
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.top_k = top_k
        self.reset_buffer()

    @staticmethod
    def _map_to_binary(y: torch.Tensor) -> torch.Tensor:
        """3 → 0,   7 → 1  (값이 이미 0/1이면 그대로)"""
        if y.ndim == 2:
            y = y.argmax(1)
        return (y == 7).long() if y.max() > 1 else y.long()
    
    def reset_buffer(self):
        self.all_preds = []
        self.all_probs = []
        self.all_targets = []
        self.all_fnames = []
    
    # ------------------------------------------------------------------
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        x, y_raw = batch
        y_bin = self._map_to_binary(y_raw)                              # (B,)

        logits = pl_module(x).squeeze()                            # (B,)
        prob_pos = torch.sigmoid(logits)                           # (B,)
        probs = torch.stack([1 - prob_pos, prob_pos], dim=1)       # (B,2)
        preds = (prob_pos > 0.5).long()                            # (B,)

        # 파일 이름 수집
        if hasattr(batch, "filenames"):
            fnames = batch.filenames
        elif hasattr(trainer.datamodule.train_ds, "samples"):
            start = batch_idx * len(x)
            fnames = [
                trainer.datamodule.val_ds.samples[i][0]
                for i in range(start, start + len(x))
            ]
        else:
            fnames = [f"sample_{batch_idx}_{i}.png" for i in range(len(x))]

        # 버퍼 저장
        self.all_preds.extend(preds.cpu().numpy())
        self.all_probs.extend(prob_pos.cpu().numpy())              # 1-클래스 확률
        self.all_targets.extend(y_bin.cpu().numpy())
        self.all_fnames.extend(fnames)

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        df = pd.DataFrame(
            {
                "filename": self.all_fnames,
                "target": self.all_targets,
                "pred": self.all_preds,
                "conf_pos": self.all_probs,    # class-1 확률
            }
        )

        # 오분류만 추출
        err_df = df[df["target"] != df["pred"]].sort_values("conf_pos", ascending=False)
        save_df = err_df.head(self.top_k).copy()

        save_df["target_name"] = save_df["target"].map(lambda x: self.class_names[int(x)])
        save_df["pred_name"]   = save_df["pred"].map(lambda x: self.class_names[int(x)])
        save_df = save_df[["filename", "target_name", "pred_name", "conf_pos"]]

        # CSV 저장
        csv_path = os.path.join(self.save_dir, f"errors_ep{trainer.current_epoch:03d}.csv")
        save_df.to_csv(csv_path, index=False)
        print(f"[ErrorAnalysis] saved → {csv_path}")

        # 썸네일 저장
        self._save_thumbnails(save_df, trainer, trainer.current_epoch)
        self.reset_buffer()

    # ------------------------------------------------------------------
    def _save_thumbnails(self, save_df: pd.DataFrame, trainer, epoch: int):
        pass
        # N = len(save_df)
        # if N == 0:
        #     return
        # n_cols = 5
        # n_rows = int(np.ceil(N / n_cols))
        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        # img_root = os.path.join(
        #     trainer.datamodule.data_path, trainer.datamodule.full_data_name
        # )

        # for ax, (_, row) in zip(axes.flatten(), save_df.iterrows()):
        #     try:
        #         img_path = os.path.join(img_root, row["filename"])
        #         img = Image.open(img_path).convert("RGB")
        #         ax.imshow(img)
        #         ax.set_title(
        #             f'{row["target_name"]} → {row["pred_name"]}\n{row["conf_pos"]:.2f}',
        #             fontsize=8,
        #         )
        #         ax.axis("off")
        #     except Exception as e:
        #         ax.axis("off")
        #         print(f"[ErrorAnalysis] cannot load {row['filename']}: {e}")

        # # 여분 subplot 지우기
        # for ax in axes.flatten()[N:]:
        #     ax.axis("off")

        # plt.tight_layout()
        # png_path = os.path.join(self.save_dir, f"errors_ep{epoch:03d}.png")
        # plt.savefig(png_path, dpi=150)
        # plt.close()
        # print(f"[ErrorAnalysis] thumbnails saved → {png_path}")