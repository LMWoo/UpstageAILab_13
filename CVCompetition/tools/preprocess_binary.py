import os
import sys
import shutil
import random

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import pandas as pd

from core.utils.utils import project_path

if __name__ == "__main__":

    orig_df = pd.read_csv(os.path.join(project_path(), "data/train.csv"))

    pos_neg = [3, 7]
    neg_cls, pos_cls = pos_neg
        
    binary_df = orig_df[orig_df["target"].isin([neg_cls, pos_cls])].reset_index(drop=True)
    binary_df["target"] = binary_df["target"].map({neg_cls: 0, pos_cls: 1})

    print(binary_df.value_counts())
    binary_df.to_csv(os.path.join(project_path(), "data/train_binary.csv"), index=False)
