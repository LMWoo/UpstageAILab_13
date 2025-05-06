import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings;warnings.filterwarnings('ignore')

# Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import eli5
from eli5.sklearn import PermutationImportance

from data.load import load_data


import argparse

parser = argparse.ArgumentParser(description="Train parser")

parser.add_argument("--feature_reduction", type=bool, default=False)
parser.add_argument("--model_name", type=str, default="save_model")
parser.add_argument("--save_file", type=str, default="output")

if __name__ == "__main__":

    args = parser.parse_args()

    X_train, y_train, X_val, y_val, categorical_columns_v2, label_encoders, dt_test = load_data(args.feature_reduction)


    dt_test.head(2)      # test dataset에 대한 inference를 진행해보겠습니다.

    # 저장된 모델을 불러옵니다.
    with open('./weights/'+ args.model_name + '.pkl', 'rb') as f:
        model = pickle.load(f)

    # %%time
    X_test = dt_test.drop(['target'], axis=1)

    # Test dataset에 대한 inference를 진행합니다.
    real_test_pred = model.predict(X_test)

    # 앞서 예측한 예측값들을 저장합니다.
    preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
    preds_df.to_csv('./results/' + args.save_file + '.csv', index=False)
        
    print('finish test')
