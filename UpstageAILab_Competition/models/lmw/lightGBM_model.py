import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os
import pickle

import lightgbm as lgb
import joblib

import eli5
from eli5.sklearn import PermutationImportance

from models.base import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, X_train, X_val, Y_train, Y_val, X_test):
        super().__init__(X_train, X_val, Y_train, Y_val, X_test)
        print('LightGBM model initialize')

        self.model = lgb.LGBMRegressor(n_estimators=100000,
                                       metric='rmse',
                                       data_sample_strategy='goss',
                                       max_depth=12,
                                       num_leaves=62,
                                       min_data_in_leaf=40)
        
    def train(self):
        print('start model train')
        self.model.fit(self.X_train, self.Y_train,
                       eval_set = [(self.X_train, self.Y_train), (self.X_val, self.Y_val)],
                       eval_metric = 'rmse',
                       categorical_feature='auto',
                       callbacks=[lgb.early_stopping(stopping_rounds=50),
                                  lgb.log_evaluation(period=10, show_stdv=True)])
        print('finish model train')
    
    def validation(self):
        print('start model validation')
        pred = self.model.predict(self.X_val)
        print(f'RMSE test: {np.sqrt(metrics.mean_squared_error(self.Y_val, pred))}')
        print('finish model validation')

    def test(self):
        self.X_test = self.X_test.drop(['target'], axis=1)

        real_test_pred = self.model.predict(self.X_test)
        return real_test_pred

    def analysis_validation(self, save_path):
        # 변수 중요도 확인
        importances = pd.Series(self.model.feature_importances_, index=list(self.X_train.columns))
        importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(10,8))
        plt.title("Feature Importances")
        sns.barplot(x=importances, y=importances.index)
        plt.savefig(save_path)

        # perm = PermutationImportance(self.model,        # 위에서 학습된 모델을 이용하겠습니다.
        #                      scoring = "neg_mean_squared_error",        # 평가 지표로는 회귀문제이기에 negative rmse를 사용합니다. (neg_mean_squared_error : 음의 평균 제곱 오차)
        #                      random_state = 42,
        #                      n_iter=3).fit(self.X_val, self.Y_val)

        # def calculate_se(target, pred):
        #     squared_errors = (target - pred) ** 2
        #     return squared_errors

        # squared_errors = calculate_se(self.Y_val, self.Pred_val)
        # squared_errors = squared_errors.sort_values(ascending=False)
    
    def save_model(self, save_path):
        joblib.dump(self.model, save_path)

    def load_model(self, load_path):
        self.model = joblib.load(load_path)