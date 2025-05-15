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

    def analysis_validation(self, save_root_path, data_preprocessor):
        # 변수 중요도 확인
        importances = pd.Series(self.model.feature_importances_, index=list(self.X_train.columns))
        importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(10,8))
        plt.title("Feature Importances")
        sns.barplot(x=importances, y=importances.index)
        plt.savefig(os.path.join(save_root_path, 'FeatureImportances.png'))

        X_val_index = self.X_val.sample(n=10000, random_state=42).index
        perm = PermutationImportance(self.model,        # 위에서 학습된 모델을 이용하겠습니다.
                             scoring = "neg_mean_squared_error",        # 평가 지표로는 회귀문제이기에 negative rmse를 사용합니다. (neg_mean_squared_error : 음의 평균 제곱 오차)
                             random_state = 42,
                             n_iter=3).fit(self.X_val.loc[X_val_index], self.Y_val.loc[X_val_index])
        
        with open(os.path.join(save_root_path, 'PermutationImportance.html'), 'w', encoding='utf-8') as f:
            html = eli5.format_as_html(eli5.explain_weights(perm, feature_names=self.X_val.loc[X_val_index].columns.tolist()))
            f.write(html)


        # Best, Worst Top 100
        def calculate_se(target, pred):
            squared_errors = (target - pred) ** 2
            return squared_errors
        
        pred = self.model.predict(self.X_val)
        squared_errors = calculate_se(self.Y_val, pred)
        self.X_val['error'] = squared_errors
        self.X_val['target'] = self.Y_val

        X_val_sort = self.X_val.sort_values(by='error', ascending=False)
        X_val_sort_top100 = X_val_sort.head(100)
        X_val_sort_tail100 = X_val_sort.tail(100)

        error_top100 = X_val_sort_top100.copy()
        for column in data_preprocessor.categorical_columns:
            error_top100[column] = data_preprocessor.label_encoders[column].inverse_transform(X_val_sort_top100[column])
        
        best_top100 = X_val_sort_tail100.copy()
        for column in data_preprocessor.categorical_columns:
            best_top100[column] = data_preprocessor.label_encoders[column].inverse_transform(X_val_sort_tail100[column])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        sns.boxplot(data = error_top100, x='target', ax=axes[0])
        axes[0].set_title('Worst Top 100')

        sns.boxplot(data = best_top100, x='target', color='orange', ax=axes[1])
        axes[1].set_title('Best Top 100')

        plt.tight_layout()

        save_comparision_path = os.path.join(save_root_path, 'comparision')
        if not os.path.exists(save_comparision_path):
            os.makedirs(save_comparision_path)
        else:
            pass
        plt.savefig(os.path.join(save_comparision_path, 'top_worst_100_boxplot.png'), dpi=300)

        print(data_preprocessor.categorical_columns)
        for column in data_preprocessor.categorical_columns:
            plt.figure(figsize=(30,15))

            sns.histplot(data = error_top100, x=column, alpha=0.5)
            sns.histplot(data = best_top100, x=column, color='orange', alpha=0.5)
            
            plt.title(f'{column} Distribution Comparison')

            plt.xticks(rotation=60, ha='right')
            plt.savefig(os.path.join(save_comparision_path, column + '.png'), dpi=300)


    def save_model(self, save_path):
        joblib.dump(self.model, save_path)

    def load_model(self, load_path):
        self.model = joblib.load(load_path)