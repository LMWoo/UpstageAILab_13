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

import eli5
from eli5.sklearn import PermutationImportance

from models.base import BaseModel

class BaselineModel(BaseModel):
    def __init__(self, data_preprocessor):
        super().__init__(data_preprocessor)
        print('Baseline model initialize')

        self.label_encoders = {}

        self.X_train = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.Pred_val = None

        self.X_test = None

        self.model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=1)

    def encoding(self):
        print('start encoding')
        dt_train = self.data_preprocessor.concat.query('is_test==0')
        dt_test = self.data_preprocessor.concat.query('is_test==1')

        dt_train.drop(['is_test'], axis=1, inplace=True)
        dt_test.drop(['is_test'], axis=1, inplace=True)

        print(dt_train.shape, dt_test.shape)

        continuous_columns_v2 = []
        categorical_columns_v2 = []

        for column in dt_train.columns:
            if pd.api.types.is_numeric_dtype(dt_train[column]):
                continuous_columns_v2.append(column)
            else:
                categorical_columns_v2.append(column)

        print("연속형 변수:", continuous_columns_v2)
        print("범주형 변수:", categorical_columns_v2)

        for col in tqdm( categorical_columns_v2 ):
            lbl = LabelEncoder()

            # Label-Encoding을 fit
            lbl.fit( dt_train[col].astype(str) )
            dt_train[col] = lbl.transform(dt_train[col].astype(str))
            self.label_encoders[col] = lbl           # 나중에 후처리를 위해 레이블인코더를 저장해주겠습니다.

            # Test 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가해줍니다.
            for label in np.unique(dt_test[col]):
                if label not in lbl.classes_: # unseen label 데이터인 경우
                    lbl.classes_ = np.append(lbl.classes_, label) # 미처리 시 ValueError발생하니 주의하세요!

            dt_test[col] = lbl.transform(dt_test[col].astype(str))

        self.X_train = dt_train
        self.X_test = dt_test

        print('finish encoding')

    def splitdata(self):
        print('start split data')
        self.Y_train = self.X_train['target']
        self.X_train = self.X_train.drop(['target'], axis=1)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.2, random_state=2023)
        print('finish split data')

    def train(self):
        print('start model train')
        self.model.fit(self.X_train, self.Y_train)
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
        # 학습된 모델을 저장합니다. Pickle 라이브러리를 이용하겠습니다.
        with open(os.path.join(save_path), 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, load_path):
        with open(os.path.join(load_path), 'rb') as f:
            self.model = pickle.load(f)