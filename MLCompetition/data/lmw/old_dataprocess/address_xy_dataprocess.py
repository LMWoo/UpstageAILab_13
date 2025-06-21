import pandas as pd
import pickle
from data.base import BasePreprocess
from data.lmw.baseline_dataprocess import BaselinePreprocess
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

class AddressToXYPreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()

        self.baselinePreprocess = BaselinePreprocess()

        with open("../data/adres_to_geo.pickle", "rb") as f:
            adres_to_geo = pickle.load(f)

        def address_to_xy(dt):
            for i in tqdm(range(dt.shape[0])):
                
                try:
                    add = dt['시군구'].iloc[i] + ' ' + dt['번지'].iloc[i]
                    dt.loc[i, '좌표X'] = adres_to_geo[add][1]
                    dt.loc[i, '좌표Y'] = adres_to_geo[add][0]
                except:
                    pass
            return dt

        if os.path.exists('../data/train_xy.csv') and os.path.exists('../data/test_xy.csv') :
            self.train_data = pd.read_csv("../data/train_xy.csv")
            self.test_data = pd.read_csv("../data/test_xy.csv")
        else:
            self.train_data = pd.read_csv("../data/train.csv")
            self.test_data = pd.read_csv("../data/test.csv")
            self.train_data = address_to_xy(self.train_data)
            self.test_data = address_to_xy(self.test_data)
            self.train_data.to_csv('../data/train_xy.csv')
            self.test_data.to_csv('../data/test_xy.csv')
        
        self.baselinePreprocess.train_data = self.train_data
        self.baselinePreprocess.test_data = self.test_data

    def feature_selection(self):
        self.baselinePreprocess.feature_selection()

    def feature_cleaning(self):
        self.baselinePreprocess.feature_cleaning()

    def feature_engineering(self):
        self.baselinePreprocess.feature_engineering()

        self.train_data = self.baselinePreprocess.train_data
        self.test_data = self.baselinePreprocess.test_data
        self.concat = self.baselinePreprocess.concat

    def encoding(self):
        print('start encoding')
        dt_train = self.concat.query('is_test==0')
        dt_test = self.concat.query('is_test==1')

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

        self.preprocessed_data = {'X_train': dt_train, 'X_test': dt_test}

        print('finish encoding')

    def get_preprocessed_data(self):
        return self.preprocessed_data