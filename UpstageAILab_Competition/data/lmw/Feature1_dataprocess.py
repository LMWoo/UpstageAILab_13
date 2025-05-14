import pandas as pd
import pickle
from data.base import BasePreprocess
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np

class Feature1_Preprocess(BasePreprocess):
    def __init__(self):
        super().__init__()
        self.train_data = pd.read_csv("../../data/train.csv")
        self.test_data = pd.read_csv("../../data/test.csv")

        self.train_data['is_test'] = 0
        self.test_data['is_test'] = 1
        
        self.concat = pd.concat([self.train_data, self.test_data])

        self.concat = self.concat.rename(columns={'전용면적(㎡)':'전용면적'})

        self.numerical_columns = ['전용면적', '계약년월', '계약일', '층', '건축년도', '해제사유발생일']
        self.categorical_columns = ['시군구', '번지', '본번', '부번', '아파트명', '도로명']
        self.selection_columns = self.basic_columns + self.numerical_columns + self.categorical_columns

    def feature_selection(self):
        print('start feature selection')
        existing_cols = [col for col in self.selection_columns if col in self.concat.columns]

        self.concat = self.concat[existing_cols]
        print('finish feature selection')

    def feature_cleaning(self):
        print('start feature cleaning')

        if '부번' in self.concat.columns:
            self.concat['부번'] = self.concat['부번'].astype(str)
        if '본번' in self.concat.columns:
            self.concat['본번'] = self.concat['본번'].astype(str)
        
        # 범주형 변수에 대한 보간
        self.concat[self.categorical_columns] = self.concat[self.categorical_columns].fillna('NULL')

        # 연속형 변수에 대한 보간
        self.concat[self.numerical_columns] = self.concat[self.numerical_columns].interpolate(method='linear', axis=0)

        def remove_outliers_iqr(dt, column_name):
            df = dt.query('is_test == 0')       # train data 내에 있는 이상치만 제거하도록 하겠습니다.
            df_test = dt.query('is_test == 1')

            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

            result = pd.concat([df, df_test])   # test data와 다시 합쳐주겠습니다.
            return result

        self.concat = remove_outliers_iqr(self.concat, '전용면적')
        print('finish feature cleaning')

    def feature_engineering(self):
        print('start feature engineering')

        self.concat['계약유지여부'] = self.concat['해제사유발생일'].isna()
        self.concat.drop(columns='해제사유발생일', inplace=True)

        print('finish feature engineering')

    def feature_encoding(self):
        print('start encoding')
        
        dt_train = self.concat.query('is_test==0')
        dt_test = self.concat.query('is_test==1')

        dt_train.drop(['is_test'], axis=1, inplace=True)
        dt_test.drop(['is_test'], axis=1, inplace=True)

        for col in tqdm(self.categorical_columns):
            lbl = LabelEncoder()

            lbl.fit(dt_train[col].astype(str))
            dt_train[col] = lbl.transform(dt_train[col].astype(str))
            self.label_encoders[col] = lbl

            for label in np.unique(dt_test[col]):
                if label not in lbl.classes_: # unseen label 데이터인 경우
                    lbl.classes_ = np.append(lbl.classes_, label) # 미처리 시 ValueError발생하니 주의하세요!

            dt_test[col] = lbl.transform(dt_test[col].astype(str))


        self.preprocessed_data = {'X_train': dt_train, 'X_test': dt_test}

        print('finish encoding')

    def get_preprocessed_data(self):
        return self.preprocessed_data