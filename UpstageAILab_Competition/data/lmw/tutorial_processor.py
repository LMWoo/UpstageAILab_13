import pandas as pd
import pickle
from data.base import BasePreprocess
from tqdm import tqdm
import numpy as np

class TutorialPreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()
        self.train_data = pd.read_csv("../data/train.csv")
        self.test_data = pd.read_csv("../data/test.csv")

        self.train_data['is_test'] = 0
        self.test_data['is_test'] = 1
        
        self.concat = pd.concat([self.train_data, self.test_data])
        self.concat = self.concat.rename(columns={'전용면적(㎡)':'전용면적'})

    def feature_selection(self):
        print('start feature selection')
        # 결측치 탐색 및 보간
        self.concat['등기신청일자'] = self.concat['등기신청일자'].replace(' ', np.nan)
        self.concat['거래유형'] = self.concat['거래유형'].replace('-', np.nan)
        self.concat['중개사소재지'] = self.concat['중개사소재지'].replace('-', np.nan)

        print("* 결측치가 100만개 이하인 변수들 :", list(self.concat.columns[self.concat.isnull().sum() <= 1000000]))     # 남겨질 변수들은 아래와 같습니다.
        # print("* 결측치가 100만개 이상인 변수들 :", list(self.concat.columns[self.concat.isnull().sum() >= 1000000]))

        # 결측치가 100만개 이하인 변수들 남김
        selected = list(self.concat.columns[self.concat.isnull().sum() <= 1000000])
        concat_select = self.concat[selected]

        concat_select['본번'] = concat_select['본번'].astype('str')
        concat_select['부번'] = concat_select['부번'].astype('str')

        continuous_columns = []
        categorical_columns = []

        for column in concat_select.columns:
            if pd.api.types.is_numeric_dtype(concat_select[column]):
                continuous_columns.append(column)
            else:
                categorical_columns.append(column)

        print("연속형 변수:", continuous_columns)
        print("범주형 변수:", categorical_columns)

        # 범주형 변수에 대한 보간
        concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')

        # 연속형 변수에 대한 보간
        concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)

        # 이상치 제거 방법에는 IQR을 이용하겠습니다.
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

        self.concat = remove_outliers_iqr(concat_select, '전용면적')
        print('finish feature selection')

    def feature_engineering(self):
        print('start feature engineering')

        concat_select = self.concat
        concat_select['구'] = concat_select['시군구'].map(lambda x : x.split()[1])
        concat_select['동'] = concat_select['시군구'].map(lambda x : x.split()[2])
        del concat_select['시군구']

        concat_select['계약년'] = concat_select['계약년월'].astype('str').map(lambda x : x[:4])
        concat_select['계약월'] = concat_select['계약년월'].astype('str').map(lambda x : x[4:])
        del concat_select['계약년월']

        all = list(concat_select['구'].unique())
        gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
        gangbuk = [x for x in all if x not in gangnam]

        assert len(all) == len(gangnam) + len(gangbuk)       # 알맞게 분리되었는지 체크합니다.

        # 강남의 여부를 체크합니다.
        is_gangnam = []
        for x in concat_select['구'].tolist():
            if x in gangnam :
                is_gangnam.append(1)
            else :
                is_gangnam.append(0)

        # 파생변수를 하나 만릅니다.
        concat_select['강남여부'] = is_gangnam

        # 따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.
        concat_select['신축여부'] = concat_select['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)

        self.concat = concat_select
        print('finish feature engineering')