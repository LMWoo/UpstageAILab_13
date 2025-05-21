import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

from src.utils.utils import project_path

class BaselinePreprocessor:
    def __init__(self, concat):
        self.label_encoders = {}
        self.concat = concat
        self.dt_train = None
        self.dt_test = None

    def _preprocessing(self):
        self.feature_selection()
        self.feature_cleaning()
        self.feature_engineering()
        self.feature_encoding()

    def feature_selection(self):
        self.concat = self.concat.rename(columns={'전용면적(㎡)':'전용면적'})

        # 위 처럼 아무 의미도 갖지 않는 칼럼은 결측치와 같은 역할을 하므로, np.nan으로 채워 결측치로 인식되도록 합니다.
        
        self.concat['등기신청일자'] = self.concat['등기신청일자'].replace(' ', np.nan)
        self.concat['거래유형'] = self.concat['거래유형'].replace('-', np.nan)
        self.concat['중개사소재지'] = self.concat['중개사소재지'].replace('-', np.nan)
        
        # Null값이 100만개 이상인 칼럼은 삭제해보도록 하겠습니다.
        print("* 결측치가 100만개 이하인 변수들 :", list(self.concat.columns[self.concat.isnull().sum() <= 1000000]))     # 남겨질 변수들은 아래와 같습니다.
        print("* 결측치가 100만개 이상인 변수들 :", list(self.concat.columns[self.concat.isnull().sum() >= 1000000]))

        selected = list(self.concat.columns[self.concat.isnull().sum() <= 1000000])
        self.select = self.concat[selected]

    def feature_cleaning(self):

        self.concat['본번'] = self.concat['본번'].astype('str')
        self.concat['부번'] = self.concat['부번'].astype('str')

        continuous_columns = []
        categorical_columns = []

        for column in self.select.columns:
            if pd.api.types.is_numeric_dtype(self.select[column]):
                continuous_columns.append(column)
            else:
                categorical_columns.append(column)

        print("연속형 변수:", continuous_columns)
        print("범주형 변수:", categorical_columns)

        # 범주형 변수에 대한 보간
        self.select[categorical_columns] = self.select[categorical_columns].fillna('NULL')

        # 연속형 변수에 대한 보간 (선형 보간)
        self.select[continuous_columns] = self.select[continuous_columns].interpolate(method='linear', axis=0)


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
        
        self.concat = remove_outliers_iqr(self.concat, '전용면적')

    def feature_engineering(self):
        # 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
        self.select['구'] = self.select['시군구'].map(lambda x : x.split()[1])
        self.select['동'] = self.select['시군구'].map(lambda x : x.split()[2])
        del self.select['시군구']

        self.select['계약년'] = self.select['계약년월'].astype('str').map(lambda x : x[:4])
        self.select['계약월'] = self.select['계약년월'].astype('str').map(lambda x : x[4:])
        del self.select['계약년월']

        all = list(self.select['구'].unique())
        gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
        gangbuk = [x for x in all if x not in gangnam]

        assert len(all) == len(gangnam) + len(gangbuk)       # 알맞게 분리되었는지 체크합니다.

        # 강남의 여부를 체크합니다.
        is_gangnam = []
        for x in self.concat['구'].tolist() :
            if x in gangnam :
                is_gangnam.append(1)
            else :
                is_gangnam.append(0)

        # 파생변수를 하나 만릅니다.
        self.select['강남여부'] = is_gangnam

        self.select['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])

        self.select['신축여부'] = self.select['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)

    def feature_encoding(self):

        # 이제 다시 train과 test dataset을 분할해줍니다. 위에서 제작해 놓았던 is_test 칼럼을 이용합니다.
        self.dt_train = self.select.query('is_test==0')
        self.dt_test = self.select.query('is_test==1')

        # 이제 is_test 칼럼은 drop해줍니다.
        self.dt_train.drop(['is_test'], axis = 1, inplace=True)
        self.dt_test.drop(['is_test'], axis = 1, inplace=True)

        # dt_test의 target은 일단 0으로 임의로 채워주도록 하겠습니다.
        self.dt_test['target'] = 0

        # 파생변수 제작으로 추가된 변수들이 존재하기에, 다시한번 연속형과 범주형 칼럼을 분리해주겠습니다.
        continuous_columns_v2 = []
        categorical_columns_v2 = []

        for column in self.dt_train.columns:
            if pd.api.types.is_numeric_dtype(self.dt_train[column]):
                continuous_columns_v2.append(column)
            else:
                categorical_columns_v2.append(column)

        print("연속형 변수:", continuous_columns_v2)
        print("범주형 변수:", categorical_columns_v2)

        # 아래에서 범주형 변수들을 대상으로 레이블인코딩을 진행해 주겠습니다.

        # 각 변수에 대한 LabelEncoder를 저장할 딕셔너리

        # Implement Label Encoding
        for col in tqdm( categorical_columns_v2 ):
            lbl = LabelEncoder()

            # Label-Encoding을 fit
            lbl.fit(self.dt_train[col].astype(str) )
            self.dt_train[col] = lbl.transform(self.dt_train[col].astype(str))
            self.label_encoders[col] = lbl           # 나중에 후처리를 위해 레이블인코더를 저장해주겠습니다.

            # Test 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가해줍니다.
            for label in np.unique(self.dt_test[col]):
                if label not in lbl.classes_: # unseen label 데이터인 경우
                    lbl.classes_ = np.append(lbl.classes_, label) # 미처리 시 ValueError발생하니 주의하세요!

            self.dt_test[col] = lbl.transform(self.dt_test[col].astype(str))

        assert self.dt_train.shape[1] == self.dt_test.shape[1]          # train/test dataset의 shape이 같은지 확인해주겠습니다.

# class BaselineDataset:
#     def __init__(self, df, scaler=None, label_encoder=None, baseline_preprocessor = None):
#         self.df = df
#         self.features = df.columns
#         self.labels = 


def read_dataset():
    train_path = os.path.join(project_path(), "dataset", "train.csv")
    test_path = os.path.join(project_path(), "dataset", "test.csv")

    dt = pd.read_csv(train_path)
    dt_test = pd.read_csv(test_path)
    
    dt['is_test'] = 0
    dt_test['is_test'] = 1

    concat = pd.concat([dt, dt_test])
    return concat

def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df

def get_datasets(scaler=None, label_encoder=None):
    concat = read_dataset()

    preprocessor = BaselinePreprocessor(concat)
    train_dt, test_dt = preprocessor._preprocessing()

    train_dt, val_dt = train_test_split(train_dt, test_size=0.2, random_state=42)

    train_dataset = BaselineDataset(train_dt, scaler, preprocessor.label_encoders, preprocessor)