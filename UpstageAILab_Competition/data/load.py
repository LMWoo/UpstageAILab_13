
# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns

# utils
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

def load_data(feature_reduction=False): 
   # 필요한 데이터를 load 하겠습니다. 경로는 환경에 맞게 지정해주면 됩니다.
    train_path = '../../data/train.csv'
    test_path  = '../../data/test.csv'
    dt = pd.read_csv(train_path)
    dt_test = pd.read_csv(test_path)

    # Train data와 Test data shape은 아래와 같습니다.
    print('Train data shape : ', dt.shape, 'Test data shape : ', dt_test.shape)

    # # Train과 Test data를 살펴보겠습니다.
    # display(dt.head(1))
    # display(dt_test.head(1))      # 부동산 실거래가(=Target) column이 제외된 모습입니다.

    # train/test 구분을 위한 칼럼을 하나 만들어 줍니다.
    dt['is_test'] = 0
    dt_test['is_test'] = 1
    concat = pd.concat([dt, dt_test])     # 하나의 데이터로 만들어줍니다.

    print(concat['is_test'].value_counts())     # train과 test data가 하나로 합쳐진 것을 확인할 수 있습니다.

    # 칼럼 이름을 쉽게 바꿔주겠습니다. 다른 칼럼도 사용에 따라 바꿔주셔도 됩니다!
    concat = concat.rename(columns={'전용면적(㎡)':'전용면적'})

    # 실제로 결측치라고 표시는 안되어있지만 아무 의미도 갖지 않는 element들이 아래와 같이 존재합니다.
    # 아래 3가지의 경우 모두 아무 의미도 갖지 않는 element가 포함되어 있습니다.
    # display(concat['등기신청일자'].value_counts())

    # display(concat['거래유형'].value_counts())

    # display(concat['중개사소재지'].value_counts())

    # 위 처럼 아무 의미도 갖지 않는 칼럼은 결측치와 같은 역할을 하므로, np.nan으로 채워 결측치로 인식되도록 합니다.
    concat['등기신청일자'] = concat['등기신청일자'].replace(' ', np.nan)
    concat['거래유형'] = concat['거래유형'].replace('-', np.nan)
    concat['중개사소재지'] = concat['중개사소재지'].replace('-', np.nan)

    print(concat.isnull().sum())


    '''
    # 변수별 결측치의 비율을 plot으로 그려보면 아래와 같습니다.
    fig = plt.figure(figsize=(13, 2))
    missing = concat.isnull().sum() / concat.shape[0]
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot.bar(color='orange')
    plt.title('변수별 결측치 비율')
    plt.show()
    '''

    # Null값이 100만개 이상인 칼럼은 삭제해보도록 하겠습니다.
    print("* 결측치가 100만개 이하인 변수들 :", list(concat.columns[concat.isnull().sum() <= 1000000]))     # 남겨질 변수들은 아래와 같습니다.
    print("* 결측치가 100만개 이상인 변수들 :", list(concat.columns[concat.isnull().sum() >= 1000000]))

    # 위에서 결측치가 100만개 이하인 변수들만 골라 새로운 concat_select 객체로 저장해줍니다.

    if feature_reduction == True:
        selected = list(concat.columns[concat.isnull().sum() <= 1000000])
        save_columns = [
            "계약년월", "전용면적", "시군구",
            "건축년도", "강남여부", "좌표X",
            'target', 'is_test', "도로명", "부번"]
            # '전용면적','계약년월','건축년도',
                        #  '강남여부', '신축여부','시군구',
                        #  '좌표X', '좌표Y',
                        #  'k-건설사(시공사)', 'k-시행사',
                        #  'k-주거전용면적','k-전체동수', 
                        # '주차대수',
                        #  '본번', '부번', '아파트명', '도로명','번지', 'target', 'is_test']
        selected = [x for x in selected if x in save_columns]
        concat_select = concat[selected]
    else:
        selected = list(concat.columns[concat.isnull().sum() <= 1000000])
        concat_select = concat[selected]



    print(concat_select.isnull().sum())     # 결측치가 100만개 초과인 칼럼이 제거된 모습은 아래와 같습니다.

    print(concat_select.info())

    # 본번, 부번의 경우 float로 되어있지만 범주형 변수의 의미를 가지므로 object(string) 형태로 바꾸어주고 아래 작업을 진행하겠습니다.
    concat_select['본번'] = concat_select['본번'].astype('str')
    concat_select['부번'] = concat_select['부번'].astype('str')

    # 먼저, 연속형 변수와 범주형 변수를 위 info에 따라 분리해주겠습니다.
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

    # 연속형 변수에 대한 보간 (선형 보간)
    concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)

    print(concat_select.isnull().sum())         # 결측치가 보간된 모습을 확인해봅니다.)

    print(concat_select.shape)

    '''
    # 대표적인 연속형 변수인 “전용 면적” 변수 관련한 분포를 먼저 살펴보도록 하겠습니다.
    fig = plt.figure(figsize=(7, 3))
    sns.boxplot(data = concat_select, x = "전용면적", color="lightgreen")
    plt.title("전용면적 분포")
    plt.xlabel("Area")
    plt.show()
    '''

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

    # 위 방법으로 전용 면적에 대한 이상치를 제거해보겠습니다.
    concat_select = remove_outliers_iqr(concat_select, '전용면적')

    print(concat_select.shape)

    print(concat_select['is_test'].value_counts())     # 또한, train data만 제거되었습니다.

    # 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
    concat_select['구'] = concat_select['시군구'].map(lambda x : x.split()[1])
    concat_select['동'] = concat_select['시군구'].map(lambda x : x.split()[2])
    del concat_select['시군구']

    concat_select['계약년'] = concat_select['계약년월'].astype('str').map(lambda x : x[:4])
    concat_select['계약월'] = concat_select['계약년월'].astype('str').map(lambda x : x[4:])
    del concat_select['계약년월']

    print(concat_select.columns)

    all = list(concat_select['구'].unique())
    gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
    gangbuk = [x for x in all if x not in gangnam]

    assert len(all) == len(gangnam) + len(gangbuk)       # 알맞게 분리되었는지 체크합니다.

    # 강남의 여부를 체크합니다.
    is_gangnam = []
    for x in concat_select['구'].tolist() :
        if x in gangnam :
            is_gangnam.append(1)
        else :
            is_gangnam.append(0)

    # 파생변수를 하나 만릅니다.
    concat_select['강남여부'] = is_gangnam

    print(concat_select.columns)

    # 건축년도 분포는 아래와 같습니다. 특히 2005년이 Q3에 해당합니다.
    # 2009년 이후에 지어진 건물은 10%정도 되는 것을 확인할 수 있습니다.
    concat_select['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])

    # 따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.
    concat_select['신축여부'] = concat_select['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)

    print(concat_select.head(1))      # 최종 데이터셋은 아래와 같습니다.)

    print(concat_select.shape)

    # 이제 다시 train과 test dataset을 분할해줍니다. 위에서 제작해 놓았던 is_test 칼럼을 이용합니다.
    dt_train = concat_select.query('is_test==0')
    dt_test = concat_select.query('is_test==1')

    # 이제 is_test 칼럼은 drop해줍니다.
    dt_train.drop(['is_test'], axis = 1, inplace=True)
    dt_test.drop(['is_test'], axis = 1, inplace=True)
    print(dt_train.shape, dt_test.shape)

    print(dt_test.head(1))

    dt_test['target'] = 0

    # 파생변수 제작으로 추가된 변수들이 존재하기에, 다시한번 연속형과 범주형 칼럼을 분리해주겠습니다.
    continuous_columns_v2 = []
    categorical_columns_v2 = []

    for column in dt_train.columns:
        if pd.api.types.is_numeric_dtype(dt_train[column]):
            continuous_columns_v2.append(column)
        else:
            categorical_columns_v2.append(column)

    print("연속형 변수:", continuous_columns_v2)
    print("범주형 변수:", categorical_columns_v2)

    # 아래에서 범주형 변수들을 대상으로 레이블인코딩을 진행해 주겠습니다.

    # 각 변수에 대한 LabelEncoder를 저장할 딕셔너리
    label_encoders = {}

    # Implement Label Encoding
    for col in tqdm( categorical_columns_v2 ):
        lbl = LabelEncoder()

        # Label-Encoding을 fit
        lbl.fit( dt_train[col].astype(str) )
        dt_train[col] = lbl.transform(dt_train[col].astype(str))
        label_encoders[col] = lbl           # 나중에 후처리를 위해 레이블인코더를 저장해주겠습니다.

        # Test 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가해줍니다.
        for label in np.unique(dt_test[col]):
            if label not in lbl.classes_: # unseen label 데이터인 경우
                lbl.classes_ = np.append(lbl.classes_, label) # 미처리 시 ValueError발생하니 주의하세요!

        dt_test[col] = lbl.transform(dt_test[col].astype(str))

    print(dt_train.head(1))      # 레이블인코딩이 된 모습입니다.

    assert dt_train.shape[1] == dt_test.shape[1]          # train/test dataset의 shape이 같은지 확인해주겠습니다.


    # Target과 독립변수들을 분리해줍니다.
    y_train = dt_train['target']
    X_train = dt_train.drop(['target'], axis=1)

    # Hold out split을 사용해 학습 데이터와 검증 데이터를 8:2 비율로 나누겠습니다.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2023)

    return X_train, y_train, X_val, y_val, categorical_columns_v2, label_encoders, dt_test 