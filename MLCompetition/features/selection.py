import pandas as pd
import numpy as np

def feature_selection(dt, dt_test, is_feature_reduction=False, is_subway=False):

    # train/test 구분을 위한 칼럼을 하나 만들어 줍니다.
    dt['is_test'] = 0
    dt_test['is_test'] = 1
    concat = pd.concat([dt, dt_test])     # 하나의 데이터로 만들어줍니다.


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

    print("* 결측치가 100만개 이하인 변수들 :", list(concat.columns[concat.isnull().sum() <= 1000000]))     # 남겨질 변수들은 아래와 같습니다.
    print("* 결측치가 100만개 이상인 변수들 :", list(concat.columns[concat.isnull().sum() >= 1000000]))

    # 위에서  100만개 이하인 변수들만 골라 새로운 concat_select 객체로 저장해줍니다.

    save_columns = [ "건축년도", "계약년월", "전용면적", "시군구", "부번", "신축여부", "강남여부", 'target', 'is_test']
    
    if is_subway == True:
        save_columns.append('is_subway_near')
    if is_feature_reduction == True:
        selected = list(concat.columns[concat.isnull().sum() <= 1000000])
            # "건축년도", 
            # "좌표X", "좌표Y",
            # "도로명",
            # '전용면적','계약년월','건축년도',
                        #  '강남여부', ,'시군구',
                        #  '좌표X', '좌표Y',
                        #  'k-건설사(시공사)', 'k-시행사',
                        #  'k-주거전용면적','k-전체동수', 
                        # '주차대수',
                        #  '본번', '부번', '아파트명', '도로명','번지', 'target', 'is_test']
        selected = [x for x in selected if x in save_columns]
        concat_select = concat[selected]

        concat_select['부번'] = concat_select['부번'].astype('str')
    else:
        selected = list(concat.columns[concat.isnull().sum() <= 1000000])
        concat_select = concat[selected]
 
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

    # 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
    concat_select['구'] = concat_select['시군구'].map(lambda x : x.split()[1])
    concat_select['동'] = concat_select['시군구'].map(lambda x : x.split()[2])
    del concat_select['시군구']

    concat_select['계약년'] = concat_select['계약년월'].astype('str').map(lambda x : x[:4])
    concat_select['계약월'] = concat_select['계약년월'].astype('str').map(lambda x : x[4:])
    del concat_select['계약년월']

    return concat_select
