import pandas as pd
from features.encoding import targetEncoding


def feature_engineering(concat_select, is_feature_engineering=False):
    all = list(concat_select['구'].unique())
    gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
    gangbuk = [x for x in all if x not in gangnam]

    assert len(all) == len(gangnam) + len(gangbuk)       # 알맞게 분리되었는지 체크합니다.

    if is_feature_engineering == False:
        # 강남의 여부를 체크합니다.
        is_gangnam = []
        for x in concat_select['구'].tolist() :
            if x in gangnam :
                is_gangnam.append(1)
            else :
                is_gangnam.append(0)

        # 파생변수를 하나 만릅니다.
        concat_select['강남여부'] = is_gangnam
    else:
        gu_mean_price=concat_select.groupby("구")['target'].mean().sort_values()

        gu_grade = pd.qcut(gu_mean_price,q=3, labels=['Low', 'Mid', 'High'])

        gu_grade_map = dict(zip(gu_mean_price.index, gu_grade))

        concat_select["price_grade"] = concat_select["구"].map(gu_grade_map)

        gu_encoding_map = targetEncoding(concat_select, '구')

    # concat_select['계절'] = concat_select['계약월'].map({'12':'겨울', '01':'겨울', '02':'겨울', '03':'봄', '04':'봄', '05':'봄', '06':'여름', '07':'여름', '08':'여름', '09':'가을', '10':'가을', '11':'가을'})

    # concat_select['면적당가격'] = concat_select['target'] / concat_select['전용면적']

    # targetEncoding(concat_select, '구')

    # 건축년도 분포는 아래와 같습니다. 특히 2005년이 Q3에 해당합니다.
    # 2009년 이후에 지어진 건물은 10%정도 되는 것을 확인할 수 있습니다.
    concat_select['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])

    # 따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.
    concat_select['신축여부'] = concat_select['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)

    print(concat_select.columns)

    return concat_select
