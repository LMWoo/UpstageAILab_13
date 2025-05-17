# 아파트 실거래 가격 예측
## 5G.AI

| ![이민우-photoaidcom-cropped (2)](https://github.com/user-attachments/assets/b71c7815-e0c7-407f-b593-121e11c61d05) | ![박진일2-photoaidcom-cropped (1)](https://github.com/user-attachments/assets/551cb3b6-c019-4e50-a35b-1843d3e474a3) | ![정예규-photoaidcom-cropped](https://github.com/user-attachments/assets/7127ed78-2c2e-4225-a92d-149d6452d30c) | ![조은별-photoaidcom-cropped](https://github.com/user-attachments/assets/781faa65-c309-49ee-b58a-9cc08b13b69c) | ![조재형-photoaidcom-cropped](https://github.com/user-attachments/assets/c7d4d78c-36ae-4fb1-bbd4-105ee9d722cc) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [이민우](https://github.com/LMWoo) |  [박진일](https://github.com/r0feynman)  |  [정예규](https://github.com/yegyu)   | [조은별](https://github.com/eunbyul2) | [조재형](https://github.com/Bitefriend)  |
| 팀장 / 파이프라인 설계 및 통합, EDA 분석 총괄 | 거래 패턴 분석 기반 EDA 및 핵심 변수 도출 | 모델링 / 피처 활용 최적화 및 성능 평가 | 거리 기반 피처 엔지니어링 및 분석 결과 정리 | 데이터 정제 로직 및 시각화 기반 인사이트 제공 |

## 0. Overview
본 프로젝트는 서울특별시 아파트의 실거래가를 예측하는 머신러닝 모델을 개발하는 데 목적이 있습니다. 
부동산 가격은 입지, 규모, 건축년도, 교통, 상권 등 다양한 요인에 의해 영향을 받으며, 정확한 예측은 개인·정부·시장 모두에 중요한 역할을 합니다.

- 개인은 합리적인 가격에 주택을 구입하거나 매도할 수 있고,
- 정부는 이상 거래를 감지하여 시장의 투명성을 확보할 수 있으며,
- 시장에서는 수요와 공급의 예측을 통해 안정적인 거래가 가능합니다.

대회는 2007년부터 2023년까지의 실거래 데이터를 기반으로, 2023년 이후 테스트 데이터에 대한 **매매 가격(target)** 을 예측하는 **Regression** 문제로 구성되어 있으며, **RMSE(Root Mean Squared Error)** 지표를 통해 모델 성능을 평가합니다.

### Timeline
  - 5월 1일 - Making A Group
  - 5월 2~5일 - Start Competition
  - 5월 6~8일 - Data Collection
  - 5월 10~12일 - Data EDA
  - 5월 12~14일 - Modeling & Tuning
  - 5월 15일 - Report Writing

### Evaluation
 - **평가지표:** RMSE (Root Mean Squared Error)
![RMSE](https://lh6.googleusercontent.com/cKB-Cb275gGnl_wFKTnUGB3qLDn-q8fo6phdX_sgPoQSKj2MjE2kOPjC3qE39B2NDkhEWOUQ5LPttHsl4fQiKUyhYXTvFh1A33Ru1dXgIF1NYr-eVvR_AdJlqVwEfCNcXt5W3874k_16TByJDIm3z5E)
 - RMSE는 예측된 값과 실제 값 간의 평균편차를 측정합니다. 아파트 매매의 맥락에서는 회귀 모델이 실제 거래 가격의 차이를 얼마나 잘 잡아내는지 측정합니다.

## 1. Components

### Directory

- 각 작업자 폴더명
  - lmw: 이민우, pji: 박진일, jyg: 정예규, jeb: 조은별, jjh: 조재형
   
- 프로젝트 폴더 구조
  ```
  ├── data # Data 전처리
  │   ├── base.py
  │   │── 작업자 A
  │   │   ├── dataprocess_1.py
  │   │   ├── dataprocess_2.py
  │   │   └── dataprocess_3.py
  │   │── 작업자 B
  │   │── 작업자 C
  ├── experiments # 실험별 학습, 평가, 테스트 실행하는 bash 파일
  │   │── 작업자 A
  │   │   └── exp_1
  │   │   │   ├── train.sh
  │   │   │   └── test.sh
  │   │   └── exp_2
  │   │       ├── train.sh
  │   │       └── test.sh    
  │   │── 작업자 B
  │   │── 작업자 C
  ├── features # 현재 기능 없음
  ├── models # 학습 모델
  │   ├── base.py
  │   │── 작업자 A
  │   │   ├── model_1.py
  │   │   ├── model_2.py
  │   │   └── model_3.py
  │   │── 작업자 B
  │   │── 작업자 C
  ├── notebooks # EDA 등 개인별 작업 notebook
  │   │── 작업자 A
  │   │── 작업자 B
  │   │── 작업자 C
  ├── utils 
  └── main.py
  ```

## 2. Develop Guide

### 1. Data Preprocess

 - Data folder -> 본인 폴더 -> tutorial_process.py
 - feature_selection : 결측치 처리, 변수 구분 (범주형, 연속형), 대체 및 삭제, 이상치 제거 등 개발
 - feature_engineering : 파생 변수 개발
 - 작업 위치
   ```
   ├── data
   │   ├── base.py
   │   │── 작업자 A
   │   │   ├── tutorial_process.py
   ```

 - 코드 구조 : Base
   ```
   # Base Class
   class BasePreprocess:
       def __init__(self):
           self.train_data = None
           self.test_data = None
           self.concat = None
   
       # main 에서 호출 되는 함수
       @final
       def preprocess_data(self): 
           self.feature_selection()
           self.feature_engineering()

       # 결측치, 이상치 처리
       @abstractmethod
       def feature_selection(self):
           pass

       # 파생 변수 예시
       @abstractmethod
       def feature_engineering(self):
           pass
   ```
 - 실제 개발 예시 : lmw -> tutorial_dataprocess.py, 길어서 일부 생략
   ```
   import pandas as pd
   import pickle
   from data.base import BasePreprocess
   from tqdm import tqdm
   import numpy as np
   
   # baseline 코드 모듈화
   class TutorialPreprocess(BasePreprocess):
       def __init__(self):
           super().__init__()
           self.train_data = pd.read_csv("../data/train.csv")
           self.test_data = pd.read_csv("../data/test.csv")
   
           self.train_data['is_test'] = 0
           self.test_data['is_test'] = 1
           
           self.concat = pd.concat([self.train_data, self.test_data])
           self.concat = self.concat.rename(columns={'전용면적(㎡)':'전용면적'})
   
       # 결측치, 이상치 처리 예시
       def feature_selection(self):
           print('start feature selection')
           # 결측치 탐색 및 보간
           self.concat['등기신청일자'] = self.concat['등기신청일자'].replace(' ', np.nan)
           self.concat['거래유형'] = self.concat['거래유형'].replace('-', np.nan)
           self.concat['중개사소재지'] = self.concat['중개사소재지'].replace('-', np.nan)
   
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
   
       # 파생 변수 예시
       def feature_engineering(self):
           is_gangnam = []
           for x in concat_select['구'].tolist():
               if x in gangnam :
                   is_gangnam.append(1)
               else :
                   is_gangnam.append(0)
   
           # 파생변수를 하나 만듭니다.
           concat_select['강남여부'] = is_gangnam
       self.concat = concat_select
   ```
 - 주의사항
   - BasePreprocess 상속 필수
     ```
     class BaselinePreprocess(BasePreprocess):
     ```
   - init : 이 구문 필수로 넣어야함
     ```
     self.train_data['is_test'] = 0
     self.test_data['is_test'] = 1
        
     self.concat = pd.concat([self.train_data, self.test_data])
     ```
   - feature_selection, feature_engineering 함수 마지막 부분에 concat Update 필수
     ```
     self.concat = concat_select
     ```
### 2. Experiments
 - 작업 위치
   ```
   ├── experiments
   │   │── 작업자 A
   │   │   └── exp_1
   │   │   │   ├── train.sh
   │   │   │   └── test.sh
   │   │   └── exp_2
   │   │       ├── train.sh
   │   │       └── test.sh    
   ```
   
 - bash 작성
   ```
   python main.py \
    --data=[본인 작업 폴더].[preprocess 파일명].[preprocess 클래스명] \
    --model=lmw.tutorial.TutorialModel \ # Baseline 기본 모델, RandomForest
    --data_root_path=[data_root path] \
    --model_name=[학습된 모델 이름] \
    --test_result_file=[csv 저장 이름] \
    --mode=[train or test]
   ```
   
 - bash 실행
   ```
   cd [project folder]
   bash ./experiments/[작업 폴더]/[실험 이름]/train.sh or test.sh
   # ex) bash ./experiments/lmw/tutorial/train.sh
   ```

## 3. Data descrption

### Dataset overview
 - 주요 데이터는 .csv 형태로 제공되며, test.csv의 9,272개 서울시 아파트의 각 시점에서의 실거래가 거래금액(만원)을 예측하는 것이 목표입니다.
 - 학습 데이터는 아래와 같이 1,118,822개이며, 예측해야 할 거래금액(target)을 포함한 52개의 아파트의 정보에 대한 변수와 거래시점에 대한 변수가 주어집니다.
   
1. **학습 데이터(train.csv)**  
   - 약 110만 건의 서울시 아파트 실거래 내역 포함 (2007.01 ~ 2023.06)
   - 거래금액(target) 및 아파트 위치, 면적, 층수, 건축년도, 시군구 등의 52개 feature 포함

2. **테스트 데이터(test.csv)**  
   - 약 9,272건, target(거래금액)은 비공개
   - 모델은 해당 데이터에 대한 예측값을 제출해야 함

3. **지하철 정보(subway_feature.csv)**  
   - 위도, 경도, 거리, 환승역 여부 등 포함
   - 아파트와의 거리 기반 피처 엔지니어링 가능

4. **버스 정류장 정보(bus_feature.csv)**  
   - 지하철과 동일하게 위치 기반 정보 제공

### 3-1. 주소 기반 좌표 추출 및 거리 피처 생성

모델 성능 향상을 위해 아파트 주소를 Kakao 및 Naver 지도 API를 활용해 위도/경도 좌표로 변환했습니다. 이는 지하철역, 버스정류장 등과의 거리 계산에 활용되며, 입지 특성을 반영하는 핵심 피처로 사용됩니다.

- **주소 정제**: `시군구`, `도로명`, `본번`, `부번`을 결합한 `주소ID` 생성
- **좌표 변환 로직**:
  - Kakao API 우선 호출 → 실패 시 Naver API로 fallback
  - 지번 주소와 도로명 주소를 모두 시도하여 최대한 정확도 확보
- **API 호출 최소화**:
  - 변환된 좌표를 `pkl` 캐시 파일로 저장해 중복 요청 방지 및 재현성 확보
- **좌표 병합**:
  - 좌표 정보(`좌표X`, `좌표Y`)를 원본 데이터에 `주소ID` 기준으로 병합
- **활용 목적**:
  - 추후 지하철/버스/상권 등의 위치 데이터와 연계하여 거리 기반 피처 생성
  - 예: `아파트 ↔ 최근접 지하철역 거리`, `아파트 ↔ 버스정류장 거리`

> 이 작업은 향후 거리 기반 변수(예: 통근 편의성, 자연환경 접근성 등)를 모델에 반영하기 위한 핵심 기반 작업입니다.


#### 사용법 (모듈: `data/jeb/coords_preprocess.py`)

- 실행 전 `.env` 파일이 필요합니다 (`.env` 파일은 포함되어 있지 않아 `.env.example` 파일을 참고해 `.env` 파일을 프로젝트 루트에 생성)
- 사용 예시 (Jupyter 또는 전처리 모듈 내):

```python
from data.jeb.coords_preprocess import add_coordinates_from_address

concat = add_coordinates_from_address(concat)
```


### EDA

 - ![0CA36571-BA7E-48FA-BD1C-A66DCBCE3C8C](https://github.com/user-attachments/assets/11ee5c54-0de5-4044-8547-e1967cd7d201)
 - ![output](https://github.com/user-attachments/assets/f0b9b7d3-9388-42bb-ad3e-d26bb5566424)
  - 강북쪽 거래량이 많아 보임, 가격이 낮을수록 거래량이 많고 가격이 높을수록 거래량이 적음

- ![eda_target_distribution](https://github.com/user-attachments/assets/7bbfa5d7-cc08-4320-a570-0f21d11fff0b)
 -  전처리 과정 중, 타겟 변수를 확인하고 필요에 따라 변환을 결정하기위해 생성함. 왼쪽 그래프는 원본데이터의 아파트 값들, 오른쪽 그래프는 로그 변환을 적용한 값들의 분포를 보여줌
- ![correlation_heatmap](https://github.com/user-attachments/assets/532d5326-16de-46ca-9d52-adfbbab0f48d)
 - 수치형 피처 간 상관관계 히트맵, 수치형 피처들 간의 선형적인 상관관계를 하눈에 파악하기 위해 생성

## 4. Modeling

### Model Code

## 5. Result

### Leader Board

![3E9B23A0-D81F-4519-BC3C-75975FD33365](https://github.com/user-attachments/assets/acc3820d-ad89-4a09-8174-24fa21b69615)


## 6. Getting Start

### Clone the Repository

```
git clone git@github.com:AIBootcamp13/upstage-ml-regression-ml_1.git
cd upstage-ml-regression-ml_1
```

### Environment Setting

```
virtualenv ./envs/[envname]
source ./envs/[envname]/bin/activate
```

### Install Packages

```
pip install -r requirements.txt
```

### Train

```
bash ./experiments/lmw/baseline/train.sh
```

### Test

```
bash ./experiments/lmw/baseline/test.sh
```

## 7. etc
### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
