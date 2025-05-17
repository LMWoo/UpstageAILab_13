# 아파트 실거래 가격 예측
## 5G.AI

| ![이민우-photoaidcom-cropped (2)](https://github.com/user-attachments/assets/b71c7815-e0c7-407f-b593-121e11c61d05) | ![박진일2-photoaidcom-cropped (1)](https://github.com/user-attachments/assets/551cb3b6-c019-4e50-a35b-1843d3e474a3) | ![정예규-photoaidcom-cropped](https://github.com/user-attachments/assets/7127ed78-2c2e-4225-a92d-149d6452d30c) | ![조은별-photoaidcom-cropped](https://github.com/user-attachments/assets/781faa65-c309-49ee-b58a-9cc08b13b69c) | ![조재형-photoaidcom-cropped](https://github.com/user-attachments/assets/c7d4d78c-36ae-4fb1-bbd4-105ee9d722cc) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [이민우](https://github.com/LMWoo) |  [박진일](https://github.com/r0feynman)  |  [정예규](https://github.com/yegyu)   | [조은별](https://github.com/eunbyul2) | [조재형](https://github.com/Bitefriend)  |
| 팀장 / 파이프라인 설계 및 통합, EDA 분석 총괄 | 거래 패턴 분석 기반 EDA 및 핵심 변수 도출 | 모델링 / 피처 활용 최적화 및 성능 평가 | 거리 기반 피처 엔지니어링 및 분석 결과 정리 | 데이터 정제 로직 및 시각화 기반 인사이트 제공 |

## 0. Overview
### Environment
- _Write Development environment_

### Requirements
- _Write Requirements_

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

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

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
