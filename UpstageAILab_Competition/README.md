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

 - 작업 위치
   ```
   ├── data
   │   ├── base.py
   │   │── 작업자 A
   │   │   ├── Data_1_process.py
   ```

 - 함수 설명
   - BasePreprocess 상속 필수
     ```
     class Data_1_Preprocess(BasePreprocess):
     ```
   - feature_selection 함수 : EDA 이후 원본 데이터에서 사용할 변수 고르는 함수
   - feature_cleaning 함수 : 이상치, 결측치 처리 구현 함수
   - feature_engineering 함수 : 파생 변수 구현 함수
   - feature_encoding 함수 : 범주형 변수 인코딩 함수
   - get_preprocessed_data 필수 구현, main에서 이 함수 호출해서 split data 진행
     ```
     def get_preprocessed_data(self):
        return self.preprocessed_data
     ```
     
 - 실제 개발 예시 : 길어서 일부 생략
   
   - BasePreprocess 클래스
     ```
      # Base Class
      class BasePreprocess:
          def __init__(self):
              self.train_data = None
              self.test_data = None
              self.concat = None
      
              self.basic_columns = ['target', 'is_test'] 
              self.label_encoders = {}
              self.preprocessed_data = {'X_train': None, 'X_test': None}
      
          @final
          def preprocess_data(self):
              self.feature_selection()
              self.feature_cleaning()
              self.feature_engineering()
              self.feature_encoding()
          
          @abstractmethod
          def feature_selection(self):
              pass
      
          @abstractmethod
          def feature_cleaning(self):
              pass
      
          @abstractmethod
          def feature_engineering(self):
              pass
      
          @abstractmethod
          def feature_encoding(self):
              pass
      
          @abstractmethod
          def get_preprocessed_data(self):
              return self.preprocessed_data
     ```
     
   - [Data_1_Preprocess 클래스](/UpstageAILab_Competition/data/lmw/Data_1_dataprocess.py) 

### 2. Models

 - 작업 위치
   ```
   ├── models
   │   ├── base.py
   │   │── 작업자 A
   │   │   ├── lightGBM_model.py
   ```

 - 함수 설명
   - BasePreprocess 상속 필수
     ```
     class Data_1_Preprocess(BasePreprocess):
     ```
   - feature_selection 함수 : EDA 이후 원본 데이터에서 사용할 변수 고르는 함수
   - feature_cleaning 함수 : 이상치, 결측치 처리 구현 함수
   - feature_engineering 함수 : 파생 변수 구현 함수
   - feature_encoding 함수 : 범주형 변수 인코딩 함수
   - get_preprocessed_data 필수 구현, main에서 이 함수 호출해서 split data 진행
     ```
     def get_preprocessed_data(self):
        return self.preprocessed_data
     ```
     
 - 실제 개발 예시 : 길어서 일부 생략
   
   - BaseModel 클래스
     ```
      # Base Class
      class BaseModel:
          def __init__(self, X_train, X_val, Y_train, Y_val, X_test):
              self.X_train = X_train
              self.X_val = X_val
              self.Y_train = Y_train
              self.Y_val = Y_val
              self.X_test = X_test
              pass
      
          @abstractmethod
          def train(self):
              pass
      
          @abstractmethod
          def validation(self):
              pass
      
          @abstractmethod
          def test(self):
              pass
      
          @abstractmethod
          def analysis_validation(self, save_root_path, data_preprocessor):
              pass
      
          @abstractmethod
          def save_model(self, save_path):
              pass
      
          @abstractmethod
          def load_model(self, load_path):
              pass
     ```
     
   - [LightGBMModel 클래스](/UpstageAILab_Competition/models/lmw/lightGBM_model.py) 

### 3. Experiments
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
    --model=[본인 작업 폴더].[model 파일명].[model 클래스명] \
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


### EDA
 - [EDA 결과 정리](https://steel-single-800.notion.site/EDA-1eea5b28c3bb8097a29dd7aed14a26d1)

## 4. Modeling

 - Model : 트리기반 Random Forest, LightGBM 사용
 - Model 사용 이유 : EDA 결과로 데이터 결측치와 이상치가 많이 보여 트리기반 모델로 선정
 - Ensemble : k-fold cross-validation 기반으로 앙상블 적용

## 5. Result

### Leader Board

![3E9B23A0-D81F-4519-BC3C-75975FD33365](https://github.com/user-attachments/assets/acc3820d-ad89-4a09-8174-24fa21b69615)


## 6. Getting Start

### Clone the Repository

```
git clone git@github.com:LMWoo/MachineLearning.git
cd MachineLearning
cd UpstageAILab_Competition
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
