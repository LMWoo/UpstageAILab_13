# UpstageAILab ML 경진대회

## 프로젝트 구성
### 1. Baseline 구조
기본적인 데이터 전처리 및 학습 파이프라인 구축


#### EDA (Exploratory Data Analysis)
 - 데이터 형식 : csv형태 정형 데이터 
 - 결측치 확인 (df.info(), df.describe(), df.isnull().sum())
 - 범주형, 숫자형 분리 

#### Feature Selection
 - 결측치 90% 이상인 특징 제거

#### Feature Engineering
 - 숫자형 결측치 선형 보간, 범주형 결측치 Label encoding 적용
 - 계약년월 → 계약년, 계약월로 파생

#### Model Selection
트리 기반 모델 : RandomForest 사용, Overfitting 가능성 낮음

#### Model Train & Validation
 - sklearn의 RandomForestRegressor사용 하여 학습
   - 학습 파라미터 : n_estimators=100, criterion='squared_error'
   - data : 8:2로 train, valiation data random split 

#### Feature Importance

#### Test
 - 학습된 모델로 예측 수행

### 3. EDA

### 4. Feature Selection

### 5. Feature Engineering

### 6. Model Selection

### 7. Model Training & Validation

### 8. Model Test

### 9. Experiment Notes
