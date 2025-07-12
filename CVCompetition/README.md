# CV Competition

## 1. Team

| <img src="https://github.com/user-attachments/assets/9d85bdb8-9f14-442c-bc1b-258a04f8e2ec" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/8afdcf23-9bbd-4a64-86c1-b39af8b825cf" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/a2932e92-0673-4385-96b8-e7d16861def7" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/329f4b4e-dff8-4eb8-9c51-784edf4d0a4b" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/40832770-57ba-4ba3-ae8f-8c2d2ac9847e" width="200" height="200" /> |
|:--:|:--:|:--:|:--:|:--:|
| [**이민우**](https://github.com/UpstageAILab) | [**조선미**](https://github.com/UpstageAILab) | [**이준석**](https://github.com/UpstageAILab) | [**이나경**](https://github.com/UpstageAILab) | [**황준엽**](https://github.com/UpstageAILab) |
| 팀장 / EDA, 전처리, 모델링 및 실험 | EDA, Modeling | EDA, Modeling | EDA, Modeling | EDA, Modeling |

### 2. Overview
이번 대회는 Computer vision 분야에서 핵심적인 태스크인 이미지 분류를 중심으로 진행되었습니다. 이미지 분류는 주어진 이미지를 사전에 정의된 여러 클래스 중 하나로 분류하는 작업이며, 이는 의료, 보안, 패션, 금융 등 다양한 산업 분야에서 광범위하게 활용되고 있습니다. 이번 대회에서는 특히 **문서 이미지를 다루며**, 실제 현업에서 사용되는 다양한 형태의 문서들을 분류하는 데 초점을 맞추었습니다.
 - 문서 데이터는 의료, 금융, 보험, 물류 등에서 매우 흔하며, 대규모 자동화를 위한 문서 타입 분류 시스템 구축의 수요가 매우 큽니다.

대회에 사용된 데이터는 총 17종의 문서 유형으로 구성되어 있으며, 학습 이미지: 1,570장 평가 이미지: 3,140장 **F1 score 지표**를 통해 모델 성능을 평가합니다.

### 3. Dataset Overview

대회에서 제공된 데이터는 **문서 타입 분류**를 위한 이미지 및 메타 정보로 구성되어 있습니다.  
학습 데이터는 총 **1,570장**, 평가 데이터는 총 **3,140장**으로 구성되어 있으며, 주요 파일 구성은 다음과 같습니다.

#### 학습 데이터 (`train/`, `train.csv`, `meta.csv`)

##### `train/` 폴더
- 총 **1,570장의 이미지**가 저장된 디렉토리
- 각 이미지 파일명은 `train.csv`의 `ID` 컬럼과 일치

##### `train.csv`
- 총 **1,570개 샘플**
- 각 이미지의 정답 클래스 정보를 포함

| 컬럼명 | 설명 |
|--------|------|
| `ID` | 이미지 파일명 (e.g., `0001f9be12bd5e52.jpg`) |
| `target` | 정답 클래스 번호 (0 ~ 16) |

##### `meta.csv`
- 총 **17개의 클래스** 정보를 포함
- 클래스 번호와 이름 매핑

| 컬럼명 | 설명 |
|--------|------|
| `target` | 클래스 번호 |
| `class_name` | 클래스 이름 (예: `account_number`, `application_form` 등) |

---

#### 평가 데이터 (`test/`, `sample_submission.csv`)

##### `test/` 폴더
- 총 **3,140장의 문서 이미지**가 저장된 디렉토리
- `train`과 동일한 구조이나 정답 라벨은 제공되지 않음

##### `sample_submission.csv`
- 제출용 양식 (총 3,140행)
- 예측 결과를 `target` 컬럼에 입력하여 제출

| 컬럼명 | 설명 |
|--------|------|
| `ID` | 테스트 이미지의 파일명 |
| `target` | 예측 클래스 번호 입력 (초기값은 모두 0) |

---

### 4. EDA
- 

### 5. Data Augmentation
- 

### 6. Modeling
- 

### 7. 결과 분석
 - confusion matrix, validation error를 통해 모델의 3, 7 클래스 혼동을 파악하였습니다.
<img width="400" height="1800" alt="cm_epoch_030" src="https://github.com/user-attachments/assets/7c870a35-794d-43e2-ab33-4c968c108249" />
<img width="400" height="502" alt="스크린샷 2025-07-12 오후 8 32 07" src="https://github.com/user-attachments/assets/387d62c6-f326-4bad-873a-2335c3b69749" />

### 8. 결과 분석을 통한 재학습 전략
 - Hard Negative Mining
 - 2-Stage Classifier 모델링

### 8. 설치 및 실험
 - 코드 준비
   ```
   git clone https://github.com/LMWoo/UpstageAILab_13.git
   cd UpstageAILab_13
   cd CVCompetition
   ```

 - env 설정
   * env.template를 **삭제가 아닌 복사** -> env 변경
   * WANDB_API_KEY=본인 wandb api 키 입력

 - 데이터 전처리
   ```
   python tools/preprocess_augraphy.py
   ```
   
 - 학습
   ```
   python tools/coarse_train.py
   ```
   
 - 테스트
   ```
   python tools/test.py
   ```
