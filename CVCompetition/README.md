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
- 정량 분석 (Quantitative Analysis)

| Brightness_Train | Brightness_Test |
|:------------:|:-----------:|
|<img width="400" height="292" alt="train_bright" src="https://github.com/user-attachments/assets/ecab8923-bc55-4856-8cca-4001d8d0ee75" />|<img width="400" height="291" alt="test_bright" src="https://github.com/user-attachments/assets/6a47de26-e231-4b3f-a322-43b49043648f" />|


| Contrast_Train | Contrast_Test |
|:--------------:|:-------------:|
|<img width="400" height="296" alt="train_contrast" src="https://github.com/user-attachments/assets/eb2337c3-1ad4-48bb-b9b7-f4838dc13163" />|<img width="400" height="300" alt="test_contrast" src="https://github.com/user-attachments/assets/7e81132e-ee4a-43cd-8629-c44571aae3e5" />
|

| Blurriness_Train | Blurriness_Test |
|:--------------:|:-------------:|
|<img width="400" height="292" alt="train_blur" src="https://github.com/user-attachments/assets/c9584795-9d10-4f9a-9ecf-7cd585b391e0" /> |<img width="400" height="292" alt="test_blur" src="https://github.com/user-attachments/assets/94400ed7-bb8f-4f9a-945e-c40b4c2d6a2a" />
|


| Noise_Train | Noise_Test |
|:--------------:|:-------------:|
|<img width="400" height="299" alt="train_noise" src="https://github.com/user-attachments/assets/9cd20d7d-b558-4b68-8d63-b2ce38277691" /> |<img width="400" height="298" alt="test_noise" src="https://github.com/user-attachments/assets/2347866a-d753-4ad5-af3a-129fca606930" />
|

- 정성 분석 (Qualitative Analysis)
  - Train 데이터는 Rotation이 거의 90, 180도인 반면 Test 데이터는 0~180까지 아주 다양하였습니다. -> Augmentation에 Rotation 전략을 사용하였습니다.
  - Test 데이터에 회전 뿐만 아니라 Vertical, Horizon으로 Flip 된 이미지가 보였습니다. -> Augmentation에 Flip 전략을 사용하였습니다.


### 5. Data Augmentation
- 주요 증강 기법 : 데이터 정량, 정성 분석을 통해 Train/Test셋 간의 분포 차이를 보완하기 위한 다양한 증강 기법을 도입하였습니다.
   - 기하학적 변형 : Scaling, Translate, Rotate, Flip 등을 사용하여 다양한 기하학적 변형으로 데이터 증강을 하였습니다.
   - 색상 변형 : ColorJitter를 사용하여 이미지의 밝기, 색상을 증강하였습니다.
   - 노이즈 변형 : Gaussian Blur를 사용하여 Test 데이터의 흐림 및 노이즈 특성을 모방하였습니다.

 - 추가 증강 기법
   - 문서 이미지의 특화된 증강 라이브러리인 Augraphy를 사용하였습니다.
   - Aguraphy는 처리속도 이슈로 Offline 증강을 사용하였습니다.

  - <img width="942" height="718" alt="스크린샷 2025-07-12 오후 9 58 58" src="https://github.com/user-attachments/assets/90af84b5-1725-488a-8f9a-279c7e995370" />


### 6. Modeling & Train
- ConvNext
- AdamW
- Cosine Aannealing
- EMA

### 7. 성능 고도화
 - Ensemble (weighted, seed, k fold etc)
 - TTA 기법
 - mixup, cutmix
 - augraphy 오프라인 증강
 - Online Hard Negative Mining

### 7. 결과 분석
 - confusion matrix, validation error를 통해 모델의 3과 7 클래스, 4와 14 클래스간의 혼동을 파악하였습니다.
<img width="400" height="1800" alt="cm_epoch_030" src="https://github.com/user-attachments/assets/7c870a35-794d-43e2-ab33-4c968c108249" />
<img width="400" height="502" alt="스크린샷 2025-07-12 오후 8 32 07" src="https://github.com/user-attachments/assets/387d62c6-f326-4bad-873a-2335c3b69749" />

### 8. 결과 분석을 통한 재학습 전략
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

 - 다중 학습
   ```
   python tools/coarse_train.py -m model=convnext,resnet50
   ```
