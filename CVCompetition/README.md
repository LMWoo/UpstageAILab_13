# CV Competition

## 1. Team

| <img src="https://github.com/user-attachments/assets/9d85bdb8-9f14-442c-bc1b-258a04f8e2ec" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/8afdcf23-9bbd-4a64-86c1-b39af8b825cf" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/a2932e92-0673-4385-96b8-e7d16861def7" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/329f4b4e-dff8-4eb8-9c51-784edf4d0a4b" width="200" height="200" /> | <img src="https://github.com/user-attachments/assets/40832770-57ba-4ba3-ae8f-8c2d2ac9847e" width="200" height="200" /> |
|:--:|:--:|:--:|:--:|:--:|
| [**이민우**](https://github.com/UpstageAILab) | [**조선미**](https://github.com/UpstageAILab) | [**이준석**](https://github.com/UpstageAILab) | [**이나경**](https://github.com/UpstageAILab) | [**황준엽**](https://github.com/UpstageAILab) |
| 팀장 / EDA, 전처리, 모델링 및 실험 | EDA, Modeling | EDA, Modeling | EDA, Modeling | EDA, Modeling |

### 2. Overview
 -  이번 대회는 Computer vision 분야에서 핵심적인 태스크인 이미지 분류를 중심으로 진행되었습니다. 이미지 분류는 주어진 이미지를 사전에 정의된 여러 클래스 중 하나로 분류하는 작업이며, 이는 의료, 보안, 패션, 금융 등 다양한 산업 분야에서 광범위하게 활용되고 있습니다.
 -  이번 대회에서는 특히 **문서 이미지를 다루며**, 실제 현업에서 사용되는 다양한 형태의 문서들을 분류하는 데 초점을 맞추었습니다.
    - 문서 데이터는 의료, 금융, 보험, 물류 등에서 매우 흔하며, 대규모 자동화를 위한 문서 타입 분류 시스템 구축의 수요가 매우 큽니다.

### 3. 대회 참여 목표
 - 다양한 딥러닝 기반 Backbone 모델을 직접 적용하고 성능을 비교해본다.
 - 실전 환경에 가까운 문서 분류 문제를 다룸으로써, **현업에 적용 가능한 모델 개발 능력을 강화한다.**

### 4. 대회 결과
 - Private : 1396 팀중 **168등**
   <img width="1207" alt="스크린샷 2025-06-21 오후 7 16 06" src="https://github.com/user-attachments/assets/3c340a77-62e7-4af2-8783-ba8c057bc5b2" />

### 5. 모델 실험 결과
 * data augumentation : Center Crop + Vertical Flip + Rotation + ColorJitter + Contrast + GaussianBlur + Erasing + RandAugment (모든 실험 동일)

 | 실험 | Pretrained model | Loss | LR Scheduler | Optimizer | 기타 기법 | Score (Log Loss) |
 |-----------|--------------------|-------------|-------------------|------------|------------------------|------------------|
 | 1 | ResNet50 (torchvision)   | CrossEntropy | -                 | AdamW     | -                             | **0.319**        |
 | 2 | ResNet50 (timm)          | CrossEntropy | -                 | AdamW     | -                             | **0.317**        |
 | 3 | ConvNeXt (timm)          | CrossEntropy | -                 | AdamW     | -                             | **0.251**        |
 | 4 | ConvNeXt (timm)          | Focal Loss   | CosineAnnealing   | AdamW     | -                             | **0.195**        |
 | 5 | ConvNeXt (timm)          | Focal Loss   | CosineAnnealing   | AdamW     | EMA                           | **0.185 (최종 제출)** |
 | 6 | ConvNeXt (timm)          | Focal Loss   | CosineAnnealing   | AdamW     | EMA, Mixup                    | **0.183 (대회 종료 후 추가 실험)** |
 | 7 | ConvNeXt (timm)          | Focal Loss   | CosineAnnealing   | AdamW     | EMA, Mixup, kfold Ensemble    | **0.172 (대회 종료 후 추가 실험)*** |

### 6. 설치 및 실험
 - 코드 준비
   ```
   git clone https://github.com/LMWoo/HectoAIChallenge.git
   cd HectoAIChallenge
   ```

 - 데이터 준비
   * [해당 사이트](https://dacon.io/competitions/official/236493/data)에서 데이터 받은 후 압축 해제
   * 받은 데이터 폴더 이름을 data로 변경 후 -> HectoAIChallenge/data로 이동

 - env 설정
   * env.template를 **삭제가 아닌 복사** -> env 변경
   * WANDB_API_KEY=본인 wandb api 키 입력

 - docker 및 container 설치
   ```
   docker compose up
   ```

 - 실행
   ```
   docker exec -it hecto-exp-container bash
   bash ./src/experiment_EMA_KFOLD_ENSEMBLE_FOCAL_LOSS_early_stopping_freeze_adamw_timm_convnext.sh
   ```
   
 - 기타 사항
   *  GPU 부족시 utils/utils.py에서 CFG['BATCH_SIZE'] 조절
   
### 7. 대회 후기
 - **Pretrained 모델 변경(ResNet50 → ConvNeXt)** 자체도 성능에 영향을 주는 것 뿐만 아니라 **Optimizer, Scheduler, Loss 조합**도 모델 성능 향상에 중요한 요소라는 점을 경험

 - 대회를 [Upstage MLOps](https://github.com/LMWoo/UpstageAILab_13/tree/master/MLOps) 프로젝트와 병행해서 진행하다 보니, 실험 결과를 체계적으로 기록하고 관리하지 못한 점과 TTA, Ensemble을 깊이 있게 다루지 못한 것이 아쉬움으로 남음
   - 초반에 wandb등을 사용하였으나 체계적인 기록없이 (val loss만 기록)사용하여 나중에 분석을 제대로 하지 못함, 후반에 confusion matrix 등을 도입하였으나 활용은 못하였고 다음 대회 때 다양한 분석 기법을 도입해 실험을 할 계획

 - TTA(Test-Time Augmentation)를 너무 단순하게 좌우 반전만 적용해, 추론 성능 향상이 거의 없었음
   - 다음 대회에서는 rotation, crop, scaling, blur 등 다양한 TTA 조합을 적용해 좀 더 강건한 예측 성능 확보를 목표로 할 예정

 - 실험 종류가 많아지면서 이전 실험에서 사용했던 파라미터들을 체계적으로 관리할 필요성을 절실히 느낌
   - 이를 해결하기 위해 hydra를 도입하여 복잡한 코드 간소화와 config기반 실험 관리 체계를 도입하여 실험할 예정


 - **실험 기록이 제일 중요**

### 8. 기타 대회 작업 내용
 - [EDA 작업 내용](./notebooks/EDA.ipynb)
 - [Augmentation 시각화 코드](./notebooks/Augmentation.ipynb)
 - Augmentation 시각화 이미지 예시
   <img width="707" alt="스크린샷 2025-06-26 오후 9 06 37" src="https://github.com/user-attachments/assets/283db780-e231-4d1d-a46a-4e00026013c9" />

### 9. Confusion Matrix를 이용한 분석 예시 (이 대회 이후 할 예정)
   * epoch 10에서 confusion matrix : 대각선 이외에 파란색 점들이 흩어져 있음, 오분류 다수 존재
