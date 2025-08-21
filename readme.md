# CLIP 모델을 이용한 이미지-텍스트 분류
  - <*2025 Samsung Collegiate Programming Challenge* : AI 챌린지>본선 진출 참가/제출 코드
## 프로젝트 개요

본 프로젝트는 OpenAI의 CLIP(Contrastive Language-Image Pre-Training) 모델을 활용하여 주어진 이미지와 텍스트 설명을 매칭하는 과제를 수행합니다. `train.py`로 모델을 학습하고, `clip_inference_learning.py`를 통해 추론 및 제출 파일을 생성합니다.

## 프로젝트 구조

```
.
├── clip_finetuned/         # 파인튜닝된 CLIP 모델 파일
├── test_input_images/      # 테스트 이미지
├── train_input_images/     # 학습 이미지
├── baseline_submit_tuned_verified.csv # 검증된 제출 파일
├── baseline_submit_tuned.csv # 튜닝된 제출 파일
├── clip_inference_learning.py # 추론 및 제출 파일 생성 스크립트
├── comparsion.py           # 결과 비교 스크립트
├── test.csv                # 테스트 데이터 정보
├── train.csv               # 학습 데이터 정보
├── train.py                # 모델 학습 스크립트
└── requirements.txt        # 개발 환경 의존성 파일
```

## 실행 방법

### 1. 환경 설정

프로젝트에 필요한 라이브러리는 `requirements.txt` 파일에 명시되어 있습니다. 다음 명령어를 사용하여 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

### 2. 모델 학습

`train.py` 스크립트를 실행하여 CLIP 모델을 학습시킵니다. 학습된 모델은 `clip_finetuned` 디렉토리에 저장됩니다.

```bash
python train.py
```

### 3. 추론 및 제출 파일 생성

`clip_inference_learning.py` 스크립트를 실행하여 학습된 모델로 추론을 수행하고, `baseline_submit_tuned.csv` 형식의 제출 파일을 생성합니다.

```bash
python clip_inference_learning.py
```

## 스크립트 설명

*   **`train.py`**: `train.csv`와 `train_input_images`를 사용하여 CLIP 모델을 파인튜닝합니다.
*   **`clip_inference_learning.py`**: `test.csv`와 `test_input_images`를 읽어 파인튜닝된 `clip_finetuned` 모델로 추론을 수행하고, 상위 5개의 예측 결과를 `baseline_submit_tuned.csv` 파일로 저장합니다.
*   **`comparsion.py`**: `baseline_submit_tuned.csv`와 `baseline_submit_tuned_verified.csv` 파일을 비교하여 성능을 평가하는 스크립트로 보입니다.

*   fork from https://github.com/choi00608/clip_vqa_finetuning.git
