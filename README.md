# 🧍‍♂️ KTS_MediaPipe-LSTM - 전동킥보드 탑승자 판별 시스템

`MediaPipe` 기반 관절 추출 + `LSTM` 딥러닝 모델을 활용하여 **사진 속 인물이 킥보드 탑승자인지 보행자인지 자동 분류**하는 프로젝트입니다.  
AI 기반 통합 분석 시스템인 **KTS_AI_Analysis**의 하위 모듈로, **위반 이미지를 정밀 판별**하는 데 활용됩니다.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python"/>
  <img src="https://img.shields.io/badge/MediaPipe-PoseTracking-green"/>
  <img src="https://img.shields.io/badge/LSTM-ActionRecognition-orange"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen"/>
</p>

---

## 📅 프로젝트 정보

- **기간**: 2025.07.07 ~ 2025.07.18
- **개발자**: 박창률
- **주요 모듈**: `mediaPipe.py`, `lstm_training.py`, `lstm_Analysis.py`
- **연계 모듈**: `KTS_AI_Analysis`

---

## 📌 핵심 기능

### 🔎 관절 추출 (MediaPipe)
- `MediaPipe Pose`를 통해 이미지 속 인물의 3D 관절 좌표(x, y, z) 99개 추출
- 학습 시 `kickboard`와 `walking` 두 클래스로 구분된 이미지셋 사용

### 🧠 포즈 인식 (LSTM)
- 관절 좌표를 기반으로 사람의 행위를 시퀀스 입력으로 분석
- 단일 프레임 기반 LSTM 모델 학습 (`shape: (1, 99)`)
- 추론 결과: `0 = 보행자`, `1 = 킥보드 탑승자`

---

## 🧩 주요 모듈 설명

### 📄 mediaPipe.py  
학습용 데이터 전처리 모듈입니다. 각 이미지에서 관절 좌표(99차원)를 추출하고, `kickboard`, `walking` 클래스별 `.npy` 파일을 생성합니다.

### 📄 lstm_training.py  
`.npy`로 저장된 데이터셋을 불러와 LSTM 모델 학습을 진행합니다. 모델은 2-class softmax 분류기로 구성되며, 최종적으로 `.keras` 파일로 저장됩니다.

### 📄 lstm_Analysis.py  
학습된 모델을 불러와 감지 이미지 속 인물의 포즈를 분석하고, 킥보드 탑승자인지 여부를 판별합니다. MediaPipe로 관절을 추출해 모델에 입력하고 예측합니다.

---

## 🖥 실행 예시

```bash
# 학습용 데이터 추출
python mediaPipe.py

# 모델 학습
python lstm_training.py

# 이미지 판별
python lstm_Analysis.py
```

```bash
# 분석 함수 직접 사용 예시
from lstm_Analysis import lstm_Analysis_per1
img = cv2.imread("output_image.jpg")
result = lstm_Analysis_per1(img)
```
>출력: ✅ 킥보드 탑승자 또는 ✅ 보행자

---

## 📦 통합 프로젝트 구조 안내
본 프로젝트는 전체 AI 기반 분석 시스템인 **KTS_AI_Analysis**의 하위 모듈 중 하나입니다.

`KTS_MediaPipe-LSTM`은 `KTS_AI_Analysis` 내에서 다음과 같은 역할을 수행합니다:

- YOLO로 감지된 위반자 이미지에서 관절 추출
- LSTM 모델로 포즈 분석 및 위반자 판단 보조
- 보행자/탑승자 판단을 통해 탑승 위반 여부 검증 수행

👉 전체 파이프라인은 **[KTS_AI_Analysis](https://github.com/ParkerQH/KTS_AI_Analysis)** 리포지토리에서 확인할 수 있습니다.

---

## 💡 향후 개선 아이디어
- 🎥 연속된 프레임 기반 시계열 모델 확장 (CNN-LSTM 등)
- 📈 학습 데이터 다양성 증가 (다양한 복장, 카메라 각도)

---

## 📬 Contact
- **개발자**: 박창률  
- **GitHub**: [ParkerQH](https://github.com/ParkerQH)
