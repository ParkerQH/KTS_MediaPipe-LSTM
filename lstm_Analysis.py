import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# MediaPipe Pose 모델
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 관절 데이터 추출
def extract_keypoints_from_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    else:
        return None

# 사진 속 사람의 자세 분석
def lstm_Analysis_per1(image):
    check_keypoints = extract_keypoints_from_image(image)

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_PATH, "lstm_model")

    # LSTM 모델 불러오기 및 예측
    model = load_model(os.path.join(MODEL_PATH,'best_3000_50.keras'))

    if check_keypoints is not None:
        check_input = check_keypoints.reshape(1, 1, 99)
        pred = model.predict(check_input)
        pred_label = np.argmax(pred)
        print('✅ 킥보드 탑승자' if pred_label == 1 else '✅ 보행자')
        return pred_label
    else:
        print("관절 데이터 추출 실패(사람이 인식되지 않음)")
        return None
