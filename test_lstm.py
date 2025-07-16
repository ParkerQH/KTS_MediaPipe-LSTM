import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# MediaPipe Pose 모델
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# 관절 데이터 추출
def extract_keypoints_from_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    else:
        return None

# 3. 테스트할 이미지 파일 경로
img_path = 'testData/image14.jpg'
img = cv2.imread(img_path)
if img is None:
    print("이미지를 불러올 수 없습니다. 경로/파일명을 확인하세요.")
else:
    print("이미지 불러오기 성공!")
    resized_img = cv2.resize(img, (640, 480))
    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_keypoints = extract_keypoints_from_image(img_path)

# 4. 모델 불러오기 및 예측
model = load_model('lstm_model/best_1500.keras')

if test_keypoints is not None:
    test_input = test_keypoints.reshape(1, 1, 99)
    pred = model.predict(test_input)
    pred_label = np.argmax(pred)
    print('킥보드 탑승자' if pred_label == 1 else '보행자')
else:
    print("관절 데이터 추출 실패(사람이 인식되지 않음)")
