import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints_from_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])  # (x, y, z) 정규화 좌표
        return np.array(keypoints)  # shape: (99,)
    else:
        return None

# 예시: 두 클래스 폴더가 있다고 가정
data = []
labels = []
class_folders = {'kickboard':1, 'walking':0}

for class_name, label in class_folders.items():
    folder_path = f'./{class_name}'
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        keypoints = extract_keypoints_from_image(img_path)
        if keypoints is not None:
            data.append(keypoints)
            labels.append(label)
        else:
            print("관절 데이터 추출 실패(사람이 인식되지 않음)")

X = np.array(data)      # shape: (샘플수, 99)
y = np.array(labels)    # shape: (샘플수,)
np.save('mediaPipe/pose_X.npy', X)
np.save('mediaPipe/pose_y.npy', y)
