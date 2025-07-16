import cv2
import mediapipe as mp

# 1. MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True)  # 단일 이미지 분석

# 2. 분석할 이미지 경로 지정
img_path = 'testData/image13.jpg'  # 실제 파일명으로 변경

# 3. 이미지 불러오기
img = cv2.imread(img_path)
if img is None:
    print("이미지를 불러올 수 없습니다. 경로/파일명을 확인하세요.")
    exit()

# 4. BGR → RGB 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 5. 포즈 추정 실행
results = pose.process(img_rgb)

# 6. 관절(랜드마크) 시각화
if results.pose_landmarks:
    # 이미지에 관절 점과 연결선 그리기
    annotated_img = img.copy()
    mp_drawing.draw_landmarks(
        annotated_img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),  # 점 스타일
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)   # 선 스타일
    )
    # 7. 결과 이미지 보기
    resized_img = cv2.resize(annotated_img, (800, 600))  # 창 크기 조절(선택)
    cv2.imshow('Pose Landmarks', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("관절 데이터 추출 실패(사람이 인식되지 않음)")
