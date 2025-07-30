import os
import cv2
import mediapipe as mp

# 1. MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

# 2. 분석할 이미지 경로 지정
img_path = 'testData/walk.jpg'  # 실제 파일명으로 변경

# 3. 이미지 불러오기
img = cv2.imread(img_path)
if img is None:
    print("이미지를 불러올 수 없습니다. 경로/파일명을 확인하세요.")
    exit()

# 4. BGR → RGB 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 5. 포즈 추정 실행
results = pose.process(img_rgb)

# 6. 관절 시각화
if results.pose_landmarks:
    annotated_img = img.copy()
    mp_drawing.draw_landmarks(
        annotated_img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=3),  # 점 스타일
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=2)   # 선 스타일
    )

    # 7. 이미지 저장용 폴더 생성
    os.makedirs("output", exist_ok=True)

    # 8. 저장 파일명 생성 (원본 파일 이름 기반)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join("output", f"{filename}_pose.jpg")

    # 9. 이미지 저장
    cv2.imwrite(output_path, annotated_img)
    print(f"[INFO] 시각화된 이미지가 저장되었습니다: {output_path}")

    # (선택) 이미지 보기
    resized_img = cv2.resize(annotated_img, (800, 600))
    cv2.imshow('Pose Landmarks', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("관절 데이터 추출 실패(사람이 인식되지 않음)")
