import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,         # 실시간 영상이므로 False
    model_complexity=1,              # 0: Lite, 1: Full, 2: Heavy
    enable_segmentation=False,       # 배경 분할 사용 여부
    min_detection_confidence=0.5,    # 검출 신뢰도
    min_tracking_confidence=0.5      # 추적 신뢰도
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        # 랜드마크가 감지되면 그리기
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

cap.release()
cv2.destroyAllWindows()
