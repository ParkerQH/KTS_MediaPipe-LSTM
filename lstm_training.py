import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
X = np.load('mediaPipe/pose_X.npy')   # shape: (샘플수, 99)
y = np.load('mediaPipe/pose_Y.npy')   # shape: (샘플수,)

# 2. LSTM 입력 형태로 변환 (시퀀스 길이=1)
X = X.reshape(-1, 1, 99)    # (샘플수, 1, 99)

# 3. 라벨 원-핫 인코딩
y_cat = to_categorical(y)   # (샘플수, 클래스수)

# 4. 학습/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 5. LSTM 모델 설계
model = Sequential([
    LSTM(32, input_shape=(1, 99)),
    Dense(16, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 6. 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val)
)

# 7. 모델 저장
model.save('lstm_model/kickboard_lstm_model.keras')  # 원하는 파일명으로 저장

# 8. 모델 평가
loss, acc = model.evaluate(X_val, y_val)
print(f'검증 정확도: {acc:.4f}')

