import os
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Параметры
LIP_INDICES = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415
]))
CLASSES = ['yes', 'no', 'hello', 'thanks', 'sorry']
SEQUENCE_LENGTH = 20
IMG_SIZE = 64
DATA_PATH = 'data/lips'
MODEL_PATH = 'models/lipnet_model.h5'

def extract_mouth_sequence(image, face_landmarks):
    h, w, _ = image.shape
    lip_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LIP_INDICES]
    x_coords, y_coords = zip(*lip_points)
    min_x, max_x = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
    min_y, max_y = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)
    mouth_img = image[min_y:max_y, min_x:max_x]
    if mouth_img.size == 0:
        return None
    return cv2.resize(mouth_img, (IMG_SIZE, IMG_SIZE))

def predict_word(model, sequence):
    sequence = np.expand_dims(sequence, axis=0)  # (1, T, H, W, C)
    prediction = model.predict(sequence, verbose=0)[0]
    return CLASSES[np.argmax(prediction)], max(prediction)

def create_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(CLASSES), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    X, y = [], []
    for label in CLASSES:
        label_dir = os.path.join(DATA_PATH, label)
        if not os.path.exists(label_dir):
            print(f"Каталог для класса '{label}' не найден, пропускаем.")
            continue
        for sequence_folder in os.listdir(label_dir):
            sequence_path = os.path.join(label_dir, sequence_folder)
            frames = []
            for i in range(SEQUENCE_LENGTH):
                img_path = os.path.join(sequence_path, f"{i}.jpg")
                if not os.path.exists(img_path):
                    break
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                frames.append(img)
            if len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y.append(CLASSES.index(label))
    print(f"Загружено {len(X)} последовательностей.")
    return train_test_split(np.array(X), to_categorical(y, num_classes=len(CLASSES)), test_size=0.2, random_state=42)

if __name__ == '__main__':
    print("Старт программы")
    if os.path.exists(MODEL_PATH):
        print("Загружаем модель...")
        model = load_model(MODEL_PATH)
        print("Модель загружена.")
    else:
        print("Обучаем модель с нуля...")
        X_train, X_test, y_train, y_test = load_data()
        model = create_model()
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8)
        loss, acc = model.evaluate(X_test, y_test)
        print(f"Точность модели на тесте: {acc*100:.2f}%")
        os.makedirs('models', exist_ok=True)
        model.save(MODEL_PATH)
        print("Модель сохранена.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    cap = cv2.VideoCapture(0)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    predicted_word = "..."
    last_word = None
    last_prediction_time = 0
    prediction_interval = 2.0  # Задержка между словами в секундах

    print("Запуск камеры. Нажмите 'q' для выхода.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        image = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            mouth_img = extract_mouth_sequence(image, results.multi_face_landmarks[0])
            if mouth_img is not None:
                frame_buffer.append(mouth_img)

            if len(frame_buffer) == SEQUENCE_LENGTH and (time.time() - last_prediction_time) > prediction_interval:
                input_sequence = np.array(frame_buffer) / 255.0
                word, confidence = predict_word(model, input_sequence)
                if word != last_word or confidence < 0.75:
                    predicted_word = word
                    last_word = word
                    last_prediction_time = time.time()
                    print(f"Предсказано слово: {word} с уверенностью {confidence:.2f}")
                frame_buffer.clear()
        else:
            predicted_word = "..."

        cv2.putText(image, f"Word: {predicted_word}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Lip Reading AI", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Выход из программы...")
            break

    cap.release()
    cv2.destroyAllWindows()
