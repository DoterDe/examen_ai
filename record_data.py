import os
import cv2
import numpy as np
import mediapipe as mp

# === Константы ===
CLASSES = ['yes', 'no', 'hello', 'thanks', 'sorry']
SEQUENCE_LENGTH = 20
IMG_SIZE = 64
DATA_PATH = 'data/lips'
LIP_INDICES = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415
]))

# === Подготовка MediaPipe ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# === Функция для вырезания области рта ===
def extract_mouth(image, face_landmarks):
    h, w, _ = image.shape
    points = [(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in LIP_INDICES]
    x, y = zip(*points)
    min_x, max_x = max(min(x) - 10, 0), min(max(x) + 10, w)
    min_y, max_y = max(min(y) - 10, 0), min(max(y) + 10, h)
    mouth = image[min_y:max_y, min_x:max_x]
    return cv2.resize(mouth, (IMG_SIZE, IMG_SIZE))

# === Основной код ===
cap = cv2.VideoCapture(0)

for label in CLASSES:
    os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)
    count = len(os.listdir(os.path.join(DATA_PATH, label)))
    while True:
        input(f"\nГотов записать {SEQUENCE_LENGTH} кадров для слова '{label}' (ENTER — начать)")
        frames = []
        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue
            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                mouth_img = extract_mouth(image, results.multi_face_landmarks[0])
                frames.append(mouth_img)
                cv2.putText(image, f"Собрано: {len(frames)}/{SEQUENCE_LENGTH}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Сбор данных", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Сохранение последовательности
        save_path = os.path.join(DATA_PATH, label, str(count))
        os.makedirs(save_path, exist_ok=True)
        for i, img in enumerate(frames):
            cv2.imwrite(os.path.join(save_path, f"{i}.jpg"), img)
        print(f"Слово '{label}' — сохранено как последовательность №{count}")
        count += 1

        if input("Добавить ещё одну последовательность? (y/N): ").strip().lower() != 'y':
            break

cap.release()
cv2.destroyAllWindows()
