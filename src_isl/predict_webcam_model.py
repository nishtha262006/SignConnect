
import cv2
import mediapipe as mp
import joblib
import sys
import os

from collections import deque, Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_isl import extract_both_hands

MODEL_PATH = "models_isl/isl_webcam_model.pkl"

model = joblib.load(MODEL_PATH)

prediction_history = deque(maxlen=7)

print("Webcam-specific model loaded successfully!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    exit()

print("Press Q to quit.")

frame_number = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_number += 1
        timestamp = frame_number * 33

        result = landmarker.detect_for_video(
            mp_image, timestamp
        )

        prediction_text = "No hand detected"

        if result.hand_landmarks:

            features = extract_both_hands(
                result.hand_landmarks,
                result.handedness
            )

            prediction = model.predict([features])[0]

            prediction_history.append(prediction)

            stable_prediction = Counter(
                prediction_history
            ).most_common(1)[0][0]

            prediction_text = "ISL: " + stable_prediction

            for hand in result.hand_landmarks:
                for landmark in hand:

                    height, width, _ = frame.shape

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

        else:
            prediction_history.clear()

        cv2.putText(
            frame,
            prediction_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SignConnect - Webcam Model",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("Webcam test stopped.")
