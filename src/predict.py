
import cv2
import mediapipe as mp
import joblib

from collections import deque, Counter
from preprocess import normalize_landmarks

# --------------------------------
# LOAD MODEL
# --------------------------------

model = joblib.load("models/sign_language_model_webcam_v2.pkl")

print("Model loaded successfully!")

# Prediction history
prediction_history = deque(maxlen=7)

# Sentence storage
sentence = ""

# --------------------------------
# MEDIAPIPE SETUP
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

print("\nSentence formation started!")
print("ENTER = Add predicted letter")
print("SPACE = Add space")
print("BACKSPACE = Delete last character")
print("C = Clear sentence")
print("Q = Quit")

# --------------------------------
# REAL-TIME RECOGNITION
# --------------------------------

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = (
                cv2.getTickCount() * 1000
                // cv2.getTickFrequency()
            )

            result = landmarker.detect_for_video(
                mp_image, int(timestamp)
            )

            stable_prediction = "Waiting..."

            # --------------------------------
            # HAND DETECTION
            # --------------------------------

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                height, width, _ = frame.shape

                for point in hand:

                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

                features = normalize_landmarks(hand)

                if len(features) == 63:

                    prediction = model.predict([features])[0]

                    prediction_history.append(prediction)

                    counts = Counter(prediction_history)

                    letter, count = counts.most_common(1)[0]

                    if len(prediction_history) == 7 and count >= 5:
                        stable_prediction = letter

            else:
                prediction_history.clear()
                stable_prediction = "No hand detected"

            # --------------------------------
            # DISPLAY PREDICTION
            # --------------------------------

            cv2.putText(
                frame,
                f"Prediction: {stable_prediction}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # --------------------------------
            # DISPLAY SENTENCE
            # --------------------------------

            display_sentence = sentence[-35:]

            cv2.rectangle(
                frame,
                (10, 100),
                (frame.shape[1] - 10, 165),
                (40, 40, 40),
                -1
            )

            cv2.putText(
                frame,
                "Sentence:",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                display_sentence,
                (20, 153),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Instructions
            cv2.putText(
                frame,
                "ENTER:Add | SPACE:Space | BACKSPACE:Delete",
                (20, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "C:Clear | Q:Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow("ASL Sentence Formation", frame)

            # --------------------------------
            # KEYBOARD CONTROLS
            # --------------------------------

            key = cv2.waitKey(10) & 0xFF

            # Add predicted letter
            if key in (13, 10):

                if stable_prediction in "ABCDEFGHIKLMNOPQRSTUVWXY" and len(stable_prediction) == 1:
                    sentence += stable_prediction
                    print("Sentence:", sentence)

            # Add space
            elif key == 32:
                sentence += " "

            # Delete last character
            elif key in (8, 127):
                sentence = sentence[:-1]

            # Clear sentence
            elif key in (ord("c"), ord("C")):
                sentence = ""

            # Quit
            elif key in (ord("q"), ord("Q")):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\nFinal sentence:", sentence)
print("Recognition stopped.")
