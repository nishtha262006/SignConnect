
import cv2
import mediapipe as mp
import time
import os

from preprocess import normalize_landmarks, display_features

# Model path
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

print("Webcam started!")
print("Show your hand to the camera.")
print("Press S to extract normalized coordinates.")
print("Press Q to exit.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

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

            timestamp = int(time.perf_counter() * 1000)

            result = landmarker.detect_for_video(
                mp_image, timestamp
            )

            if result.hand_landmarks:
                cv2.putText(
                    frame,
                    "Hand Detected - Press S to Extract",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Landmark Extraction Test", frame)

            key = cv2.waitKey(1) & 0xFF

            # Extract coordinates when S is pressed
            if key == ord('s'):

                if result.hand_landmarks:

                    hand = result.hand_landmarks[0]

                    features = normalize_landmarks(hand)

                    print("\n--- NORMALIZED LANDMARKS ---")

                    display_features(features)

                    print("\nTotal feature count:", len(features))

                else:
                    print("No hand detected! Try again.")

            if key == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
