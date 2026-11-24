
import cv2
import mediapipe as mp
import os
import time

# Model path
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# Check model file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Hand landmark model not found!")

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

# Hand connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = int(time.perf_counter() * 1000)

            result = landmarker.detect_for_video(
                mp_image, timestamp
            )

            if result.hand_landmarks:

                for hand in result.hand_landmarks:

                    points = []

                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    # Draw connections
                    for start, end in connections:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0), 2
                        )

                    # Draw landmarks
                    for x, y in points:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255), -1
                        )

                    # Bounding box
                    x_min = max(0, min(p[0] for p in points) - 20)
                    y_min = max(0, min(p[1] for p in points) - 20)
                    x_max = min(width - 1, max(p[0] for p in points) + 20)
                    y_max = min(height - 1, max(p[1] for p in points) + 20)

                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (255, 0, 0), 2
                    )

                    cv2.putText(
                        frame,
                        "Hand Detected - 21 Landmarks",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0), 2
                    )

            else:
                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255), 2
                )

            cv2.imshow("Sign Language Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
