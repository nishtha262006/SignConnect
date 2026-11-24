
import cv2
import mediapipe as mp
import os
import csv
import time

from preprocess import normalize_landmarks

# -----------------------------------
# PROJECT SETTINGS
# -----------------------------------

MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_FILE = "dataset/webcam_landmarks.csv"

LABELS = ["E", "O"]

TARGET_SAMPLES = {
    "E": 50,
    "O": 100
}

CAPTURE_INTERVAL = 0.5

os.makedirs("dataset", exist_ok=True)

columns = [f"feature_{i}" for i in range(63)] + ["label"]

# -----------------------------------
# MEDIAPIPE SETUP
# -----------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# -----------------------------------
# DATA COLLECTION
# -----------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

try:
    # Append new samples without deleting existing data
    with open(OUTPUT_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        # Write header only if the file is empty
        file.seek(0, os.SEEK_END)

        if file.tell() == 0:
            writer.writerow(columns)

        with HandLandmarker.create_from_options(options) as landmarker:

            for label in LABELS:

                target = TARGET_SAMPLES[label]
                saved = 0
                last_capture = 0

                print("\n--------------------------------")
                print("Prepare ASL letter:", label)
                print("Press SPACE in the webcam window to start.")
                print("Press Q to quit.")
                print("--------------------------------")

                started = False

                while saved < target:

                    success, frame = cap.read()

                    if not success:
                        print("Camera frame unavailable.")
                        break

                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    timestamp = int(time.perf_counter() * 1000)

                    result = landmarker.detect_for_video(
                        mp_image, timestamp
                    )

                    height, width, _ = frame.shape

                    if result.hand_landmarks:

                        hand = result.hand_landmarks[0]

                        for point in hand:
                            x = int(point.x * width)
                            y = int(point.y * height)

                            cv2.circle(
                                frame, (x, y), 4,
                                (0, 255, 0), -1
                            )

                        cv2.putText(
                            frame,
                            "Hand detected",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2
                        )

                        if started:
                            now = time.perf_counter()

                            if now - last_capture >= CAPTURE_INTERVAL:

                                features = normalize_landmarks(hand)

                                if len(features) == 63:
                                    writer.writerow(features + [label])
                                    file.flush()

                                    saved += 1
                                    last_capture = now

                                    print(
                                        f"{label}: "
                                        f"{saved}/{target} saved"
                                    )

                    else:
                        cv2.putText(
                            frame,
                            "No hand detected",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2
                        )

                    if not started:
                        instruction = f"Show {label} | SPACE to start"
                    else:
                        instruction = (
                            f"Collecting {label}: {saved}/{target}"
                        )

                    cv2.putText(
                        frame,
                        instruction,
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2
                    )

                    cv2.imshow("Webcam Dataset Collector", frame)

                    key = cv2.waitKey(10) & 0xFF

                    if key == 32 and not started:
                        started = True
                        last_capture = 0
                        print("Started collecting", label)

                    elif key in (ord("q"), ord("Q")):
                        raise KeyboardInterrupt

                if saved < target:
                    print("Collection interrupted.")
                    break

                print(f"Completed {label}: {saved} samples.")

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\nWebcam collection session finished.")
print("Dataset saved to:", OUTPUT_FILE)
