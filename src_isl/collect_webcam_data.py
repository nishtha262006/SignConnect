
import cv2
import mediapipe as mp
import csv
import os
import sys

from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_isl import extract_both_hands

# Settings
OUTPUT_DIR = "dataset_isl/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "webcam_landmarks.csv")

LETTERS = ["A", "B", "C", "P", "M", "O", "Q"]
SAMPLES_PER_LETTER = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create CSV if it does not exist
file_exists = os.path.exists(OUTPUT_FILE)

feature_names = [
    f"feature_{i}" for i in range(126)
]

with open(OUTPUT_FILE, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
        writer.writerow(feature_names + ["label"])

print("SignConnect Webcam Data Collection")
print("Letters:", LETTERS)
print("Samples per letter:", SAMPLES_PER_LETTER)

# Initialize MediaPipe
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
    print("Error: Could not open webcam.")
    exit()

selected_letter = None
collecting = False
sample_count = 0
frame_number = 0

# Collect approximately 5 samples per second
frame_interval = 6

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to capture webcam frame.")
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

            # Draw detected hand landmarks
            if result.hand_landmarks:
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

            # Collect samples
            if (
                collecting
                and result.hand_landmarks
                and frame_number % frame_interval == 0
            ):

                features = extract_both_hands(
                    result.hand_landmarks,
                    result.handedness
                )

                if len(features) == 126:

                    with open(
                        OUTPUT_FILE, "a", newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow(
                            features + [selected_letter]
                        )

                    sample_count += 1

                    print(
                        f"{selected_letter}: "
                        f"{sample_count}/{SAMPLES_PER_LETTER}"
                    )

                if sample_count >= SAMPLES_PER_LETTER:
                    collecting = False
                    print(
                        f"\n{selected_letter} collection completed!"
                    )

            # Display instructions
            cv2.putText(
                frame,
                "Select letter: A B C M O P Q",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            status = (
                f"Collecting {selected_letter}: "
                f"{sample_count}/{SAMPLES_PER_LETTER}"
                if collecting
                else f"Selected: {selected_letter}"
                if selected_letter
                else "Select a letter"
            )

            cv2.putText(
                frame,
                status,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "SPACE: Start | ESC: Exit",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "SignConnect - Webcam Data Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if not collecting:

                if key in [ord(letter.lower()) for letter in LETTERS]:
                    selected_letter = chr(key).upper()
                    sample_count = 0
                    print("\nSelected letter:", selected_letter)

                elif key == ord(" "):

                    if selected_letter is None:
                        print("Select a letter first!")
                    else:
                        sample_count = 0
                        collecting = True
                        print(
                            f"\nStarting collection for "
                            f"{selected_letter}..."
                        )

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nData collection stopped.")
print("Saved data:", OUTPUT_FILE)
