
import os
import csv
import cv2
import mediapipe as mp

from preprocess_isl import extract_both_hands

# -----------------------------------
# PROJECT SETTINGS
# -----------------------------------

DATASET_DIR = "dataset_isl/realsign"

OUTPUT_DIR = "dataset_isl/processed"

MODEL_PATH = "models/hand_landmarker.task"

SPLITS = ["Training", "Testing", "Validation"]

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_IMAGES_PER_CLASS = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# MEDIAPIPE SETUP
# -----------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

# -----------------------------------
# DATASET EXTRACTION
# -----------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    for split in SPLITS:

        input_dir = os.path.join(DATASET_DIR, split)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{split.lower()}_landmarks.csv"
        )

        print("\n================================")
        print("Processing:", split)
        print("================================")

        total_images = 0
        successful_images = 0

        with open(output_file, "w", newline="") as file:

            writer = csv.writer(file)

            columns = [
                f"feature_{i}" for i in range(126)
            ] + ["label"]

            writer.writerow(columns)

            for letter in LETTERS:

                letter_dir = os.path.join(
                    input_dir, letter
                )

                if not os.path.isdir(letter_dir):
                    print("Missing folder:", letter)
                    continue
                images = [
                    f for f in os.listdir(letter_dir)
                    if f.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    )
                ]

                if MAX_IMAGES_PER_CLASS is not None:
                    images = images[:MAX_IMAGES_PER_CLASS]

                letter_success = 0

                for image_name in images:

                    image_path = os.path.join(
                        letter_dir, image_name
                    )

                    image = cv2.imread(image_path)

                    if image is None:
                        continue
                    # Mirror the dataset image to match webcam input
                    image = cv2.flip(image, 1)

                    rgb = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    result = landmarker.detect(mp_image)

                    total_images += 1

                    if not result.hand_landmarks:
                        continue

                    features = extract_both_hands(
                        result.hand_landmarks,
                        result.handedness
                    )

                    if len(features) != 126:
                        continue

                    writer.writerow(features + [letter])

                    successful_images += 1
                    letter_success += 1

                print(
                    f"{letter}: "
                    f"{letter_success}/{len(images)} images processed"
                )

        print("\nTotal images:", total_images)
        print("Successful extractions:", successful_images)
        print("Saved to:", output_file)

print("\nISL dataset extraction completed!")
