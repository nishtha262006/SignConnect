
import os
import csv
import cv2
import mediapipe as mp

from preprocess import normalize_landmarks

# -----------------------------------------
# PROJECT PATHS
# -----------------------------------------

DATASET_DIR = os.path.join("dataset", "asl_alphabet_train")
OUTPUT_FILE = os.path.join("dataset", "landmarks.csv")
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# Process only static ASL letters.
# J and Z are excluded because they are dynamic signs.
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# First test: maximum 100 images per alphabet
MAX_IMAGES_PER_CLASS = 500

# 63 landmark features + 1 label
COLUMNS = [f"feature_{i}" for i in range(63)] + ["label"]

# -----------------------------------------
# MEDIAPIPE SETUP
# -----------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

# -----------------------------------------
# IMAGE EXTRACTION
# -----------------------------------------

def get_image_files(folder):
    extensions = (".jpg", ".jpeg", ".png", ".bmp")

    return sorted([
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(extensions)
    ])


def main():

    if not os.path.exists(DATASET_DIR):
        print("Dataset folder not found:", DATASET_DIR)
        return

    if not os.path.exists(MODEL_PATH):
        print("MediaPipe model not found:", MODEL_PATH)
        return

    total_saved = 0
    total_failed = 0

    print("\nStarting automatic landmark extraction...")
    print("Classes:", len(LABELS))
    print("Maximum images per class:", MAX_IMAGES_PER_CLASS)

    # Open CSV in write mode to create a fresh dataset
    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.writer(file)
        writer.writerow(COLUMNS)

        with HandLandmarker.create_from_options(options) as landmarker:

            for label in LABELS:

                folder = os.path.join(DATASET_DIR, label)

                if not os.path.exists(folder):
                    print(f"\nFolder missing: {label}")
                    continue

                image_files = get_image_files(folder)

                if not image_files:
                    print(f"\nNo images found for {label}")
                    continue

                # Select evenly spaced images from the folder
                step = max(
                    1,
                    len(image_files) // MAX_IMAGES_PER_CLASS
                )

                selected_files = image_files[
                    ::step
                ][:MAX_IMAGES_PER_CLASS]

                class_saved = 0
                class_failed = 0

                print(f"\nProcessing alphabet: {label}")

                for index, image_path in enumerate(
                    selected_files, start=1
                ):

                    image = cv2.imread(image_path)

                    if image is None:
                        class_failed += 1
                        continue

                    # Convert BGR image to RGB
                    rgb_image = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_image
                    )

                    # Detect hand landmarks
                    result = landmarker.detect(mp_image)

                    if not result.hand_landmarks:
                        class_failed += 1
                        continue

                    hand = result.hand_landmarks[0]

                    # Normalize 21 landmarks relative to wrist
                    features = normalize_landmarks(hand)

                    if len(features) != 63:
                        class_failed += 1
                        continue

                    # Save features and alphabet label
                    writer.writerow(features + [label])

                    class_saved += 1
                    total_saved += 1

                    if index % 20 == 0:
                        print(
                            f"Checked {index}/"
                            f"{len(selected_files)} images"
                        )

                total_failed += class_failed
                file.flush()

                print(
                    f"{label}: {class_saved} saved, "
                    f"{class_failed} not detected"
                )

    # -----------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------

    print("\n--------------------------------")
    print("LANDMARK EXTRACTION COMPLETED")
    print("--------------------------------")

    print("Total successful samples:", total_saved)
    print("Total failed images:", total_failed)
    print("Output file:", OUTPUT_FILE)

    if total_saved == 0:
        print("\nNo landmarks were extracted.")
        print("We need to inspect the images and detection settings.")
    else:
        print("\nYour landmark dataset has been generated!")


if __name__ == "__main__":
    main()
