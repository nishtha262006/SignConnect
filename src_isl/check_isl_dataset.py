
import cv2
import mediapipe as mp
import os
import random

# Dataset location
DATASET_DIR = "dataset_isl/realsign/Training"

# Select sample letters
LETTERS = ["A", "B", "C", "M", "N", "Z"]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:

    for letter in LETTERS:

        folder = os.path.join(DATASET_DIR, letter)
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not images:
            print(f"{letter}: No images found")
            continue

        sample = random.choice(images)
        image_path = os.path.join(folder, sample)

        image = cv2.imread(image_path)

        if image is None:
            print(f"{letter}: Image could not be loaded")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        hand_count = len(result.hand_landmarks)

        print(f"{letter}: {hand_count} hand(s) detected")

        for hand in result.hand_landmarks:
            for point in hand:
                x = int(point.x * image.shape[1])
                y = int(point.y * image.shape[0])

                cv2.circle(
                    image, (x, y), 4,
                    (0, 255, 0), -1
                )

        cv2.imshow(f"ISL Letter {letter}", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

print("\nISL MediaPipe testing completed!")
