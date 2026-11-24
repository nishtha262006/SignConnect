
import os
import cv2
import mediapipe as mp

DATASET_DIR = "dataset/asl_alphabet_train"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

with HandLandmarker.create_from_options(options) as landmarker:

    for label in ["A", "B", "C"]:

        folder = os.path.join(DATASET_DIR, label)

        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:10]

        detected = 0

        print(f"\nTesting alphabet {label}")

        for filename in images:

            image = cv2.imread(os.path.join(folder, filename))

            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                detected += 1

        print("Images tested:", len(images))
        print("Hands detected:", detected)
        print("Hands not detected:", len(images) - detected)

print("\nDetection test completed.")
