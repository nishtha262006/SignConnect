
import cv2
import mediapipe as mp
import os
import csv
import time

from preprocess import normalize_landmarks

# ---------------------------------
# PROJECT PATHS
# ---------------------------------

MODEL_PATH = os.path.join("models", "hand_landmarker.task")
DATASET_PATH = os.path.join("dataset", "landmarks.csv")

os.makedirs("dataset", exist_ok=True)

# Static ASL alphabets (J and Z excluded)
VALID_LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# CSV columns: 63 features + label
columns = [f"feature_{i}" for i in range(63)] + ["label"]

# ---------------------------------
# MEDIAPIPE SETUP
# ---------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# ---------------------------------
# CREATE / OPEN DATASET
# ---------------------------------

file_exists = (
    os.path.exists(DATASET_PATH)
    and os.path.getsize(DATASET_PATH) > 0
)

# ---------------------------------
# DATA COLLECTION
# ---------------------------------

cap = None

try:
    with open(DATASET_PATH, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(columns)
            file.flush()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam!")

        print("\nWebcam opened successfully!")

        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

                label = input(
                    "\nEnter alphabet (A-Y, excluding J), or EXIT: "
                ).strip().upper()

                if label == "EXIT":
                    break

                if label not in VALID_LABELS:
                    print("Invalid alphabet! Try again.")
                    continue

                print(f"\nCollecting samples for {label}")
                print("Click inside the webcam window.")
                print("Press S to save a sample.")
                print("Press Q to finish this alphabet.")

                sample_count = 0
                last_result = None

                while True:

                    success, frame = cap.read()

                    if not success:
                        print("Unable to capture frame.")
                        break

                    # Mirror webcam image
                    frame = cv2.flip(frame, 1)

                    # Convert BGR to RGB
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

                    last_result = result

                    # ---------------------------------
                    # HAND DETECTION AND DISPLAY
                    # ---------------------------------

                    if result.hand_landmarks:

                        hand = result.hand_landmarks[0]

                        height, width, _ = frame.shape

                        points = [
                            (int(p.x * width), int(p.y * height))
                            for p in hand
                        ]

                        # Draw landmarks
                        for x, y in points:
                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 0, 255),
                                -1
                            )

                        # Draw label
                        cv2.putText(
                            frame,
                            f"Label: {label}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        status = "Hand Detected"

                    else:

                        status = "No Hand Detected"

                        cv2.putText(
                            frame,
                            status,
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                    # Display sample count
                    cv2.putText(
                        frame,
                        f"Samples: {sample_count}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "S = Save | Q = Next Alphabet",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.imshow("ASL Dataset Collection", frame)

                    # ---------------------------------
                    # KEYBOARD INPUT
                    # ---------------------------------

                    key = cv2.waitKey(10) & 0xFF

                    # Debug: display detected key
                    if key != 255:
                        print("Key pressed:", key)

                    # SAVE SAMPLE
                    if key in (ord("s"), ord("S")):

                        if (
                            last_result is not None
                            and last_result.hand_landmarks
                        ):

                            hand = last_result.hand_landmarks[0]

                            features = normalize_landmarks(hand)

                            if len(features) == 63:

                                writer.writerow(features + [label])
                                file.flush()

                                sample_count += 1

                                print(
                                    f"SUCCESS! {label}: "
                                    f"{sample_count} samples saved"
                                )

                        else:
                            print("No hand detected. Try again.")

                    # FINISH CURRENT ALPHABET
                    elif key in (ord("q"), ord("Q")):

                        print(f"Finished collecting {label}.")
                        break

                print(
                    f"\nCompleted {label}: "
                    f"{sample_count} samples collected."
                )

finally:

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()

print("\nDataset collection session finished.")
print("Dataset saved at:", DATASET_PATH)
