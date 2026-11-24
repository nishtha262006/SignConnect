
import cv2
import mediapipe as mp
import joblib
import sys
import os
from collections import deque, Counter

# Import ISL preprocessing function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_isl import extract_both_hands

# Load trained ISL model
MODEL_PATH = "models_isl/isl_sign_language_model.pkl"
model = joblib.load(MODEL_PATH)

# Store the last 10 predictions
prediction_history = deque(maxlen=10)

print("ISL model loaded successfully!")

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

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nISL Webcam Recognition Started!")
print("Press Q to quit.")

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to capture webcam frame.")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image for MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        timestamp = int(
            cv2.getTickCount() * 1000 / cv2.getTickFrequency()
        )

        result = landmarker.detect_for_video(mp_image, timestamp)

        prediction_text = "No hand detected"

        if result.hand_landmarks:

            # Extract 126 features
            features = extract_both_hands(
                result.hand_landmarks,
                result.handedness
            )
            print("Feature count:", len(features))
            print("First 10 features:", features[:10])
            print("Second hand slot:", features[63:73])
            print("Hand labels:", [
                h[0].category_name for h in result.handedness
            ])

            print("First 10 features:", features[:10])
            print("Second 10 features:", features[63:73])

            # Predict ISL alphabet
            prediction = model.predict([features])[0]

            hand_labels = [
                h[0].category_name for h in result.handedness
            ]

            print("Detected hand:", hand_labels)
            print("Raw prediction:", prediction)

            # Add prediction to history
            prediction_history.append(prediction)

            # Find the most frequent prediction
            stable_prediction = Counter(
                prediction_history
            ).most_common(1)[0][0]

            prediction_text = "ISL: " + stable_prediction

            # Draw detected landmarks
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
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

        else:
            # Clear history when no hand is detected
            prediction_history.clear()

        # Display prediction
        cv2.putText(
            frame,
            prediction_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("SignConnect - ISL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("Webcam recognition stopped.")
