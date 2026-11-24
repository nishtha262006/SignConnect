
import numpy as np


def extract_landmarks(hand_landmarks):
    """
    Extract X, Y, Z coordinates from 21 hand landmarks.
    Returns 63 numerical features.
    """

    features = []

    for landmark in hand_landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features


def normalize_landmarks(hand_landmarks):
    """
    Normalize all landmarks relative to wrist (landmark 0).
    """

    wrist = hand_landmarks[0]

    normalized_features = []

    for landmark in hand_landmarks:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        normalized_features.extend([x, y, z])

    return normalized_features


def display_features(features):

    print("\nTotal features:", len(features))
    print("\nNormalized landmark coordinates:")

    for i in range(21):
        x = features[i * 3]
        y = features[i * 3 + 1]
        z = features[i * 3 + 2]

        print(
            f"Landmark {i}: "
            f"X={x:.4f}, Y={y:.4f}, Z={z:.4f}"
        )


if __name__ == "__main__":
    print("Preprocessing module loaded successfully!")
    print("Functions available:")
    print("1. extract_landmarks()")
    print("2. normalize_landmarks()")
    print("3. display_features()")
