
def normalize_hand(hand_landmarks):
    """
    Convert 21 hand landmarks into 63 wrist-relative features.
    """

    if not hand_landmarks:
        return [0.0] * 63

    wrist = hand_landmarks[0]

    features = []

    for landmark in hand_landmarks:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        features.extend([x, y, z])

    return features


def extract_both_hands(hand_landmarks, handedness):
    """
    Extract features for one or two hands.

    Single hand:
        Always uses the first 63-feature slot.

    Two hands:
        Left hand  = first 63 features.
        Right hand = next 63 features.

    Total = 126 features.
    """

    left_features = [0.0] * 63
    right_features = [0.0] * 63

    # CASE 1: Only one hand detected
    if len(hand_landmarks) == 1:

        features = normalize_hand(hand_landmarks[0])

        # Always place a single hand in the first slot
        left_features = features

    # CASE 2: Two hands detected
    elif len(hand_landmarks) >= 2:

        for hand, classification in zip(
            hand_landmarks, handedness
        ):

            if not classification:
                continue

            hand_label = classification[0].category_name

            features = normalize_hand(hand)

            if hand_label == "Left":
                left_features = features

            elif hand_label == "Right":
                right_features = features

    return left_features + right_features


def display_features(features):

    print("\nTotal features:", len(features))

    print("\nFirst hand features: 63")
    print("Second hand features: 63")

    print("\nFirst 10 features:")
    print(features[:10])


if __name__ == "__main__":

    print("ISL preprocessing module loaded successfully!")

    print("\nAvailable functions:")
    print("1. normalize_hand()")
    print("2. extract_both_hands()")
    print("3. display_features()")

    print("\nExpected feature count: 126")
