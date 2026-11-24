
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# LOAD DATASETS
# --------------------------------

original = pd.read_csv("dataset/landmarks.csv")
webcam = pd.read_csv("dataset/webcam_landmarks.csv")

print("Original samples:", len(original))
print("Webcam samples:", len(webcam))

# --------------------------------
# SPLIT WEBCAM DATA
# --------------------------------

# Keep 20% of webcam samples separate for testing.

webcam_train, webcam_test = train_test_split(
    webcam,
    test_size=0.20,
    random_state=42,
    stratify=webcam["label"]
)

print("\nWebcam training samples:", len(webcam_train))
print("Webcam testing samples:", len(webcam_test))

# --------------------------------
# PREPARE TRAINING DATA
# --------------------------------

# Combine original dataset with webcam training samples.

combined_train = pd.concat(
    [original, webcam_train],
    ignore_index=True
)

X_train = combined_train.drop("label", axis=1)
y_train = combined_train["label"]

X_test = webcam_test.drop("label", axis=1)
y_test = webcam_test["label"]

print("\nTotal training samples:", len(X_train))
print("Webcam testing samples:", len(X_test))

# --------------------------------
# TRAIN NEW MODEL
# --------------------------------

model = LinearSVC(
    C=1.0,
    max_iter=10000,
    random_state=42
)

print("\nTraining improved webcam Linear SVC...")

model.fit(X_train, y_train)

# --------------------------------
# EVALUATE MODEL
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n--------------------------------")
print("WEBCAM MODEL EVALUATION")
print("--------------------------------")

print(f"\nWebcam accuracy: {accuracy * 100:.2f}%")

webcam_labels = sorted(webcam["label"].unique())

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=webcam_labels,
        zero_division=0
    )
)

# --------------------------------
# CONFUSION MATRIX
# --------------------------------

print("\nConfusion Matrix:")
print("Labels:", webcam_labels)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=webcam_labels
)

print(matrix)

# --------------------------------
# SAVE NEW MODEL
# --------------------------------

joblib.dump(
    model,
    "models/sign_language_model_webcam_v2.pkl"
)

print("\nNew model saved successfully!")
print("Location: models/sign_language_model_webcam_v2.pkl")

print("\nPrevious models remain unchanged.")
