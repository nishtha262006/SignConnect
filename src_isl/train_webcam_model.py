
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "dataset_isl/processed/webcam_landmarks.csv"
MODEL_PATH = "models_isl/isl_webcam_model.pkl"

print("Loading webcam dataset...")

data = pd.read_csv(DATA_PATH)

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining webcam SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nWEBCAM MODEL RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

os.makedirs("models_isl", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nWebcam model saved successfully!")
print("Model location:", MODEL_PATH)
