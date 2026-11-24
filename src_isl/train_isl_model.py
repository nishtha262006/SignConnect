
import os
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset paths
TRAIN_PATH = "dataset_isl/processed/training_landmarks.csv"
TEST_PATH = "dataset_isl/processed/testing_landmarks.csv"
VALIDATION_PATH = "dataset_isl/processed/validation_landmarks.csv"

# Model output path
MODEL_PATH = "models_isl/isl_sign_language_model.pkl"

# Load datasets
print("Loading ISL datasets...")

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
validation_data = pd.read_csv(VALIDATION_PATH)

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))
print("Validation samples:", len(validation_data))

# Separate features and labels
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]

X_validation = validation_data.drop("label", axis=1)
y_validation = validation_data["label"]

# Train model
print("\nTraining ISL SVM model...")

model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

# Evaluate on validation data
print("\nVALIDATION RESULTS")

validation_predictions = model.predict(X_validation)

validation_accuracy = accuracy_score(
    y_validation, validation_predictions
)

print("Validation Accuracy:", round(validation_accuracy * 100, 2), "%")

# Evaluate on test data
print("\nTEST RESULTS")

test_predictions = model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)

print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# Save model
os.makedirs("models_isl", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nISL model saved successfully!")
print("Model location:", MODEL_PATH)
