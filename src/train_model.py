
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
DATASET_PATH = "dataset/landmarks.csv"

data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total samples:", len(data))

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

print("Number of features:", X.shape[1])
print("Number of classes:", y.nunique())

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Linear SVC model
model = LinearSVC(
    C=1.0,
    max_iter=10000,
    random_state=42
)

# Train model
print("\nTraining Linear SVC model...")

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n--------------------------------")
print("MODEL EVALUATION")
print("--------------------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
joblib.dump(model, "models/sign_language_model.pkl")

print("\nModel saved successfully!")
print("Location: models/sign_language_model.pkl")
