# SignConnect
### AI-Based Sign Language Recognition System

SignConnect is an AI-powered sign language recognition application designed to bridge communication gaps between sign language users and non-signers. It uses computer vision and machine learning to recognise hand gestures and convert them into English alphabets.

The application supports two sign languages:
- **ASL:** American Sign Language
- **ISL:** Indian Sign Language

Users can select their preferred sign language and perform real-time recognition through a webcam.

## Features

- Real-time hand detection using MediaPipe.
- ASL alphabet recognition.
- ISL alphabet recognition.
- Separate trained machine learning models for ASL and ISL.
- Hand landmark extraction and preprocessing.
- Webcam-based sign recognition.
- ASL sentence formation using recognised alphabets.
- Streamlit-based user interface.
- Start and stop recognition controls.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand detection and landmark extraction |
| Scikit-learn | Machine learning classification |
| SVM / Linear SVC | Sign classification |
| NumPy | Numerical operations |
| Pandas | Dataset processing |
| Joblib | Model saving and loading |
| Streamlit | Application interface |

## Project Structure

```text
Sign-Language-Recognition/
│
├── dataset/
│   ├── landmarks.csv
│   └── webcam_landmarks.csv
│
├── dataset_isl/
│   └── processed/
│       ├── training_landmarks.csv
│       ├── testing_landmarks.csv
│       ├── validation_landmarks.csv
│       └── webcam_landmarks.csv
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── hand_landmarker.task
│   ├── sign_language_model.pkl
│   ├── sign_language_model_webcam.pkl
│   └── sign_language_model_webcam_v2.pkl
│
├── models_isl/
│   ├── isl_sign_language_model.pkl
│   └── isl_webcam_model.pkl
│
├── src/
│   ├── collect_data.py
│   ├── extract_dataset.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── train_webcam_model.py
│   └── predict.py
│
├── src_isl/
│   ├── extract_isl_dataset.py
│   ├── preprocess_isl.py
│   ├── train_isl_model.py
│   ├── train_webcam_model.py
│   ├── predict_isl_webcam.py
│   └── predict_combined_isl.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

1. The webcam captures the user's hand gestures.
2. MediaPipe detects the hand and extracts 21 hand landmarks.
3. The landmark coordinates are preprocessed and normalised.
4. The extracted features are passed to the selected machine learning model.
5. The model predicts the corresponding sign language alphabet.
6. The recognised alphabet is displayed in real time.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Sign-Language-Recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run frontend/app.py
```

Open the local URL displayed in the terminal to access SignConnect.

## Model Performance

The following are the recorded evaluation results from the current development version:

| Model | Evaluation Accuracy |
|---|---:|
| ASL webcam model | 81.60% |
| ISL 26-letter model | 92.27% |
| ISL webcam-specific model | 98.57% |

The webcam-specific ISL result is based on a random train-test split of collected webcam samples. Actual performance can vary with hand position, lighting and recording conditions.

## Current Scope and Limitations

- The current system focuses on alphabet-level recognition.
- ASL sentence formation is implemented using recognised alphabets.
- Full continuous sign language translation is not yet implemented.
- Dynamic signs and complete sentence-level interpretation require further development.
- Recognition performance may vary depending on lighting, camera quality and hand positioning.

## Future Enhancements

- Improve recognition accuracy across all alphabets.
- Develop continuous sign recognition.
- Implement sentence-level translation.
- Expand the supported sign language vocabulary.
- Improve the user interface and accessibility.
- Explore speech output for recognised sentences.

## Project Objective

SignConnect aims to make communication more accessible through artificial intelligence, computer vision and sign language recognition.

**Technology for Inclusive Communication.**
