# EEG Emotion Recognition System using Machine Learning and Deep Learning

## Project Overview

This project presents an EEG-based Emotion Recognition System developed using Machine Learning and Deep Learning techniques. The system analyzes EEG-derived feature data and classifies human emotions into three categories:

- NEGATIVE
- NEUTRAL
- POSITIVE

The project follows a complete data science workflow including data preprocessing, model training, evaluation, testing, visualization, and deployment using a Streamlit web application.

---

## Objectives

- Analyze EEG feature data for emotion classification.
- Implement Machine Learning and Deep Learning models.
- Compare model performance using evaluation metrics.
- Deploy the trained model as a web-based application.

---

## Models Used

The following models are implemented in this project:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Convolutional Neural Network (CNN)

---

## Project Structure

```bash
EEG_Emotion_Project/
│
├── data/
│   ├── eeg_emotion.csv
├── tests/
│   ├── unit_test.py
│   ├── integration_test.py
│   ├── validation_test.py
│   └── system_test.py
│
├── train_model.py
├── app.py
├── eeg_emotion_model.keras
├── requirements.txt
└── README.md
