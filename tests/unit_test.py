import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("Running Unit Tests...")

# Test 1: Data loading
try:
    df = pd.read_csv("data/emotions.csv")
    print("Data Loading Test: PASSED")
except:
    print("Data Loading Test: FAILED")

# Test 2: Feature scaling
try:
    X = df.iloc[:, :-1] if 'label' in df.columns else df
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature Scaling Test: PASSED")
except:
    print("Feature Scaling Test: FAILED")

# Test 3: Model loading
try:
    model = tf.keras.models.load_model("eeg_emotion_model.keras")
    print("Model Loading Test: PASSED")
except:
    print("Model Loading Test: FAILED")

print("Unit Tests Completed.")
