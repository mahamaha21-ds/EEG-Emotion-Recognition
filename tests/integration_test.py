import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

print("Running Integration Test...")

try:
    # Load data
    df = pd.read_csv("data/emotions.csv")
    X = df.iloc[:, :-1] if 'label' in df.columns else df

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model
    model = tf.keras.models.load_model("eeg_emotion_model.keras")

    # Reshape for CNN
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Predict
    preds = model.predict(X_scaled)

    print("Integration Test: PASSED")

except Exception as e:
    print("Integration Test: FAILED")
    print("Error:", e)
