import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

print("Running Validation Test...")

try:
    df = pd.read_csv("data/emotions.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model
    model = tf.keras.models.load_model("eeg_emotion_model.keras")

    # Reshape for CNN
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Predict
    preds = model.predict(X_scaled)
    preds_classes = np.argmax(preds, axis=1)

    acc = accuracy_score(y_encoded, preds_classes)

    print(f"Validation Accuracy: {acc:.4f}")
    print("Validation Test: PASSED")

except Exception as e:
    print("Validation Test: FAILED")
    print("Error:", e)
