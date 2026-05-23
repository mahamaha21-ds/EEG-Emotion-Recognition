import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

print("Running System Test...")

try:
    # Simulate user upload
    df = pd.read_csv("data/emotions.csv")

    X = df.iloc[:, :-1] if 'label' in df.columns else df

    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Load model
    model = tf.keras.models.load_model("eeg_emotion_model.keras")

    # Predict
    preds = model.predict(X_scaled)
    pred_class = np.argmax(preds, axis=1)

    print("Predicted classes:", pred_class[:5])
    print("System Test: PASSED")

except Exception as e:
    print("System Test: FAILED")
    print("Error:", e)
