import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------
model = tf.keras.models.load_model("eeg_emotion_model.keras")

# Emotion labels (same order as LabelEncoder)
emotion_map = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE"
}

# --------------------------------------------
# PAGE SETTINGS
# --------------------------------------------
st.set_page_config(page_title="EEG Emotion Recognition", layout="centered")

st.title("EEG Emotion Recognition System")
st.write(
    "This web application predicts human emotions "
    "using EEG brain signal data and a trained CNN model."
)

# --------------------------------------------
# FILE UPLOAD
# --------------------------------------------
uploaded_file = st.file_uploader(
    "Upload EEG CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded EEG Data Preview")
    st.dataframe(df.head())

    # Drop label column if present
    if "label" in df.columns:
        df = df.drop("label", axis=1)

    # ----------------------------------------
    # PREPROCESSING (same as training)
    # ----------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    X_input = X_scaled[:, :, np.newaxis]

    # ----------------------------------------
    # PREDICTION
    # ----------------------------------------
    predictions = model.predict(X_input)
    final_prediction = np.argmax(np.mean(predictions, axis=0))

    st.success(
        f" Predicted Emotion: **{emotion_map[final_prediction]}**"
    )

st.markdown("---")
st.caption(
    "EEG Emotion Recognition"
)
