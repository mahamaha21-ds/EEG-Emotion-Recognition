# ==========================================
# MSc Data Science Project
# EEG Emotion Classification
# train_model.py
# ==========================================

# ---------- STEP 0: IMPORT LIBRARIES ----------
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks



# --------------------------------------------
# STEP 4: LOAD EEG DATASET
# --------------------------------------------

DATA_PATH = "data"

all_data = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".csv"):
        file_path = os.path.join(DATA_PATH, file)
        df = pd.read_csv(file_path)
        all_data.append(df)

# Combine all CSV files
data = pd.concat(all_data, ignore_index=True)

print("Dataset loaded successfully\n")

print("First 5 rows:")
print(data.head(), "\n")

print("Dataset shape (rows, columns):")
print(data.shape, "\n")

print("Column names:")
print(data.columns, "\n")

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)

print("\nUnique emotion labels:")
print(y.unique())

# --------------------------------------------
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------

# 5.1 Emotion class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Emotion Class Distribution")
plt.xlabel("Emotion Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5.2 Basic statistics
print("\nBasic statistics of EEG features:")
print(X.describe())

# 5.3 Feature variance analysis
feature_variance = X.var()

plt.figure(figsize=(8,4))
feature_variance.plot(kind="line")
plt.title("Variance of EEG Features")
plt.xlabel("Feature Index")
plt.ylabel("Variance")
plt.tight_layout()
plt.show()

# --------------------------------------------
# STEP 6: DATA PREPROCESSING (SCALING)
# --------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nSTEP 6 OUTPUT")
print("Feature scaling completed")
print("Scaled feature shape:", X_scaled.shape)

print("\nMean of first 5 scaled features (approx 0):")
print(np.mean(X_scaled, axis=0)[:5])

print("\nStd of first 5 scaled features (approx 1):")
print(np.std(X_scaled, axis=0)[:5])

# --------------------------------------------
# STEP 7: LABEL ENCODING + TRAIN-TEST SPLIT
# --------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded labels:", np.unique(y_encoded))
print("Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(i, "->", label)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nSTEP 7 COMPLETED SUCCESSFULLY")

print("\nSTEP 8: BASELINE MACHINE LEARNING MODELS")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr,
                            target_names=label_encoder.classes_))


# Support Vector Machine
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("\nSVM Accuracy:",
      accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm,
                            target_names=label_encoder.classes_))

print("\nSTEP 9: CNN MODEL")

# Reshape data for CNN (samples, features, 1)
X_train_cnn = X_train[:, :, np.newaxis]
X_test_cnn = X_test[:, :, np.newaxis]

cnn_model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu',
                  input_shape=(X_train_cnn.shape[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = cnn_model.fit(
    X_train_cnn, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

print("\nCNN Accuracy:",
      accuracy_score(y_test, y_pred_cnn))
print(classification_report(y_test, y_pred_cnn,
                            target_names=label_encoder.classes_))

# --------------------------------------------
# STEP 11: MODEL COMPARISON
# --------------------------------------------

models_list = ["Logistic Regression", "SVM", "CNN"]
accuracy_list = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_svm),
    accuracy_score(y_test, y_pred_cnn)
]

plt.figure(figsize=(6,4))
sns.barplot(x=models_list, y=accuracy_list)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()
# --------------------------------------------
# STEP 12: SAVE FINAL MODEL
# --------------------------------------------

cnn_model.save("eeg_emotion_model.keras")
print("\nFinal CNN model saved as eeg_emotion_model.keras")
