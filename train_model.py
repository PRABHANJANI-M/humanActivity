import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ------------------------------
# Load Datasets
# ------------------------------
print("📂 Loading datasets...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Ensure target column is named correctly
TARGET_COL = "Activity"  # change if your label name differs

if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
    raise ValueError(f"Dataset must have a column named '{TARGET_COL}'.")

# ------------------------------
# Split Features & Labels
# ------------------------------
X_train = train_df.drop(TARGET_COL, axis=1).values
y_train = train_df[TARGET_COL].values

X_test = test_df.drop(TARGET_COL, axis=1).values
y_test = test_df[TARGET_COL].values

# ------------------------------
# Scale Features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Encode Labels
# ------------------------------
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# ------------------------------
# Reshape for LSTM
# ------------------------------
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# ------------------------------
# Build LSTM Model
# ------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(1, X_train_scaled.shape[2])),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ------------------------------
# Train Model
# ------------------------------
print("🚀 Training model...")
history = model.fit(
    X_train_scaled, y_train_encoded,
    epochs=25,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_encoded),
    verbose=1
)

# ------------------------------
# Evaluate Model
# ------------------------------
loss, acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print(f"\n✅ Validation Accuracy: {acc:.4f}")

# ------------------------------
# Save Model and Preprocessors
# ------------------------------
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n🎉 Model, Scaler, and Label Encoder saved successfully!")
