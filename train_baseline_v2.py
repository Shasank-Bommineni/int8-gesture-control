import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Load Dataset
# NEW
DATASET_PATH = r"C:\Users\Shasank\OneDrive\Desktop\2\dataset\master_gesture_dataset.csv"
df = pd.read_csv(DATASET_PATH)

feature_cols = [c for c in df.columns if c.startswith("feat_")] + ["is_right_hand"]
X = df[feature_cols].values.astype(np.float32)
y = df["label"].values.astype(np.int32)

# 2. Split Data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 3. Fit Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler parameters for live inference in Stage 6
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 4. Model Architecture with Regularization
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
]

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=120,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n==========================================")
print(f" NEW FP32 TEST ACCURACY (SCALED): {test_acc * 100:.2f}%")
print(f"==========================================")

# Save updated assets
model.save("models/baseline_fp32.keras")
np.savez("models/test_data.npz", X_test=X_test_scaled, y_test=y_test)
print("[SAVED] Updated models/baseline_fp32.keras, scaler.pkl, and test_data.npz")