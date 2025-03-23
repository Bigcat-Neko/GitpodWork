# train_cnn.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets

data = pd.read_csv("data/forex_data_preprocessed.csv", parse_dates=["date", "fetched_at"])
X, y = create_features_targets(data, window_size=10)
X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = create_cnn_model((X_train.shape[1], X_train.shape[2]))
print("Training CNN model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
model.save("models/cnn_model.h5")
print("CNN model saved as models/cnn_model.h5")
