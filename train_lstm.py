# train_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets  # renamed file from your previous preprocess.py

# Load data from data folder
data = pd.read_csv("data/forex_data_preprocessed.csv", parse_dates=["date", "fetched_at"])
X, y = create_features_targets(data, window_size=10)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
print("Training LSTM model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
model.save("models/lstm_model.h5")
print("LSTM model saved as models/lstm_model.h5")
