#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_anomaly():
    print("\n===== Training Anomaly Model =====")
    data = load_price_data()
    prices = data['close'].values.reshape(-1, 1)
    window_size = 10
    X = np.array([prices[i:i+window_size].flatten() for i in range(len(prices) - window_size)])
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]
    encoding_dim = 5
    model = Sequential([
        Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), verbose=1)
    model.save("models/anomaly_model.keras")
    print("Anomaly model saved as models/anomaly_model.keras")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_anomaly()
