#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])

def train_cnn():
    print("\n===== Training CNN Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
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
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/cnn_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    print("CNN model saved as models/cnn_model.keras")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_cnn()
