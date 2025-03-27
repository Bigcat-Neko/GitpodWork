#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])

def train_transformer():
    print("\n===== Training Transformer Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    input_layer = Input(shape=(X_train.shape[1], 1))
    attn_output = MultiHeadAttention(num_heads=2, key_dim=32)(input_layer, input_layer)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + input_layer)
    ffn_output = Dense(64, activation="relu")(attn_output)
    ffn_output = Dense(1)(ffn_output)
    gap = GlobalMaxPooling1D()(ffn_output)
    output_layer = Dense(1, activation="linear")(gap)
    transformer_model = Model(inputs=input_layer, outputs=output_layer)
    transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/transformer_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    transformer_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    print("Transformer model saved as models/transformer_model.keras")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_transformer()
