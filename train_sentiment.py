#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def train_sentiment():
    print("\n===== Training Sentiment Model =====")
    sentiment_path = "data/sentiment_data.csv"
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"{sentiment_path} not found.")
    df = pd.read_csv(sentiment_path)
    texts = df['text'].astype(str).values
    labels = df['sentiment'].values
    max_words = 5000
    max_len = 100
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)
    y = labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=max_words, output_dim=128),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/sentiment_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    joblib.dump(tokenizer, "models/sentiment_tokenizer.pkl")
    print("Sentiment model saved as models/sentiment_model.keras")
    print("Sentiment tokenizer saved as models/sentiment_tokenizer.pkl")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_sentiment()
