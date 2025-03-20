# train_models.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------------
# 1. Load Preprocessed Forex Data
# -----------------------------------------------
print("Loading preprocessed Forex data...")
df = pd.read_csv("data/forex_data_preprocessed.csv")
print("Columns in preprocessed data:", df.columns)

if "close" not in df.columns:
    raise Exception("The preprocessed data must have a 'close' column for price data.")

prices = df["close"].values.reshape(-1, 1)

# -----------------------------------------------
# 2. Normalize the Price Data
# -----------------------------------------------
print("Normalizing price data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Save the scaler for later use during inference
with open("models/price_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Price scaler saved to 'models/price_scaler.pkl'.")

# -----------------------------------------------
# 3. Prepare Data for LSTM (Sequences and Labels)
# -----------------------------------------------
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        # Label: 1 if the price at time (i+look_back) is higher than at (i+look_back-1)
        y.append(1 if dataset[i+look_back, 0] > dataset[i+look_back-1, 0] else 0)
    return np.array(X), np.array(y)

look_back = 60
X_data, y_data = create_dataset(scaled_prices, look_back)
X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))  # Shape for LSTM

# -----------------------------------------------
# 4. Build and Train the LSTM Model
# -----------------------------------------------
print("Training LSTM model for price prediction...")
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

# Use EarlyStopping to avoid overfitting: if loss does not improve for 5 epochs, stop training.
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

lstm_model.fit(X_data, y_data, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])
lstm_model.save("models/trade_signal_model.h5")
print("LSTM model trained and saved as 'models/trade_signal_model.h5'.")

# -----------------------------------------------
# 5. Load a Real Sentiment Analysis Model
# -----------------------------------------------
print("Loading real sentiment analysis model from Hugging Face...")
# IMPORTANT: Ensure you have installed tf-keras:
#    pip install tf-keras
from transformers import pipeline
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    raise Exception("Error loading sentiment analysis model: " + str(e))

# Save the sentiment model using pickle (note: pickle isn't ideal for large models but is acceptable for prototyping)
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_pipeline, f)
print("Sentiment analysis model saved as 'models/sentiment_model.pkl'.")

# -----------------------------------------------
# 6. Train an Anomaly Detection Model (Placeholder)
# -----------------------------------------------
print("Training anomaly detection model...")
# For simplicity, generate dummy features using the last look_back prices for each training sample.
X_anomaly = np.random.rand(len(scaled_prices) - look_back, look_back)
# Dummy binary labels for anomaly (in a real scenario, use a proper method)
y_anomaly = np.random.randint(0, 2, size=(len(scaled_prices) - look_back,))
anomaly_model = RandomForestClassifier()
anomaly_model.fit(X_anomaly, y_anomaly)
with open("models/anomaly_model.pkl", "wb") as f:
    pickle.dump(anomaly_model, f)
print("Anomaly detection model trained and saved as 'models/anomaly_model.pkl'.")

print("All models have been trained and saved successfully!")
