import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load your preprocessed data (ensure it has a 'close' column)
df = pd.read_csv("data/forex_data_preprocessed.csv")
if "close" not in df.columns:
    raise Exception("The preprocessed data must have a 'close' column.")
prices = df["close"].values.reshape(-1, 1)

# Normalize the price data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Save the scaler for later use in inference
with open("models/price_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Price scaler saved.")

# Create sequences for GRU training (using a look-back of 60)
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        # Label: 1 if next price is higher than previous, else 0
        y.append(1 if dataset[i+look_back, 0] > dataset[i+look_back-1, 0] else 0)
    return np.array(X), np.array(y)

look_back = 60
X_data, y_data = create_dataset(scaled_prices, look_back)
X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))  # for GRU input

# ------------------------------
# Step 1.2: Build and Train the GRU Model
# ------------------------------
gru_model = Sequential([
    GRU(50, return_sequences=True, input_shape=(look_back, 1)),
    GRU(50),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

# Train extensively (e.g., 50 epochs, with early stopping)
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
gru_model.fit(X_data, y_data, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])
gru_model.save("models/gru_model.h5")
print("GRU model trained and saved as 'models/gru_model.h5'.")
