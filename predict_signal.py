import numpy as np
import tensorflow as tf  # or torch if using PyTorch
from fetch_live_prices import get_live_forex_prices  # Import the function

# Load trained AI model
model = tf.keras.models.load_model("lstm_model.h5")  # Change for GRU/RL

# Get live data and make a prediction
live_prices = get_live_forex_prices()
predicted_signal = model.predict(live_prices.reshape(1, live_prices.shape[0], 1))

print("Predicted Trade Signal:", predicted_signal)
