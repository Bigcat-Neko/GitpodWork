# train_rl.py
import numpy as np
import pandas as pd  # Import pandas as pd
from trading_env import TradingEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import pickle

# Load historical price data from your preprocessed CSV file
df = pd.read_csv("data/forex_data_preprocessed.csv")
if "close" not in df.columns:
    raise Exception("The CSV file must have a 'close' column.")
prices = df["close"].values.reshape(-1, 1)

# Load your scaler to normalize prices
with open("models/price_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
# Normalize prices and flatten to 1D array
scaled_prices = scaler.transform(prices).flatten()

# Create the trading environment using the normalized prices
env = TradingEnv(scaled_prices)

# Optional: Check if the environment follows Gymnasium’s interface
check_env(env, warn=True)

# Initialize the DQN agent using a multilayer perceptron policy
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent extensively (e.g., 100,000 timesteps)
model.learn(total_timesteps=100000)

# Save the trained RL model
model.save("models/trading_dqn")
print("RL agent trained and saved as 'models/trading_dqn'")
