# preprocess_features.py
import numpy as np
import pandas as pd

def create_features_targets(data, window_size=10):
    """
    Creates features and targets for time series prediction.
    For each point in time, creates features from the past window_size closing prices.
    The target is the closing price of the day following the window.
    """
    X, y = [], []
    prices = data["close"].values
    for i in range(len(prices) - window_size):
        # Features: sliding window of past prices
        X.append(prices[i:i+window_size])
        # Target: the next day's closing price
        y.append(prices[i+window_size])
    return np.array(X), np.array(y)

# For testing when run directly:
if __name__ == "__main__":
    data = pd.read_csv("data/forex_data_preprocessed.csv", parse_dates=["date", "fetched_at"])
    X, y = create_features_targets(data, window_size=10)
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
