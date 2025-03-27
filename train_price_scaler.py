#!/usr/bin/env python
import os
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_price_scaler():
    print("\n===== Training Price Scaler Model =====")
    data = load_price_data()
    prices = data['close'].values.reshape(-1, 1)
    scaler = RobustScaler()
    scaler.fit(prices)
    joblib.dump(scaler, "models/price_scaler.pkl")
    print("Price scaler saved as models/price_scaler.pkl")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_price_scaler()
