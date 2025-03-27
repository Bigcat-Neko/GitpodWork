#!/usr/bin/env python
import os
import pandas as pd
import pickle
from prophet import Prophet
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_prophet():
    print("\n===== Training Prophet Model =====")
    data = load_price_data()
    df_prophet = data[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Prophet model saved as models/prophet_model.pkl")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_prophet()
