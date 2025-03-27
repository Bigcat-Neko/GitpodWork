#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_catboost():
    print("\n===== Training CatBoost Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, verbose=50)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("CatBoost Validation RMSE:", rmse)
    model.save_model("models/catboost_model.cbm")
    print("CatBoost model saved as models/catboost_model.cbm")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_catboost()
