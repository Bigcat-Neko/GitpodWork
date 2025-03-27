#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_lightgbm():
    print("\n===== Training LightGBM Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    lgb_model = lgb.train(params, train_data, num_boost_round=200, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=20)])
    y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("LightGBM Validation RMSE:", rmse)
    lgb_model.save_model("models/lightgbm_model.txt")
    print("LightGBM model saved as models/lightgbm_model.txt")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_lightgbm()
