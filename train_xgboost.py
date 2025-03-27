#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_xgboost():
    print("\n===== Training XGBoost Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 6
    }
    evallist = [(dval, 'eval')]
    xgb_model = xgb.train(params, dtrain, num_boost_round=200, evals=evallist, early_stopping_rounds=20)
    y_pred = xgb_model.predict(dval, ntree_limit=xgb_model.best_ntree_limit)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("XGBoost Validation RMSE:", rmse)
    xgb_model.save_model("models/xgboost_model.json")
    print("XGBoost model saved as models/xgboost_model.json")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_xgboost()
