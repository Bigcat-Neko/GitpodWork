#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_svr():
    print("\n===== Training SVR Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("SVR Validation RMSE:", rmse)
    joblib.dump(model, "models/svr_model.pkl")
    print("SVR model saved as models/svr_model.pkl")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_svr()
