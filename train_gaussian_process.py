#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocess_features import create_features_targets
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])
def train_gaussian_process():
    print("\n===== Training Gaussian Process Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    kernel = RBF(length_scale=1.0)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    model.fit(X_train, y_train)
    y_pred, std = model.predict(X_val, return_std=True)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("Gaussian Process Validation RMSE:", rmse)
    joblib.dump(model, "models/gaussian_process_model.pkl")
    print("Gaussian Process model saved as models/gaussian_process_model.pkl")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_gaussian_process()
