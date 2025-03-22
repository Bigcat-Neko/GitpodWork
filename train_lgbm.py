# train_lightgbm.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess_features import create_features_targets

# Load data from the data folder
data = pd.read_csv("data/forex_data_preprocessed.csv", parse_dates=["date", "fetched_at"])
X, y = create_features_targets(data, window_size=10)

# For LightGBM, our X is already in a flat format (samples, window_size)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Set training parameters (tweak these for production tuning)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

print("Training LightGBM model...")
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

# Evaluate model
y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("LightGBM Validation RMSE:", rmse)

# Save the model
lgb_model.save_model("models/lightgbm_model.txt")
print("LightGBM model saved as models/lightgbm_model.txt")
