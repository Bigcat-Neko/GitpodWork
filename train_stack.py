# train_stacking.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocess_features import create_features_targets
import joblib

# Load data from the data folder
data = pd.read_csv("data/forex_data_preprocessed.csv", parse_dates=["date", "fetched_at"])
X, y = create_features_targets(data, window_size=10)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models for regression
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    # You could also add a LightGBM model here if desired.
]

# Define the meta-model (a simple linear regressor)
meta_model = LinearRegression()

# Create the stacking regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

print("Training stacking model...")
stacking_model.fit(X_train, y_train)

# Evaluate the stacking model
y_pred = stacking_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Stacking Model Validation RMSE:", rmse)

# Save the stacking model using joblib
joblib.dump(stacking_model, "models/stacking_model.pkl")
print("Stacking model saved as models/stacking_model.pkl")
