import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load the CSV data
data = pd.read_csv("training_data.csv")

# Ensure that we only work on non-gold assets if desired:
data = data[data["asset"] != "XAU/USD"]

# For this example, we create a simple feature:
# Assume our target is the percentage change from current close to next close.
data["close_shifted"] = data.groupby("asset")["close"].shift(-1)
data.dropna(inplace=True)
data["target"] = (data["close_shifted"] - data["close"]) / data["close"]

# Use the 'close' price as the only feature for now.
features = data[["close"]]
target = data["target"]

# Split into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model with small parameters for quick training.
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=50,      # Fewer trees for a quick training run.
    learning_rate=0.1,
    max_depth=3,          # Shallow trees for faster training.
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Test Mean Squared Error: {mse}")

# Save the model to a file
os.makedirs("models", exist_ok=True)
model.save_model("models/xgboost_model.json")
print("XGBoost model saved as models/xgboost_model.json")
