import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ----- Function to Add Selected Technical Indicators Using pandas_ta -----
def add_selected_ta_indicators(df):
    # Trend indicators:
    df["SMA_5"] = ta.sma(df["close"], length=5)
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["EMA_20"] = ta.ema(df["close"], length=20)
    
    # Momentum indicators:
    df["RSI"] = ta.rsi(df["close"], length=14)
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["STOCH_K"] = stoch["STOCHk_14_3_3"]
    df["STOCH_D"] = stoch["STOCHd_14_3_3"]
    
    # MACD: with its signal line.
    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd_df["MACD_12_26_9"]
    df["MACD_signal"] = macd_df["MACDs_12_26_9"]
    
    # Volatility indicators:
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    bb = ta.bbands(df["close"], length=20, std=2)
    df["BBL"] = bb["BBL_20_2.0"]  # lower band
    df["BBM"] = bb["BBM_20_2.0"]  # middle band
    df["BBU"] = bb["BBU_20_2.0"]  # upper band
    
    # Additional: ADX
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = adx_df["ADX_14"]
    
    return df

# ----- Function to Normalize Features per Asset -----
def normalize_features(df, feature_cols):
    scaler = StandardScaler()
    # Explicitly cast selected feature columns to float before scaling.
    df.loc[:, feature_cols] = scaler.fit_transform(df[feature_cols].astype(float))
    return df

# ----- Main Script -----

# Step 1: Load and sort your data.
data = pd.read_csv("training_data.csv", parse_dates=["datetime"])
data.sort_values("datetime", inplace=True)

# Ensure column names are lowercase for consistency.
data.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)

# Step 2: Group by asset and add selected technical indicators.
# The function is applied per asset so that indicators are computed separately.
data = data.groupby("asset", group_keys=False).apply(add_selected_ta_indicators)

# Step 3: Add time-based features.
data["day_of_week"] = data["datetime"].dt.dayofweek
day_dummies = pd.get_dummies(data["day_of_week"], prefix="dow")
data = pd.concat([data, day_dummies], axis=1)

# Step 4: (Re)Ensure SMA_5 exists (it should from our function).
if "SMA_5" not in data.columns:
    data["SMA_5"] = data.groupby("asset")["close"].transform(lambda x: x.rolling(window=5).mean())

# Step 5: Create the target variable.
# For each asset, target = 1 if next day's close is higher than today's, else 0.
data["Target"] = data.groupby("asset")["close"].shift(-1) > data["close"]
data["Target"] = data["Target"].astype(int)
data.dropna(inplace=True)

# Step 6: Check data balancing.
print("Data balancing (target counts):")
print(data["Target"].value_counts())

# Step 7: Select features.
# We'll use base price info, our selected TA indicators, and day-of-week dummies.
selected_features = [
    "close", "SMA_5", "SMA_20", "EMA_20", "RSI", "MACD", "MACD_signal",
    "ATR", "BBL", "BBM", "BBU", "ADX", "STOCH_K", "STOCH_D"
]
features = selected_features + list(day_dummies.columns)

X = data[features]
y = data["Target"]

# Step 8: Normalize features per asset.
X_normalized = data.groupby("asset", group_keys=False).apply(lambda df: normalize_features(df, features))
X_normalized = X_normalized[features]

print("Any missing values in features:", X_normalized.isnull().any().any())

# Step 9: Split into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 10: Train a RandomForest model with hyperparameter tuning.
rf = RandomForestClassifier(random_state=42)
param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, 7]}
grid_search = GridSearchCV(rf, param_grid, cv=3, error_score="raise")
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Best parameters:", grid_search.best_params_)
print("Test set accuracy:", accuracy)
