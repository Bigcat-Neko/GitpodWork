# train_conflict_models.py

import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from sklearn.svm import SVR
import joblib  # Add this as line 1 or near the top
os.makedirs('models', exist_ok=True)

def load_data_conflict():
    print("⎯⎯"*40)
    print("Loading and validating data...")
    try:
        # Read CSV with BOM handling and column verification
        df = pd.read_csv('live_data.csv', encoding='utf-8-sig')
        
        # Store original columns for debugging
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
        cleaned_columns = df.columns.tolist()
        
        print(f"\nOriginal columns: {original_columns}")
        print(f"Cleaned columns: {cleaned_columns}")
        
        # Verify required columns
        required_cols = {'datetime', 'open', 'high', 'low', 'close', 'symbol'}
        missing = required_cols - set(cleaned_columns)
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert data types
        numeric_cols = ['open', 'high', 'low', 'close']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['symbol', 'datetime']).dropna()
        
        print("\nFirst 3 rows of processed data:")
        print(df[['datetime', 'symbol', 'open', 'high', 'low', 'close']].head(3))
        
        X = df[['open', 'high', 'low']]
        y = df['close']
        return X, y

    except Exception as e:
        print(f"\n❌ CRITICAL DATA ERROR: {str(e)}")
        print("🔍 Troubleshooting Guide:")
        print("1. Verify CSV header exactly matches: datetime,open,high,low,close,symbol")
        print("2. Check for hidden spaces/characters in column names")
        print("3. Ensure numeric columns contain valid numbers")
        print("4. Confirm datetime format: YYYY-MM-DD HH:MM:SS")
        print("5. Validate comma separation (no semicolons/tabs)")
        raise SystemExit(1)

def train_gru(X, y):
    print("\n" + "⎯⎯"*40)
    print("Training GRU Time Series Model...")
    X_gru = X.values.reshape((X.shape[0], X.shape[1], 1))
    model = keras.Sequential([
        layers.GRU(64, activation='tanh', return_sequences=False, input_shape=(X_gru.shape[1], 1)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    model.fit(X_gru, y, epochs=10, batch_size=64, verbose=1)
    model.save("models/gru_model.keras")
    print("✅ GRU Model Saved")

def train_prophet_models(y):
    print("\n" + "⎯⎯"*40)
    print("Training Prophet with Stacked Regression...")
    df_prophet = pd.DataFrame({
        'ds': pd.read_csv('live_data.csv', encoding='utf-8-sig')['datetime'],
        'y': y.values.flatten()
    })
    
    # Prophet Model
    prophet = Prophet(weekly_seasonality=False, daily_seasonality=True)
    prophet.fit(df_prophet)
    joblib.dump(prophet, "models/prophet_base.pkl")
    
    # Prophet Stacking
    future = prophet.make_future_dataframe(periods=24, freq='H')
    forecast = prophet.predict(future)
    stacking = LinearRegression()
    stacking.fit(forecast[['trend', 'yhat']].values[:len(y)], y)
    joblib.dump(stacking, "models/prophet_stacking.pkl")
    print("✅ Prophet Models Saved")

def train_autoencoder(X):
    print("\n" + "⎯⎯"*40)
    print("Training Anomaly Detection Autoencoder...")
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(X.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, X, 
                      epochs=50, 
                      batch_size=256,
                      validation_split=0.2,
                      verbose=1)
    model.save("models/autoencoder.keras")
    print("✅ Autoencoder Saved")

def train_classifiers(X, y):
    print("\n" + "⎯⎯"*40)
    print("Training Classification Models...")
    y_class = (y.shift(-1) > y).astype(int).dropna()
    X_class = X.iloc[:-1]
    
    # CatBoost Classifier
    cb = CatBoostClassifier(iterations=1000, 
                          learning_rate=0.05,
                          depth=6,
                          silent=True)
    cb.fit(X_class, y_class)
    cb.save_model("models/catboost_clf.cbm")
    
    # XGBoost Classifier
    xgb = XGBClassifier(n_estimators=200,
                      max_depth=5,
                      learning_rate=0.1,
                      use_label_encoder=False,
                      eval_metric='logloss')
    xgb.fit(X_class, y_class)
    joblib.dump(xgb, "models/xgb_clf.pkl")
    
    # LightGBM Classifier
    lgbm = lgb.LGBMClassifier(num_leaves=31,
                            learning_rate=0.05,
                            n_estimators=200)
    lgbm.fit(X_class, y_class)
    joblib.dump(lgbm, "models/lgbm_clf.pkl")
    print("✅ Classification Models Saved")

def train_regressors(X, y):
    print("\n" + "⎯⎯"*40)
    print("Training Regression Models...")
    # XGBoost Regressor
    xgb = XGBRegressor(n_estimators=200,
                     max_depth=5,
                     learning_rate=0.1)
    xgb.fit(X, y)
    joblib.dump(xgb, "models/xgb_reg.pkl")
    
    # LightGBM Regressor
    lgbm = lgb.LGBMRegressor(num_leaves=31,
                           learning_rate=0.05,
                           n_estimators=200)
    lgbm.fit(X, y)
    joblib.dump(lgbm, "models/lgbm_reg.pkl")
    
    # SVR with feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svr = SVR(C=1.0, epsilon=0.1)
    svr.fit(X_scaled, y.values.ravel())
    joblib.dump((scaler, svr), "models/svr_reg.pkl")
    print("✅ Regression Models Saved")

def train_cnn(X, y):
    print("\n" + "⎯⎯"*40)
    print("Training CNN for Price Movement Prediction...")
    y_cnn = (y.shift(-1) > y).astype(int).dropna().values
    X_cnn = X.iloc[:-1].values
    
    model = keras.Sequential([
        layers.Reshape((X_cnn.shape[1], 1), input_shape=(X_cnn.shape[1],)),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.fit(X_cnn, y_cnn, 
            epochs=20,
            batch_size=128,
            validation_split=0.2,
            verbose=1)
    model.save("models/cnn_model.keras")
    print("✅ CNN Model Saved")

def train_transformer(X, y):
    print("\n" + "⎯⎯"*40)
    print("Training Transformer Model...")
    y_trans = (y.shift(-1) > y).astype(int).dropna().values
    X_trans = X.iloc[:-1].values
    
    inputs = keras.Input(shape=(X_trans.shape[1],))
    x = layers.Embedding(input_dim=10000, output_dim=128)(inputs)
    x = layers.Transformer(num_heads=4, 
                         key_dim=64, 
                         dropout=0.1)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.fit(X_trans, y_trans,
            epochs=15,
            batch_size=256,
            validation_split=0.2,
            verbose=1)
    model.save("models/transformer.keras")
    print("✅ Transformer Model Saved")

def main():
    print("⎯⎯"*40)
    print("Multi-Model Training System v2.0")
    print("⎯⎯"*40)
    X, y = load_data_conflict()
    start_time = time.time()
    
    # Model Training Pipeline
    train_gru(X, y)
    train_prophet_models(y)
    train_autoencoder(X)
    train_classifiers(X, y)
    train_regressors(X, y)
    train_cnn(X, y)
    train_transformer(X, y)
    
    # Final Report
    print("\n" + "⎯⎯"*40)
    print("Training Complete - All Models Saved")
    print(f"Total Execution Time: {(time.time() - start_time)/60:.2f} minutes")
    print("Saved Models Directory:")
    print("\n".join([f"• {f}" for f in os.listdir('models')]))
    print("⎯⎯"*40)

if __name__ == "__main__":
    main()