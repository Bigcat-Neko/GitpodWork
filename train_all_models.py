#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
import pickle

# Force TensorFlow to use CPU only.
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# TensorFlow / Keras imports
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding, GlobalMaxPooling1D, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Scikit-learn and others
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# LightGBM, XGBoost, CatBoost
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Prophet (install via: pip install prophet)
from prophet import Prophet

# Reinforcement Learning (Stable-Baselines3)
import gym
from stable_baselines3 import PPO, DQN, A2C, SAC

# Import your custom modules for the trading environment and feature engineering
# Make sure trading_env.py and preprocess_features.py are available in the same directory.
from trading_env import TradingEnv
from preprocess_features import create_features_targets

# =============================================================================
# Utility: Load Price Data
# =============================================================================
DATA_PATH = "data/forex_data_preprocessed.csv"
def load_price_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date", "fetched_at"])

# =============================================================================
# 1. Deep Learning Models for Price Data
# =============================================================================

def train_lstm():
    print("\n===== Training LSTM Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    # Reshape for sequence models (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/lstm_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
              callbacks=callbacks, verbose=1)
    model.save("models/lstm_model.keras")
    print("LSTM model saved as models/lstm_model.keras")

def train_gru():
    print("\n===== Training GRU Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(50, return_sequences=True),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/gru_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
              callbacks=callbacks, verbose=1)
    model.save("models/gru_model.keras")
    print("GRU model saved as models/gru_model.keras")

def train_cnn():
    print("\n===== Training CNN Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        tf.keras.Input(shape=(X_train.shape[1], 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/cnn_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
              callbacks=callbacks, verbose=1)
    model.save("models/cnn_model.keras")
    print("CNN model saved as models/cnn_model.keras")

def train_transformer():
    print("\n===== Training Transformer Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_layer = Input(shape=(X_train.shape[1], 1))
    attn_output = MultiHeadAttention(num_heads=2, key_dim=32)(input_layer, input_layer)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + input_layer)
    ffn_output = Dense(64, activation="relu")(attn_output)
    ffn_output = Dense(1)(ffn_output)
    gap = GlobalMaxPooling1D()(ffn_output)
    output_layer = Dense(1, activation="linear")(gap)
    transformer_model = Model(inputs=input_layer, outputs=output_layer)
    transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/transformer_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    transformer_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                           callbacks=callbacks, verbose=1)
    transformer_model.save("models/transformer_model.keras")
    print("Transformer model saved as models/transformer_model.keras")

# =============================================================================
# 2. Sentiment Model
# =============================================================================
def train_sentiment():
    print("\n===== Training Sentiment Model =====")
    sentiment_path = "data/sentiment_data.csv"
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"{sentiment_path} not found. Please ensure sentiment data is available.")
    df = pd.read_csv(sentiment_path)
    texts = df['text'].astype(str).values
    labels = df['sentiment'].values

    max_words = 3000
    max_len = 80

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        tf.keras.Input(shape=(max_len,)),
        Embedding(input_dim=max_words, output_dim=64),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
        ModelCheckpoint("models/sentiment_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val),
              callbacks=callbacks, verbose=1)
    joblib.dump(tokenizer, "models/sentiment_tokenizer.pkl")
    model.save("models/sentiment_model.keras")
    print("Sentiment model saved as models/sentiment_model.keras")
    print("Sentiment tokenizer saved as models/sentiment_tokenizer.pkl")

# =============================================================================
# 3. Price Scaler Model
# =============================================================================
def train_price_scaler():
    print("\n===== Training Price Scaler Model =====")
    data = load_price_data()
    prices = data['close'].values.reshape(-1, 1)
    scaler = RobustScaler()
    scaler.fit(prices)
    joblib.dump(scaler, "models/price_scaler.pkl")
    print("Price scaler saved as models/price_scaler.pkl")

# =============================================================================
# 4. Anomaly Model (Autoencoder)
# =============================================================================
def train_anomaly():
    print("\n===== Training Anomaly Model =====")
    data = load_price_data()
    prices = data['close'].values.reshape(-1, 1)
    window_size = 10
    X = [prices[i:i+window_size].flatten() for i in range(len(prices) - window_size)]
    X = np.array(X)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]
    encoding_dim = 5
    autoencoder = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(encoding_dim, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), verbose=1)
    autoencoder.save("models/anomaly_model.keras")
    print("Anomaly model saved as models/anomaly_model.keras")

# =============================================================================
# 5. Traditional ML Models
# =============================================================================
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
    lgb_model = lgb.train(params, train_data, num_boost_round=200,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=20)])
    y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("LightGBM Validation RMSE:", rmse)
    lgb_model.save_model("models/lightgbm_model.txt")
    print("LightGBM model saved as models/lightgbm_model.txt")

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
    y_pred = xgb_model.predict(dval, ntree_limit=xgb_model.best_iteration + 1)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("XGBoost Validation RMSE:", rmse)
    xgb_model.save_model("models/xgboost_model.json")
    print("XGBoost model saved as models/xgboost_model.json")

def train_catboost():
    print("\n===== Training CatBoost Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    cat_model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, verbose=50)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
    y_pred = cat_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("CatBoost Validation RMSE:", rmse)
    cat_model.save_model("models/catboost_model.cbm")
    print("CatBoost model saved as models/catboost_model.cbm")

def train_svm():
    print("\n===== Training SVR Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("SVR Validation RMSE:", rmse)
    joblib.dump(svr, "models/svr_model.pkl")
    print("SVR model saved as models/svr_model.pkl")

def train_gaussian_process():
    print("\n===== Training Gaussian Process Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    kernel = RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(X_train, y_train)
    y_pred, std = gp.predict(X_val, return_std=True)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("Gaussian Process Validation RMSE:", rmse)
    joblib.dump(gp, "models/gaussian_process_model.pkl")
    print("Gaussian Process model saved as models/gaussian_process_model.pkl")

def train_stacking():
    print("\n===== Training Stacking Model =====")
    data = load_price_data()
    X, y = create_features_targets(data, window_size=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    meta_model = LinearRegression()
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("Stacking Model Validation RMSE:", rmse)
    joblib.dump(stacking_model, "models/stacking_model.pkl")
    print("Stacking model saved as models/stacking_model.pkl")

# =============================================================================
# 6. Prophet Model (Time Series Forecasting)
# =============================================================================
def train_prophet():
    print("\n===== Training Prophet Model =====")
    data = load_price_data()
    df_prophet = data[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    m = Prophet()
    m.fit(df_prophet)
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(m, f)
    print("Prophet model saved as models/prophet_model.pkl")

# =============================================================================
# 7. Reinforcement Learning Models
# =============================================================================
def train_ppo():
    print("\n===== Training PPO Model =====")
    env = TradingEnv()  # Ensure TradingEnv has a proper interface
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("models/ppo_model")
    print("PPO model saved as models/ppo_model")

def train_dqn():
    print("\n===== Training DQN Model =====")
    env = TradingEnv()
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("models/dqn_model")
    print("DQN model saved as models/dqn_model")

def train_a2c():
    print("\n===== Training A2C Model =====")
    env = TradingEnv()
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("models/a2c_model")
    print("A2C model saved as models/a2c_model")

def train_sac():
    print("\n===== Training SAC Model =====")
    env = TradingEnv()  # Ensure that your environment supports the SAC settings
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("models/sac_model")
    print("SAC model saved as models/sac_model")

# =============================================================================
# 8. Graph Neural Network (Placeholder)
# =============================================================================
def train_gnn():
    print("\n===== Training Graph Neural Network Model =====")
    print("GNN training requires specialized libraries (e.g., PyTorch Geometric or DGL).")
    print("This is a placeholder. Implement GNN training when you have the proper dataset and libraries.")

# =============================================================================
# Main: Train All Models
# =============================================================================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    # Deep learning models
    train_lstm()
    train_gru()
    train_cnn()
    train_transformer()
    
    # Sentiment model
    train_sentiment()
    
    # Preprocessing models
    train_price_scaler()
    train_anomaly()
    
    # Traditional ML models
    train_lightgbm()
    train_xgboost()
    train_catboost()
    train_svm()
    train_gaussian_process()
    train_stacking()
    
    # Prophet forecasting model
    train_prophet()
    
    # Reinforcement Learning models
    train_ppo()
    train_dqn()
    train_a2c()
    train_sac()
    
    # Placeholder for GNN training
    train_gnn()
    
    print("\nAll models have been trained and saved.")
