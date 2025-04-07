# trading_system.py
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix telegram imports
sys.modules['telegram'] = __import__('telegram')
sys.modules['telegram.vendor.ptb_urllib3.urllib3'] = __import__('urllib3')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dotenv import load_dotenv
from twelvedata import TDClient
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import StackingClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
import vectorbt as vbt
import gym
from gym import spaces

# Initialize environment
load_dotenv()
os.makedirs('models', exist_ok=True)

# ========================
# Configuration
# ========================
ASSETS = ['EUR/USD', 'GBP/USD', 'BTC/USD', 'ETH/USD']
LOOKBACK = 60
INDICATORS = [
    'sma_10', 'sma_20', 'ema_10', 'ema_20', 'wma_20', 'hma_14', 
    'macd', 'rsi', 'stoch', 'adx', 'cci', 'willr', 'uo', 'mom', 
    'roc', 'atr', 'obv', 'cmf', 'vwap', 'bbands', 'kc', 'psar', 
    'supertrend', 'ichimoku', 'stc', 'vwma', 'pvo', 'efi', 'vortex'
]

MODEL_CONFIG = {
    'LSTM': {'units': 256, 'dropout': 0.3, 'epochs': 50, 'batch_size': 64},
    'GRU': {'units': 256, 'dropout': 0.3, 'epochs': 50, 'batch_size': 64},
    'CNN': {'filters': 128, 'kernel_size': 3, 'epochs': 30, 'batch_size': 64},
    'Transformer': {'heads': 4, 'key_dim': 32, 'epochs': 30, 'batch_size': 64},
    'XGBoost': {'n_estimators': 500, 'learning_rate': 0.01},
    'LightGBM': {'n_estimators': 500, 'learning_rate': 0.05},
    'CatBoost': {'iterations': 500, 'learning_rate': 0.03},
    'SVR': {'C': 1.0, 'kernel': 'rbf'},
    'Gaussian': {'n_restarts_optimizer': 3},
    'DQN': {'policy': 'MlpPolicy', 'learning_rate': 1e-4, 'total_timesteps': 10000},
    'PPO': {'policy': 'MlpPolicy', 'learning_rate': 2.5e-4, 'total_timesteps': 10000}
}

# ========================
# Custom Trading Environment
# ========================
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 0: sell, 1: hold, 2: buy
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(INDICATORS), ),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.df.iloc[self.current_step][INDICATORS].values.astype(np.float32)

    def _calculate_reward(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        next_price = self.df.iloc[self.current_step + 1]['close']
        price_change = (next_price - current_price) / current_price
        
        if action == 2:  # Buy
            return price_change
        elif action == 0:  # Sell
            return -price_change
        return 0  # Hold

# ========================
# Enhanced ModelForge Class
# ========================
class ModelForge:
    @staticmethod
    def create_model(model_type, input_shape):
        try:
            if model_type == 'LSTM':
                model = Sequential([
                    LSTM(MODEL_CONFIG['LSTM']['units'], input_shape=input_shape),
                    Dropout(MODEL_CONFIG['LSTM']['dropout']),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            elif model_type == 'GRU':
                model = Sequential([
                    GRU(MODEL_CONFIG['GRU']['units'], input_shape=input_shape),
                    Dropout(MODEL_CONFIG['GRU']['dropout']),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            elif model_type == 'CNN':
                model = Sequential([
                    Conv1D(MODEL_CONFIG['CNN']['filters'], MODEL_CONFIG['CNN']['kernel_size'], 
                          activation='relu', input_shape=input_shape),
                    GlobalMaxPooling1D(),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            elif model_type == 'Transformer':
                inputs = Input(shape=input_shape)
                x = MultiHeadAttention(num_heads=MODEL_CONFIG['Transformer']['heads'], 
                                     key_dim=MODEL_CONFIG['Transformer']['key_dim'])(inputs, inputs)
                x = LayerNormalization(epsilon=1e-6)(x)
                x = Dense(64, activation='relu')(x)
                outputs = Dense(1, activation='sigmoid')(x)
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            elif model_type == 'XGBoost':
                model = XGBClassifier(**MODEL_CONFIG['XGBoost'])
            
            elif model_type == 'LightGBM':
                model = LGBMClassifier(**MODEL_CONFIG['LightGBM'])
            
            elif model_type == 'CatBoost':
                model = CatBoostClassifier(**MODEL_CONFIG['CatBoost'], silent=True)
            
            elif model_type == 'SVR':
                model = SVC(**MODEL_CONFIG['SVR'], probability=True)
            
            elif model_type == 'Gaussian':
                model = GaussianProcessClassifier(**MODEL_CONFIG['Gaussian'])
            
            elif model_type == 'DQN':
                env = DummyVecEnv([lambda: TradingEnv(pd.DataFrame())])
                model = DQN(MODEL_CONFIG['DQN']['policy'], env, 
                           learning_rate=MODEL_CONFIG['DQN']['learning_rate'], verbose=0)
            
            elif model_type == 'PPO':
                env = DummyVecEnv([lambda: TradingEnv(pd.DataFrame())])
                model = PPO(MODEL_CONFIG['PPO']['policy'], env, 
                           learning_rate=MODEL_CONFIG['PPO']['learning_rate'], verbose=0)
            
            return model
        except Exception as e:
            print(f"Model creation failed for {model_type}: {str(e)}")
            return None

# ========================
# Data Engine
# ========================
class DataMaster:
    def __init__(self):
        self.td = TDClient(apikey=os.getenv('TWELVEDATA_API_KEY'))
        
    def fetch_data(self, symbol):
        for _ in range(3):  # Retry mechanism
            try:
                df = self.td.time_series(
                    symbol=symbol,
                    interval='1day',
                    start_date='2020-01-01',
                    outputsize=1000
                ).as_pandas()
                time.sleep(12)  # Rate limiting
                return df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                time.sleep(30)
        return None
    
    def add_indicators(self, df):
        try:
            df = df.drop(columns=[c for c in df.columns if c != 'close'], errors='ignore')
            
            for indicator in INDICATORS:
                if '_' in indicator:
                    func, length = indicator.split('_')
                    getattr(df.ta, func)(int(length), append=True)
                else:
                    getattr(df.ta, indicator)(append=True)
            
            return df.dropna()
        except Exception as e:
            print(f"Indicator error: {str(e)}")
            return df

# ========================
# Backtesting System
# ========================
class BacktestPro:
    def __init__(self):
        self.scaler = RobustScaler()
        
    def prepare_data(self, df):
        try:
            df['returns'] = df['close'].pct_change()
            df = df.dropna()
            features = self.scaler.fit_transform(df[INDICATORS])
            target = (df['returns'].shift(-1) > 0).astype(int).values[:-1]
            return features[:-1], target
        except Exception as e:
            print(f"Data preparation failed: {str(e)}")
            return None, None
    
    def train_model(self, model, model_type, X_train, y_train):
        try:
            if model_type in ['LSTM', 'GRU', 'CNN', 'Transformer']:
                X_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                model.fit(X_reshaped, y_train, 
                         epochs=MODEL_CONFIG[model_type]['epochs'],
                         batch_size=MODEL_CONFIG[model_type]['batch_size'],
                         verbose=0)
            elif model_type in ['DQN', 'PPO']:
                model.learn(total_timesteps=MODEL_CONFIG[model_type]['total_timesteps'])
            else:
                model.fit(X_train, y_train)
        except Exception as e:
            print(f"Training failed for {model_type}: {str(e)}")
    
    def backtest(self, model, model_type, X, y, prices):
        try:
            if model_type in ['LSTM', 'GRU', 'CNN', 'Transformer']:
                X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
                preds = model.predict(X_reshaped).flatten()
            elif model_type in ['DQN', 'PPO']:
                env = TradingEnv(pd.DataFrame({'close': prices}))
                obs = env.reset()
                preds = []
                while True:
                    action, _ = model.predict(obs)
                    preds.append(action)
                    obs, _, done, _ = env.step(action)
                    if done: break
                preds = np.array(preds)
            else:
                preds = model.predict_proba(X)[:, 1]
            
            entries = preds > 0.6 if model_type not in ['DQN', 'PPO'] else preds == 2
            exits = preds < 0.4 if model_type not in ['DQN', 'PPO'] else preds == 0
            
            pf = vbt.Portfolio.from_signals(
                prices,
                entries,
                exits,
                fees=0.001,
                slippage=0.005
            )
            return pf.sharpe_ratio()
        except Exception as e:
            print(f"Backtesting failed for {model_type}: {str(e)}")
            return 0
    
    def save_model(self, model, model_type, asset):
        try:
            asset_clean = asset.replace('/', '_')
            filename = f"models/{asset_clean}_{model_type}"
            
            if isinstance(model, Sequential):
                save_model(model, f"{filename}.keras")
            elif model_type == 'XGBoost':
                model.save_model(f"{filename}.json")
            elif model_type == 'LightGBM':
                model.booster_.save_model(f"{filename}.txt")
            elif model_type == 'CatBoost':
                model.save_model(f"{filename}.cbm")
            elif model_type in ['DQN', 'PPO']:
                model.save(f"{filename}.zip")
            else:
                joblib.dump(model, f"{filename}.pkl")
        except Exception as e:
            print(f"Model save failed for {model_type}: {str(e)}")

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    dm = DataMaster()
    bt = BacktestPro()
    
    for asset in ASSETS:
        print(f"\n{'='*40}\nProcessing {asset}\n{'='*40}")
        
        try:
            df = dm.fetch_data(asset)
            if df is None or df.empty:
                continue
                
            df = dm.add_indicators(df)
            if df is None or df.empty:
                continue
                
            X, y = bt.prepare_data(df)
            if X is None or y is None:
                continue
                
            prices = df['close'].iloc[1:]
            
            for model_type in MODEL_CONFIG.keys():
                print(f"\nTraining {model_type}...")
                try:
                    model = ModelForge.create_model(model_type, (1, X.shape[1]))
                    if model is None:
                        continue
                        
                    bt.train_model(model, model_type, X, y)
                    sharpe = bt.backtest(model, model_type, X, y, prices)
                    
                    if sharpe > 0.5:
                        bt.save_model(model, model_type, asset)
                        print(f"Saved {model_type} for {asset} (Sharpe: {sharpe:.2f})")
                        
                except Exception as e:
                    print(f"{model_type} training failed: {str(e)}")
                    
        except Exception as e:
            print(f"Fatal error processing {asset}: {str(e)}")