#!/usr/bin/env python
# trading_system.py

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import warnings
import datetime
from tabulate import tabulate  # For clear console output

warnings.filterwarnings("ignore", category=FutureWarning)

from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from twelvedata import TDClient
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler

# --- Classical ML and Other Libraries ---
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import StackingRegressor
from prophet import Prophet

# --- Deep Learning & Keras ---
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, GRU, Conv1D, Dense, Dropout, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam

# --- RL Libraries ---
import gym
from gym import spaces
from stable_baselines3 import DQN, PPO

import vectorbt as vbt

# Force TensorFlow CPU only.
tf.config.set_visible_devices([], 'GPU')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

load_dotenv()
os.makedirs('models', exist_ok=True)

##############################
# Global Configuration
##############################
FOREX_PAIRS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',
    'USD/CAD', 'NZD/USD', 'USD/SGD', 'USD/HKD', 'USD/TRY',
    'USD/MXN', 'USD/ZAR', 'USD/SEK', 'USD/NOK', 'USD/DKK'
]
CRYPTO_PAIRS = [
    'BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD',
    'ADA/USD', 'DOT/USD', 'LINK/USD', 'BNB/USD', 'XLM/USD',
    'DOGE/USD', 'SOL/USD', 'MATIC/USD', 'AVAX/USD', 'ATOM/USD'
]

MODEL_LIST = [
    'LSTM', 'GRU', 'CNN', 'Transformer', 'XGBoost', 'LightGBM',
    'CatBoost', 'Stacking', 'GaussianProcess', 'Prophet', 'Sentiment',
    'Anomaly', 'SVR', 'DQN', 'PPO'
]

INDICATOR_CONFIG = [
    {'kind': 'sma', 'length': 20}, {'kind': 'ema', 'length': 20},
    {'kind': 'wma', 'length': 20}, {'kind': 'hma', 'length': 20},
    {'kind': 'macd'}, {'kind': 'rsi'}, {'kind': 'stoch'},
    {'kind': 'adx'}, {'kind': 'cci'}, {'kind': 'vwap'},
    {'kind': 'atr'}, {'kind': 'bbands'}, {'kind': 'supertrend'},
    {'kind': 'ichimoku'}, {'kind': 'obv'}, {'kind': 'vwma'},
    {'kind': 'mom'}, {'kind': 'willr'}, {'kind': 'ppo'},
    {'kind': 'roc'}, {'kind': 'psar'}, {'kind': 'kvo'},
    {'kind': 'pvo'}, {'kind': 'efi'}, {'kind': 'vortex'}
]

MODEL_CONFIG = {
    'LSTM': {'units': 256, 'dropout': 0.3, 'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001},
    'GRU': {'units': 256, 'dropout': 0.3, 'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001},
    'CNN': {'filters': 128, 'kernel_size': 3, 'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001},
    'Transformer': {'heads': 4, 'key_dim': 32, 'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001},
    'XGBoost': {'n_estimators': 1000, 'learning_rate': 0.01},
    'LightGBM': {'num_boost_round': 500, 'learning_rate': 0.05},
    'CatBoost': {'iterations': 1000, 'learning_rate': 0.03, 'verbose': False},
    'Stacking': {},
    'GaussianProcess': {},
    'Prophet': {},
    'Sentiment': {'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001},
    'Anomaly': {'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001},
    'SVR': {'C': 1.0, 'epsilon': 0.1},
    'DQN': {'total_timesteps': 10000},
    'PPO': {'total_timesteps': 10000},
}

SAVE_CONFIG = {
    'LSTM':    {'ext': '.keras',  'method': 'tf'},
    'GRU':     {'ext': '.keras',  'method': 'tf'},
    'CNN':     {'ext': '.keras',  'method': 'tf'},
    'Transformer': {'ext': '.keras', 'method': 'tf'},
    'XGBoost': {'ext': '.json',   'method': 'xgb'},
    'LightGBM':{'ext': '.txt',    'method': 'lgb'},
    'CatBoost':{'ext': '.cbm',    'method': 'catboost'},
    'Stacking':{'ext': '.pkl',    'method': 'joblib'},
    'GaussianProcess':{'ext': '.pkl', 'method': 'joblib'},
    'Prophet': {'ext': '.pkl',    'method': 'joblib'},
    'Sentiment':{'ext': '.keras',  'method': 'tf'},
    'Anomaly': {'ext': '.keras',  'method': 'tf'},
    'SVR':     {'ext': '.pkl',    'method': 'joblib'},
    'DQN':     {'ext': '.zip',    'method': 'stable_baselines3'},
    'PPO':     {'ext': '.zip',    'method': 'stable_baselines3'},
}

WINDOW_SIZE = 10

##############################
# Custom Trading Environment for RL Models
##############################
class TradingEnv(gym.Env):
    """
    A simple trading environment for RL.
    Observation: current normalized price and technical features.
    Action: 0 (sell) or 1 (buy).
    Reward: daily return * (2*action - 1).
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, prices, features):
        super().__init__()
        self.prices = prices.reset_index(drop=True)
        self.features = features
        self.current_step = 0
        self.total_steps = len(prices) - 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1] + 1,), dtype=np.float32)
        
    def reset(self):
        self.current_step = 0
        return self._get_observation()
    
    def _get_observation(self):
        price = self.prices.iloc[self.current_step]
        feat = self.features[self.current_step]
        return np.concatenate(([price], feat))
    
    def step(self, action):
        current_price = self.prices.iloc[self.current_step]
        next_price = self.prices.iloc[self.current_step + 1]
        reward = (next_price - current_price) if action == 1 else (current_price - next_price)
        self.current_step += 1
        done = self.current_step >= self.total_steps
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}
    
    def render(self, mode='human', close=False):
        pass

##############################
# Utility Functions
##############################
def create_sequences(X, y, window_size=WINDOW_SIZE):
    seq_X, seq_y = [], []
    y = np.array(y)
    for i in range(len(X) - window_size + 1):
        seq_X.append(X[i:i+window_size])
        seq_y.append(y[i+window_size-1])
    return np.array(seq_X), np.array(seq_y)

##############################
# Data Master
##############################
class DataMaster:
    def __init__(self):
        self.td = TDClient(apikey=os.getenv('TWELVEDATA_API_KEY'))
        self.scaler = RobustScaler()
        self.last_fetch_time = 0
        self.delay = 1.5

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def fetch_data(self, symbol):
        elapsed = time.time() - self.last_fetch_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        df = self.td.time_series(
            symbol=symbol,
            interval='1day',
            start_date='2010-01-01',
            outputsize=5000
        ).as_pandas()
        self.last_fetch_time = time.time()
        df.sort_index(inplace=True)
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"Missing essential column: {col}")
        if 'volume' not in df.columns:
            logging.info(f"{symbol}: Volume column not found. Adding a default volume column.")
            df['volume'] = 1
        # Apply technical indicators from INDICATOR_CONFIG
        try:
            df.ta.strategy(ta.Strategy(name="Dynamic Strategy", description="Auto indicators", ta=INDICATOR_CONFIG))
        except Exception as e:
            logging.error(f"Indicator error for {symbol}: {str(e)}")
        return df

    def preprocess_data(self, df):
        if df is None or df.empty:
            logging.error("Empty dataframe.")
            return None, None, None
        df = df.ffill().bfill().dropna(axis=1, how='all')
        if len(df) < 100:
            logging.error("Not enough data after preprocessing.")
            return None, None, None
        features = self.scaler.fit_transform(df.drop(['close'], axis=1, errors='ignore'))
        target = (df['close'].pct_change().shift(-1) > 0).astype(int)
        return features[:-1], target[:-1], df['close'].iloc[1:]

##############################
# Model Factory
##############################
class ModelFactory:
    @staticmethod
    def create_model(model_type, input_shape=None):
        try:
            if model_type == 'LSTM':
                inp = Input(shape=input_shape)
                x = LSTM(MODEL_CONFIG['LSTM']['units'])(inp)
                x = Dropout(MODEL_CONFIG['LSTM']['dropout'])(x)
                out = Dense(1, activation='sigmoid')(x)
                model = Model(inputs=inp, outputs=out)
                model.compile(optimizer=Adam(learning_rate=MODEL_CONFIG['LSTM']['learning_rate']),
                              loss='binary_crossentropy')
                return model
            elif model_type == 'GRU':
                inp = Input(shape=input_shape)
                x = GRU(MODEL_CONFIG['GRU']['units'])(inp)
                x = Dropout(MODEL_CONFIG['GRU']['dropout'])(x)
                out = Dense(1, activation='sigmoid')(x)
                model = Model(inputs=inp, outputs=out)
                model.compile(optimizer=Adam(learning_rate=MODEL_CONFIG['GRU']['learning_rate']),
                              loss='binary_crossentropy')
                return model
            elif model_type == 'CNN':
                inp = Input(shape=input_shape)
                x = Conv1D(filters=MODEL_CONFIG['CNN']['filters'],
                           kernel_size=MODEL_CONFIG['CNN']['kernel_size'],
                           activation='relu')(inp)
                x = GlobalMaxPooling1D()(x)
                out = Dense(1, activation='sigmoid')(x)
                model = Model(inputs=inp, outputs=out)
                model.compile(optimizer=Adam(learning_rate=MODEL_CONFIG['CNN']['learning_rate']),
                              loss='binary_crossentropy')
                return model
            elif model_type == 'Transformer':
                inp = Input(shape=input_shape)
                x = MultiHeadAttention(num_heads=MODEL_CONFIG['Transformer']['heads'],
                                       key_dim=MODEL_CONFIG['Transformer']['key_dim'])(inp, inp)
                x = LayerNormalization(epsilon=1e-6)(x)
                x = GlobalMaxPooling1D()(x)
                x = Dense(64, activation='relu')(x)
                out = Dense(1, activation='sigmoid')(x)
                model = Model(inputs=inp, outputs=out)
                model.compile(optimizer=Adam(learning_rate=MODEL_CONFIG['Transformer']['learning_rate']),
                              loss='binary_crossentropy')
                return model
            elif model_type == 'XGBoost':
                model = XGBRegressor(**MODEL_CONFIG['XGBoost'])
                return model
            elif model_type == 'LightGBM':
                return None  # Will be handled in the training branch.
            elif model_type == 'CatBoost':
                model = CatBoostRegressor(**MODEL_CONFIG['CatBoost'])
                return model
            elif model_type == 'Stacking':
                estimators = [
                    ('xgb', XGBRegressor(n_estimators=100)),
                    ('lgbm', XGBRegressor(n_estimators=100))
                ]
                model = StackingRegressor(estimators=estimators, final_estimator=CatBoostRegressor(iterations=100))
                return model
            elif model_type == 'GaussianProcess':
                model = GaussianProcessRegressor()
                return model
            elif model_type == 'Prophet':
                return Prophet()
            elif model_type == 'Sentiment':
                inp = Input(shape=input_shape)
                x = Dense(64, activation='relu')(inp)
                x = Dropout(0.3)(x)
                out = Dense(1, activation='sigmoid')(x)
                model = Model(inputs=inp, outputs=out)
                model.compile(optimizer=Adam(learning_rate=MODEL_CONFIG['Sentiment']['learning_rate']),
                              loss='binary_crossentropy')
                return model
            elif model_type == 'Anomaly':
                inp = Input(shape=input_shape)
                encoded = Dense(32, activation='relu')(inp)
                decoded = Dense(input_shape[0], activation='linear')(encoded)
                model = Model(inputs=inp, outputs=decoded)
                model.compile(optimizer=Adam(learning_rate=MODEL_CONFIG['Anomaly']['learning_rate']),
                              loss='mse')
                return model
            elif model_type == 'SVR':
                model = SVR(C=MODEL_CONFIG['SVR']['C'], epsilon=MODEL_CONFIG['SVR']['epsilon'])
                return model
            elif model_type in ['DQN', 'PPO']:
                return None
            else:
                logging.error(f"Model type '{model_type}' not recognized.")
                return None
        except Exception as e:
            logging.error(f"Model creation failed for {model_type}: {str(e)}")
            return None

##############################
# Backtest Engine
##############################
class BacktestEngine:
    def __init__(self):
        self.results = []

    def run_backtest(self, model, model_type, X, y, prices, dates=None, extra_data=None):
        try:
            # Generate predictions and choose price slice accordingly:
            if model_type in ['LSTM', 'GRU', 'CNN', 'Transformer']:
                preds = model.predict(X).flatten()
                prices_slice = prices.iloc[WINDOW_SIZE-1:WINDOW_SIZE-1+len(preds)]
            elif model_type == 'Sentiment':
                preds = model.predict(X).flatten()
                prices_slice = prices
            elif model_type == 'Prophet':
                prophet_df = pd.DataFrame({'ds': dates, 'y': prices.values})
                model.fit(prophet_df)
                forecast = model.predict(prophet_df)
                preds = (forecast['yhat'].values > prices.values).astype(float)
                prices_slice = prices
            elif model_type == 'Anomaly':
                recon = model.predict(X)
                error = np.mean(np.abs(X - recon), axis=1)
                preds = (error < np.percentile(error, 75)).astype(float)
                prices_slice = prices
            elif model_type in ['XGBoost', 'CatBoost', 'Stacking', 'GaussianProcess', 'SVR']:
                preds_cont = model.predict(X)
                preds = (preds_cont > 0).astype(float)
                prices_slice = prices
            elif model_type == 'LightGBM':
                dtrain = lgb.Dataset(X, label=y)
                booster = lgb.train(params={'learning_rate': MODEL_CONFIG['LightGBM']['learning_rate'],
                                            'objective': 'regression'},
                                    train_set=dtrain,
                                    num_boost_round=MODEL_CONFIG['LightGBM']['num_boost_round'])
                preds_cont = booster.predict(X)
                preds = (preds_cont > 0).astype(float)
                prices_slice = prices
                model = booster  # Use booster as the model.
            elif model_type in ['DQN', 'PPO']:
                env = extra_data['env']
                obs = env.reset()
                actions = []
                for _ in range(len(prices)-1):
                    action, _ = model.predict(obs)
                    actions.append(action)
                    obs, _, done, _ = env.step(action)
                    if done:
                        break
                preds = np.array(actions)
                prices_slice = prices
            else:
                logging.error(f"Unknown model type during backtesting: {model_type}")
                return None, None

            # Use vectorbt to generate a portfolio from signals:
            portfolio = vbt.Portfolio.from_signals(
                prices_slice,
                entries=preds > 0.6,
                exits=preds < 0.4,
                fees=0.001,
                slippage=0.005,
                freq='1D'
            )
            stats = portfolio.stats()
            stats_dict = stats.to_dict()
            stats_dict["Equity Curve"] = portfolio.total_return()
            return stats_dict, model
        except Exception as e:
            logging.error(f"Backtest failed for {model_type}: {str(e)}")
            return None, None

    def save_results(self, filename="backtest_summary.csv"):
        if self.results:
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(filename, index=False)
            logging.info(f"Saved backtest summary to {filename}")
            return df_results
        else:
            logging.warning("No backtest results to save.")
            return None

##############################
# Main Process – Re-Backtesting Pipeline
##############################
def main():
    data_master = DataMaster()
    backtester = BacktestEngine()
    model_factory = ModelFactory()
    
    all_assets = FOREX_PAIRS + CRYPTO_PAIRS
    for asset in all_assets:
        logging.info(f"Processing {asset}")
        try:
            df = data_master.fetch_data(asset)
            X_raw, y, prices = data_master.preprocess_data(df)
            if X_raw is None or y is None or prices is None:
                logging.warning(f"Skipping {asset} due to insufficient data.")
                continue
            X_seq, y_seq = create_sequences(X_raw, y)
            X_classical = X_raw.copy()
            dates = prices.index
            extra = {}
            
            for model_type in MODEL_LIST:
                logging.info(f"Training {model_type} on {asset}")
                model = None
                stats = None
                try:
                    # Build & train model based on type:
                    if model_type in ['LSTM', 'GRU', 'CNN', 'Transformer']:
                        input_shape = (WINDOW_SIZE, X_raw.shape[1])
                        model = model_factory.create_model(model_type, input_shape)
                        if model is None:
                            continue
                        model.fit(X_seq, y_seq,
                                  epochs=MODEL_CONFIG[model_type]['epochs'],
                                  batch_size=MODEL_CONFIG[model_type]['batch_size'],
                                  verbose=0)
                        stats, model = backtester.run_backtest(model, model_type, X_seq, y_seq, prices, dates)
                    elif model_type == 'Sentiment':
                        input_shape = (X_classical.shape[1],)
                        model = model_factory.create_model(model_type, input_shape)
                        if model is None:
                            continue
                        model.fit(X_classical, y,
                                  epochs=MODEL_CONFIG[model_type]['epochs'],
                                  batch_size=MODEL_CONFIG[model_type]['batch_size'],
                                  verbose=0)
                        stats, model = backtester.run_backtest(model, model_type, X_classical, y, prices, dates)
                    elif model_type == 'Anomaly':
                        input_shape = (X_classical.shape[1],)
                        model = model_factory.create_model(model_type, input_shape)
                        if model is None:
                            continue
                        model.fit(X_classical, X_classical,
                                  epochs=MODEL_CONFIG[model_type]['epochs'],
                                  batch_size=MODEL_CONFIG[model_type]['batch_size'],
                                  verbose=0)
                        stats, model = backtester.run_backtest(model, model_type, X_classical, y, prices, dates)
                    elif model_type in ['XGBoost', 'CatBoost', 'Stacking', 'GaussianProcess', 'SVR']:
                        model = model_factory.create_model(model_type)
                        if model is None:
                            continue
                        model.fit(X_classical, y)
                        stats, model = backtester.run_backtest(model, model_type, X_classical, y, prices, dates)
                    elif model_type == 'LightGBM':
                        stats, model = backtester.run_backtest(None, model_type, X_classical, y, prices, dates)
                    elif model_type == 'Prophet':
                        model = model_factory.create_model(model_type)
                        stats, model = backtester.run_backtest(model, model_type, X_classical, y, prices, dates)
                    elif model_type in ['DQN', 'PPO']:
                        env = TradingEnv(prices, X_classical)
                        extra['env'] = env
                        if model_type == 'DQN':
                            dqn_config = MODEL_CONFIG['DQN'].copy()
                            total_timesteps = dqn_config.pop('total_timesteps', 10000)
                            agent = DQN("MlpPolicy", env, verbose=0, **dqn_config)
                        else:
                            ppo_config = MODEL_CONFIG['PPO'].copy()
                            total_timesteps = ppo_config.pop('total_timesteps', 10000)
                            agent = PPO("MlpPolicy", env, verbose=0, **ppo_config)
                        agent.learn(total_timesteps=total_timesteps)
                        model = agent
                        stats, model = backtester.run_backtest(model, model_type, X_classical, y, prices, dates, extra)
                    else:
                        logging.error(f"Model type {model_type} not explicitly handled.")
                        continue

                    if stats:
                        # Append asset/model result to summary results.
                        result_row = {'Asset': asset, 'Model': model_type}
                        result_row.update(stats)
                        backtester.results.append(result_row)
                        # Save the model:
                        save_conf = SAVE_CONFIG[model_type]
                        filename = f"models/{asset.replace('/', '_')}_{model_type.lower()}_model{save_conf['ext']}"
                        if save_conf['method'] == 'tf':
                            save_model(model, filename, overwrite=True)
                        elif save_conf['method'] == 'xgb':
                            model.save_model(filename)
                        elif save_conf['method'] == 'lgb':
                            model.save_model(filename)
                        elif save_conf['method'] == 'catboost':
                            model.save_model(filename)
                        elif save_conf['method'] == 'joblib':
                            joblib.dump(model, filename)
                        elif save_conf['method'] == 'stable_baselines3':
                            model.save(filename)
                        logging.info(f"Saved {model_type} model for {asset} as {filename}")
                    else:
                        logging.warning(f"No stats generated for {asset} with model {model_type}.")
                except Exception as inner_e:
                    logging.error(f"Model {model_type} failed on {asset}: {str(inner_e)}")
        except Exception as e:
            logging.error(f"Asset {asset} processing failed: {str(e)}")
            continue

    # Save auxiliary components.
    joblib.dump(data_master.scaler, 'models/price_scaler.pkl')
    joblib.dump({"tokenizer": "dummy"}, 'models/sentiment_tokenizer.pkl')

    # Save and print consolidated backtest results.
    summary_df = backtester.save_results()
    if summary_df is not None:
        print("\nBacktest Summary:")
        print(tabulate(summary_df, headers="keys", tablefmt="psql", showindex=False))
    else:
        print("No backtest results available.")

if __name__ == "__main__":
    main()
