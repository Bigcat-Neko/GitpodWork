#!/usr/bin/env python
"""
backtest_models.py

This script performs extensive backtesting over all major Forex and crypto assets.
It loads models from the "models/" folder, computes technical indicators and prepares features
(using integrated functions), runs backtests on each asset for all models, saves backtest results 
to a JSON file, sends a summary via Telegram, and then copies the original models into the "b_models/" folder.
"""

import os
import json
import glob
import time
import shutil
import joblib
import numpy as np
import pandas as pd
import datetime
import logging
import requests
import aiohttp
import asyncio
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
import nest_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtester")

# -------------------------------
# Configuration Settings
# -------------------------------
HISTORICAL_DATA_PATH = "historical_data.csv"  # Optional CSV cache for historical data
MODEL_LOAD_DIR = "models"                     # Load models from here (original production models)
MODEL_SAVE_DIR = "b_models"                   # Backtested models will be copied here
BACKTEST_RESULTS_FILE = "backtest_results.json"

# Set the backtest time range (you can adjust these as needed)
BACKTEST_START_DATE = "1900-01-01"
BACKTEST_END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# Define your asset universe (as in main.py)
forex_assets = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]
crypto_assets = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD"]
asset_universe = forex_assets + crypto_assets

# Ensure the folder for saving backtested models exists
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# -------------------------------
# Data Loading Functions
# -------------------------------
def fetch_twelvedata_data(symbol: str) -> pd.DataFrame:
    """
    Fetch historical daily OHLC data for a symbol using TwelveData.
    A short delay is added to help avoid rate limiting.
    """
    if not TWELVEDATA_API_KEY:
        logger.error("TWELVEDATA_API_KEY not set in environment.")
        return pd.DataFrame()
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON"
    }
    try:
        response = requests.get(url, params=params)
        time.sleep(1)  # delay
        data = response.json()
        if "values" in data:
            df = pd.DataFrame(data["values"])
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", inplace=True)
            df.set_index("timestamp", inplace=True)
            logger.info(f"TwelveData returned {len(df)} rows for {symbol}.")
            return df
        else:
            logger.warning(f"TwelveData did not return 'values' for {symbol}. Data: {data}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching TwelveData data for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_yfinance_data(symbol: str) -> pd.DataFrame:
    """
    Fetch historical daily OHLC data for a symbol using yfinance.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, period="max", interval="1d")
        if df.empty:
            logger.warning(f"yfinance returned empty data for {symbol}.")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        logger.info(f"yfinance returned {len(df)} rows for {symbol}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching yfinance data for {symbol}: {str(e)}")
        return pd.DataFrame()

def load_full_historical_data(symbol: str) -> pd.DataFrame:
    """
    Retrieve the fullest available historical data.
    First attempt with TwelveData; if data is sparse, fallback to yfinance.
    """
    df = fetch_twelvedata_data(symbol)
    if df.empty or len(df) < 100:
        logger.info(f"Falling back to yfinance for {symbol}.")
        df = fetch_yfinance_data(symbol)
    return df

def load_historical_data(csv_path: str) -> pd.DataFrame:
    """
    Load historical data from a CSV file.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        logger.info(f"Loaded {len(df)} rows from {csv_path}.")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        return pd.DataFrame()

# -------------------------------
# Technical Indicator and Feature Functions
# -------------------------------
def compute_professional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators.
    Indicators include: SMA(50), RSI(14), ADX(14) and its directional indices, and Bollinger Bands (middle).
    Defaults are used if calculations fail.
    """
    try:
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        if df.empty:
            default_price = 1.0
            df = pd.DataFrame({
                'open': [default_price],
                'high': [default_price],
                'low': [default_price],
                'close': [default_price]
            })
        closes = df['close']
        sma_50 = closes.rolling(window=50, min_periods=1).mean()
        if sma_50.isna().all():
            sma_50 = closes.rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = sma_50.ffill().bfill()
        try:
            import pandas_ta as ta
            rsi = ta.rsi(closes, length=14)
            if (rsi is None) or (isinstance(rsi, pd.Series) and rsi.empty):
                df['RSI_14'] = 50
            else:
                df['RSI_14'] = rsi.fillna(50)
        except Exception as e:
            logger.error(f"RSI error: {str(e)}")
            df['RSI_14'] = 50
        try:
            import pandas_ta as ta
            adx_data = ta.adx(df['high'], df['low'], closes, length=14)
            if (adx_data is None) or (isinstance(adx_data, pd.DataFrame) and adx_data.empty):
                df['ADX_14'] = 25
                df['+DI_14'] = 25
                df['-DI_14'] = 25
            else:
                df['ADX_14'] = adx_data.get('ADX_14', pd.Series(25, index=df.index)).fillna(25)
                df['+DI_14'] = adx_data.get('DMP_14', pd.Series(25, index=df.index)).fillna(25)
                df['-DI_14'] = adx_data.get('DMN_14', pd.Series(25, index=df.index)).fillna(25)
        except Exception as e:
            logger.error(f"ADX error: {str(e)}")
            df['ADX_14'] = 25
            df['+DI_14'] = 25
            df['-DI_14'] = 25
        try:
            import pandas_ta as ta
            bb = ta.bbands(closes, length=20)
            if (bb is None) or (isinstance(bb, pd.DataFrame) and bb.empty):
                df['BB_MIDDLE'] = closes.rolling(window=20, min_periods=1).mean()
            else:
                middle = bb.iloc[:, 1]
                if middle.isna().all():
                    middle = closes.rolling(window=20, min_periods=1).mean()
                df['BB_MIDDLE'] = middle.ffill().bfill()
        except Exception as e:
            logger.error(f"Bollinger Bands error: {str(e)}")
            df['BB_MIDDLE'] = closes.rolling(window=20, min_periods=1).mean()
        for col, default in [('RSI_14', 50), ('ADX_14', 25), ('+DI_14', 25), ('-DI_14', 25)]:
            if df[col].isna().all():
                df[col] = default
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}")
        default_price = 1.0
        default_data = {
            'open': default_price,
            'high': default_price,
            'low': default_price,
            'close': default_price,
            'SMA_50': default_price,
            'RSI_14': 50,
            'ADX_14': 25,
            '+DI_14': 25,
            '-DI_14': 25,
            'BB_MIDDLE': default_price
        }
        return pd.DataFrame([default_data])

def prepare_features(df: pd.DataFrame, model_type: str):
    """
    Prepare features for backtesting.
    Uses a fixed set of features. For RL models, adds an extra feature and pads the array.
    """
    fixed_features = ['close', 'RSI_14', 'STOCH_%K', 'ADX_14', '+DI_14', '-DI_14',
                      'BB_MIDDLE', 'ATR_14', 'SMA_50', 'EMA_9']
    try:
        for col in fixed_features:
            if col not in df.columns:
                df[col] = df['close']
        features = df[fixed_features].dropna()
        if features.empty:
            return None
        if model_type == "rl":
            features = features.tail(1).copy()
            atr_val = features['ATR_14'].iloc[0]
            close_val = features['close'].iloc[0]
            features.loc[:, 'ATR_Ratio'] = atr_val / close_val if close_val != 0 else 0
            feature_array = features.values
            current_cols = feature_array.shape[1]
            if current_cols < 30:
                feature_array = np.pad(feature_array, ((0, 0), (0, 30 - current_cols)), 'constant')
            return feature_array
        elif model_type == "ml":
            return features.tail(1).values.reshape(1, -1)
        elif model_type == "dl":
            seq = df['close'].dropna().tail(10).values
            if len(seq) < 10:
                seq = np.pad(seq, (10 - len(seq), 0), 'edge')
            return seq.reshape(1, 10, 1)
        return None
    except Exception as e:
        logger.error(f"Feature preparation error: {str(e)}")
        return None

# -------------------------------
# Backtesting Functions
# -------------------------------
def simple_label_generation(df: pd.DataFrame, lookahead=1) -> pd.Series:
    """
    Generate labels for backtesting as the percentage change of 'close' over the next period.
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-lookahead)
    df["pct_change"] = (df["future_close"] - df["close"]) / df["close"]
    return df.dropna(subset=["pct_change"])["pct_change"]

def generate_backtest_features(df: pd.DataFrame, model_type: str):
    """
    Compute indicators and prepare features for backtesting.
    """
    df_ind = compute_professional_indicators(df.copy())
    features = prepare_features(df_ind, model_type)
    return features

def backtest_model(model_name: str, model, df: pd.DataFrame, model_type: str):
    """
    Backtest a single model over a rolling window.
    Returns performance metrics as a dictionary.
    """
    # Early check: if the model doesn't support prediction and it's not prophet_model, skip it.
    if model_name.lower() != "prophet_model" and (not hasattr(model, "predict") or not callable(model.predict)):
        logger.warning(f"Model '{model_name}' does not support prediction. Skipping backtest for this model.")
        return None

    logger.info(f"Backtesting model '{model_name}' ({model_type}).")
    predictions, actuals, timestamps = [], [], []
    label_series = simple_label_generation(df, lookahead=1)
    window_size = 50  # adjust as needed

    for current_end in label_series.index[window_size:]:
        window_df = df.loc[:current_end].tail(window_size)
        features = prepare_features(window_df, model_type)
        if features is None:
            continue

        try:
            if model_name.lower() == "prophet_model":
                # Prophet requires a DataFrame with a "ds" column.
                forecast_input = pd.DataFrame({"ds": window_df.index})
                pred = model.predict(forecast_input)
                # Use the last forecasted value (or choose another heuristic)
                prediction = float(pred["yhat"].iloc[-1])
            elif model_type in ["ml", "dl"]:
                pred = model.predict(features)
                prediction = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)
            elif model_type == "rl":
                prediction, _ = model.predict(features)
                prediction = float(prediction[0])
            else:
                prediction = 0.0
        except Exception as e:
            logger.error(f"Prediction error for '{model_name}' at {current_end}: {str(e)}")
            continue

        try:
            actual = label_series.loc[current_end]
        except KeyError:
            continue

        predictions.append(prediction)
        actuals.append(actual)
        timestamps.append(current_end)

    if not predictions:
        logger.error(f"No predictions generated for model '{model_name}'. Skipping.")
        return None

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    result = {
        "model_name": model_name,
        "num_predictions": len(predictions),
        "mse": mse,
        "mae": mae,
        "timestamp_start": str(timestamps[0]) if timestamps else "",
        "timestamp_end": str(timestamps[-1]) if timestamps else ""
    }
    logger.info(f"Finished backtesting model '{model_name}': MSE = {mse:.6f}, MAE = {mae:.6f}")
    return result

# -------------------------------
# Model Loading and Saving Functions
# -------------------------------
def load_models():
    """
    Load models from the MODEL_LOAD_DIR folder.
    """
    models = {}
    if not os.path.exists(MODEL_LOAD_DIR):
        logger.warning(f"Model folder '{MODEL_LOAD_DIR}' does not exist.")
        return models
    for file_path in glob.glob(os.path.join(MODEL_LOAD_DIR, "*.pkl")):
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            models[model_name] = joblib.load(file_path)
            logger.info(f"Loaded model '{model_name}'.")
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {str(e)}")
    return models

def save_backtested_models(models):
    """
    Copy the original model files from the MODEL_LOAD_DIR to the MODEL_SAVE_DIR.
    """
    for model_name in models.keys():
        source_file = os.path.join(MODEL_LOAD_DIR, model_name + ".pkl")
        dest_file = os.path.join(MODEL_SAVE_DIR, model_name + ".pkl")
        try:
            shutil.copy(source_file, dest_file)
            logger.info(f"Copied model '{model_name}' to '{MODEL_SAVE_DIR}'.")
        except Exception as e:
            logger.error(f"Error copying model '{model_name}': {str(e)}")

# -------------------------------
# Backtesting Execution
# -------------------------------
def run_backtests():
    """
    Run backtesting on all assets in the asset universe.
    For each asset, load historical data, run backtests using each model loaded from the MODEL_LOAD_DIR,
    save the backtest results to a JSON file, send a Telegram summary, and copy models to MODEL_SAVE_DIR.
    """
    models_loaded = load_models()
    if not models_loaded:
        logger.error("No models loaded. Aborting backtest.")
        return

    overall_results = {}  # Structure: { asset: { model_name: result, ... }, ... }
    for asset in asset_universe:
        logger.info(f"Processing asset '{asset}'.")
        df_asset = load_full_historical_data(asset)
        if df_asset.empty:
            logger.error(f"No data for asset '{asset}'. Skipping.")
            continue
        # Optionally cache the raw data to CSV (overwrite or append asset name)
        asset_csv = f"{asset.replace('/', '_')}_historical_data.csv"
        df_asset.to_csv(asset_csv)
        df = load_historical_data(asset_csv)
        df = df.loc[BACKTEST_START_DATE:BACKTEST_END_DATE]
        if df.empty:
            logger.error(f"No data for asset '{asset}' in the specified range. Skipping.")
            continue
        asset_results = {}
        for model_name, model in models_loaded.items():
            # Determine model type based on naming conventions
            if any(prefix in model_name.lower() for prefix in ["xgb", "lightgbm", "catboost", "svr", "stacking", "gaussian"]):
                model_type = "ml"
            elif any(prefix in model_name.lower() for prefix in ["lstm", "gru", "transformer", "cnn"]):
                model_type = "dl"
            elif any(prefix in model_name.lower() for prefix in ["dqn", "ppo"]):
                model_type = "rl"
            else:
                logger.warning(f"Model type for '{model_name}' not identified; defaulting to 'ml'.")
                model_type = "ml"
            result = backtest_model(model_name, model, df, model_type)
            if result:
                asset_results[model_name] = result
        if asset_results:
            overall_results[asset] = asset_results

    # Save overall backtesting results to JSON file
    with open(BACKTEST_RESULTS_FILE, "w") as outfile:
        json.dump(overall_results, outfile, indent=4)
    logger.info(f"Saved backtest results to '{BACKTEST_RESULTS_FILE}'.")

    # Prepare summary message for Telegram
    summary_lines = ["<b>Backtest Summary:</b>"]
    for asset, results in overall_results.items():
        summary_lines.append(f"\nAsset: {asset}")
        for model_name, res in results.items():
            summary_lines.append(f"  {model_name}: {res['num_predictions']} predictions, MSE: {res['mse']:.6f}, MAE: {res['mae']:.6f}")
    summary_message = "\n".join(summary_lines)
    asyncio.run(send_telegram_message(summary_message))

    # Copy the backtested models to MODEL_SAVE_DIR
    save_backtested_models(models_loaded)

async def send_telegram_message(message: str):
    """
    Send a message via Telegram.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        logger.error("Telegram credentials not set properly.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHANNEL_ID, "text": message, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Telegram message sent.")
                else:
                    logger.error(f"Telegram message failed: {await response.text()}")
    except Exception as e:
        logger.error(f"Telegram error: {str(e)}")

if __name__ == "__main__":
    run_backtests()
