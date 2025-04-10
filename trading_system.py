#!/usr/bin/env python
"""
backtest_models.py

This script performs an extensive backtest of all your models using the full historical
dataset. It attempts to pull data from both TwelveData (with rate-limit safe delays) and 
yfinance (as a fallback). It loads models using your main.py loader (Option A) to ensure
consistency with production. Finally, it sends a summary via Telegram.
"""

import os
import json
import glob
import time
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
nest_asyncio.apply()
from tenacity import retry, stop_after_attempt, wait_exponential

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
HISTORICAL_DATA_PATH = "historical_data.csv"  # Enriched CSV (if you use one), or you can call load_full_historical_data below
MODEL_DIR = "models"  # Models are stored here
BACKTEST_RESULTS_FILE = "backtest_results.json"

# Set these dates to capture as much historical data as possible.
BACKTEST_START_DATE = "1900-01-01"
BACKTEST_END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# -------------------------------
# Data Loading Functions
# -------------------------------
def fetch_twelvedata_data(symbol: str) -> pd.DataFrame:
    """
    Fetch historical daily OHLC data for a given symbol using TwelveData.
    Applies a 1-second delay to avoid rate limiting (free API).
    """
    if not TWELVEDATA_API_KEY:
        logger.error("TWELVEDATA_API_KEY not set in environment.")
        return pd.DataFrame()

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,  # maximum allowed output size for free tier
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON"
    }
    try:
        response = requests.get(url, params=params)
        time.sleep(1)  # Delay to mitigate rate limiting for free API
        data = response.json()
        if "values" in data:
            df = pd.DataFrame(data["values"])
            # Rename and convert datetime
            df.rename(columns={"datetime": "timestamp", "open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", inplace=True)
            df.set_index("timestamp", inplace=True)
            logger.info(f"TwelveData returned {len(df)} rows for {symbol}.")
            return df
        else:
            logger.warning(f"TwelveData did not return 'values' for {symbol}. Data: {data}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data from TwelveData for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_yfinance_data(symbol: str) -> pd.DataFrame:
    """
    Fetch the maximum available historical daily OHLC data for a given symbol using yfinance.
    """
    import yfinance as yf
    try:
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
        logger.error(f"Error fetching data from yfinance for {symbol}: {str(e)}")
        return pd.DataFrame()

def load_full_historical_data(symbol: str) -> pd.DataFrame:
    """
    Load the fullest historical data by trying TwelveData first (with delay for rate limiting)
    and then falling back to yfinance if necessary.
    """
    df = fetch_twelvedata_data(symbol)
    if df.empty or len(df) < 100:  # if very little data is returned, fallback
        logger.info(f"Falling back to yfinance for {symbol}.")
        df = fetch_yfinance_data(symbol)
    return df

def load_historical_data(csv_path: str) -> pd.DataFrame:
    """
    If you prefer using an enriched CSV, this function loads it.
    Otherwise, use load_full_historical_data() to directly retrieve data.
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
# Model Loading (Option A)
# -------------------------------
def load_models():
    """
    Load models using the method defined in main.py.
    This ensures the same models used in production are loaded.
    """
    try:
        from main import load_all_models, app
        load_all_models()
        models = app.models
        if models:
            logger.info("Successfully loaded models via main.py's load_all_models().")
            return models
    except Exception as e:
        logger.error(f"Error loading models via main.py: {str(e)}")
    # Fallback method: if needed, load from MODEL_DIR using joblib.
    models = {}
    for file_path in glob.glob(os.path.join(MODEL_DIR, "*.pkl")):
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            models[model_name] = joblib.load(file_path)
            logger.info(f"Fallback: Loaded model {model_name} from file.")
        except Exception as e:
            logger.error(f"Fallback: Error loading model {model_name}: {str(e)}")
    return models

# -------------------------------
# Backtesting Functions
# -------------------------------
# Import key functions from main.py for indicators and feature preparation.
try:
    from main import compute_professional_indicators, prepare_features
except ImportError:
    raise ImportError("Ensure compute_professional_indicators and prepare_features are accessible from main.py.")

def generate_backtest_features(df: pd.DataFrame, model_type: str):
    """
    Compute technical indicators and prepare features for backtesting.
    """
    df_ind = compute_professional_indicators(df.copy())
    features = prepare_features(df_ind, model_type)
    return features

def simple_label_generation(df: pd.DataFrame, lookahead=1) -> pd.Series:
    """
    Generate a simple label: percentage change of 'close' over the next period.
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-lookahead)
    df["pct_change"] = (df["future_close"] - df["close"]) / df["close"]
    return df.dropna(subset=["pct_change"])["pct_change"]

def backtest_model(model_name: str, model, df: pd.DataFrame, model_type: str):
    """
    Sequentially run a backtest for a single model over a rolling window.
    Returns a dictionary with error metrics.
    """
    logger.info(f"Starting backtest for {model_name} ({model_type}).")
    predictions, actuals, timestamps = [], [], []
    label_series = simple_label_generation(df, lookahead=1)
    
    window_size = 50  # Adjust the window size as needed for your indicator lookback.
    for current_end in label_series.index[window_size:]:
        window_df = df.loc[:current_end].tail(window_size)
        features = generate_backtest_features(window_df, model_type)
        if features is None:
            continue
        
        try:
            if model_type in ["ml", "dl"]:
                pred = model.predict(features)
                prediction = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)
            elif model_type == "rl":
                prediction, _ = model.predict(features)
                prediction = float(prediction[0])
            else:
                prediction = 0.0
        except Exception as e:
            logger.error(f"Prediction error for {model_name} at {current_end}: {str(e)}")
            continue

        try:
            actual = label_series.loc[current_end]
        except KeyError:
            continue
        
        predictions.append(prediction)
        actuals.append(actual)
        timestamps.append(current_end)

    if not predictions:
        logger.error(f"No predictions for {model_name}; skipping.")
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
    logger.info(f"Backtest for {model_name} complete: MSE = {mse:.6f}, MAE = {mae:.6f}")
    return result

async def send_telegram_message(message: str):
    """
    Send a summary message via Telegram using your bot token and channel ID.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        logger.error("Telegram bot credentials not properly set.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Telegram message sent.")
                else:
                    logger.error(f"Telegram message failed: {await response.text()}")
    except Exception as e:
        logger.error(f"Telegram error: {str(e)}")

def run_backtests():
    """
    Execute backtesting over the full available historical data.
    Loads models using the main.py function and runs backtests sequentially.
    Saves results and sends a Telegram summary.
    """
    # For a given asset symbol, try fetching data using both data sources.
    # Change the symbol as necessary. For multi-asset systems, consider looping.
    symbol = "AAPL"  # Example symbol; replace with your asset or handle multiple.
    logger.info(f"Fetching full historical data for {symbol}.")
    df_td = load_full_historical_data(symbol)
    if df_td.empty:
        logger.error("No data available from TwelveData or yfinance. Aborting.")
        return
    
    # Optionally, save the enriched data to CSV for future use.
    df_td.to_csv(HISTORICAL_DATA_PATH)
    df = load_historical_data(HISTORICAL_DATA_PATH)
    df = df.loc[BACKTEST_START_DATE:BACKTEST_END_DATE]
    if df.empty:
        logger.error("No historical data in the specified range. Aborting backtest.")
        return

    models = load_models()
    if not models:
        logger.error("No models loaded. Aborting backtest.")
        return

    backtest_results = {}
    for model_name, model in models.items():
        # Determine model type based on name keywords.
        if any(prefix in model_name.lower() for prefix in ["xgb", "lightgbm", "catboost", "svr", "stacking", "gaussian"]):
            model_type = "ml"
        elif any(prefix in model_name.lower() for prefix in ["lstm", "gru", "transformer", "cnn"]):
            model_type = "dl"
        elif any(prefix in model_name.lower() for prefix in ["dqn", "ppo"]):
            model_type = "rl"
        else:
            logger.warning(f"Model type for {model_name} not identified; defaulting to 'ml'.")
            model_type = "ml"

        result = backtest_model(model_name, model, df, model_type)
        if result:
            backtest_results[model_name] = result

    # Save backtest results for future reference.
    with open(BACKTEST_RESULTS_FILE, "w") as outfile:
        json.dump(backtest_results, outfile, indent=4)
    logger.info(f"Saved backtest results to {BACKTEST_RESULTS_FILE}")

    # Compose summary message for Telegram.
    summary_lines = ["<b>Backtest Summary:</b>"]
    for model_name, res in backtest_results.items():
        summary_lines.append(f"{model_name}: {res['num_predictions']} predictions, MSE: {res['mse']:.6f}, MAE: {res['mae']:.6f}")
    summary_message = "\n".join(summary_lines)

    # Send Telegram summary.
    asyncio.run(send_telegram_message(summary_message))

if __name__ == "__main__":
    run_backtests()
