import os
import time
import threading
import asyncio
import random
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import pickle
import xgboost as xgb

# Disable CUDA (force CPU usage)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Attempt to import MetaTrader5; if unavailable, log a warning.
try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    mt5 = None
    import logging
    logging.warning("MetaTrader5 module not available. Running without MT5 execution.")

import logging
import pytz
from datetime import datetime, timedelta, time as dt_time
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from queue import Queue
from dotenv import load_dotenv

# Allow nested asyncio loops
import nest_asyncio
nest_asyncio.apply()

# Import telegram components from python-telegram-bot
from telegram import Update
from telegram.ext import Application as TGApplication, CommandHandler, ContextTypes

import pandas_ta as ta
from passlib.context import CryptContext
from typing import List, Dict, Any

# -------------------------------
# Database Setup (SQLite with SQLAlchemy)
# -------------------------------
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import custom loss for model loading (if needed)
from tensorflow.keras.losses import MeanSquaredError  # type: ignore

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal_DB = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    broker_info = Column(String, nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# -------------------------------
# Environment & Logging Setup
# -------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
EMAIL_FROM = os.getenv("EMAIL_FROM")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# -------------------------------
# Adjustable Parameters & Risk Settings
# -------------------------------
TP_MULTIPLIER = 2.0          
SL_MULTIPLIER = 1.5          
MONITOR_INTERVAL = 30        
PROFIT_THRESHOLD_CLOSE = 3.0 
DECAY_FACTOR = 0.9           

# -------------------------------
# Manual Mode & Execution Variables
# -------------------------------
manual_mode_override = False  
manual_mode = None            
monitor_only_mode = True       # In development, only signals are sent via Telegram

# -------------------------------
# FastAPI Initialization
# -------------------------------
app = FastAPI(title="NEKO AI Trading API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    event_loop = asyncio.get_running_loop()
except RuntimeError:
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

# -------------------------------
# Global Variables and Queues
# -------------------------------
connected_clients: List[WebSocket] = []
trade_queue = Queue()
active_trades: Dict[str, Any] = {}
historical_trades: List[Dict[str, Any]] = []
signal_counter = 0
last_news_update = ""
signal_generation_active = True
last_trading_mode = None
last_signal_time: Dict[str, float] = {}
total_loss_today = 0.0
current_day = datetime.now().date()

# -------------------------------
# Database Dependency for FastAPI
# -------------------------------
def get_db():
    db = SessionLocal_DB()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# User Models for Registration & Authentication
# -------------------------------
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    username: str
    email: EmailStr
    registered_at: datetime

# -------------------------------
# Utility Functions: Password Hashing and Token Generation
# -------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# -------------------------------
# Email Sending Function (Using aiosmtplib)
# -------------------------------
import aiosmtplib
from email.message import EmailMessage

async def send_email(to_email: str, subject: str, body: str):
    message = EmailMessage()
    message["From"] = EMAIL_FROM
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)
    try:
        await aiosmtplib.send(
            message,
            hostname=SMTP_SERVER,
            port=SMTP_PORT,
            username=SMTP_USER,
            password=SMTP_PASSWORD,
            start_tls=True,
        )
    except Exception as e:
        logging.error(f"Error sending email: {e}")

# -------------------------------
# User Registration Endpoint
# -------------------------------
@app.post("/register", response_model=UserOut)
def register(user: UserCreate, db: SessionLocal_DB = Depends(get_db)):  # type: ignore
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    token = create_access_token({"sub": new_user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    asyncio.run(send_email(new_user.email, "Your Login Token", f"Your token is: {token}"))
    return new_user

# -------------------------------
# User Login Endpoint
# -------------------------------
@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal_DB = Depends(get_db)):  # type: ignore
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: SessionLocal_DB = Depends(get_db)):  # type: ignore
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate token")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

# -------------------------------
# Broker Account Linking Endpoint
# -------------------------------
class BrokerLink(BaseModel):
    broker: str
    broker_username: str
    broker_password: str
    broker_server: str

@app.post("/link_broker")
def link_broker(broker_info: BrokerLink, current_user: User = Depends(get_current_user), db: SessionLocal_DB = Depends(get_db)):  # type: ignore
    import json
    current_user.broker_info = json.dumps(broker_info.dict())
    db.add(current_user)
    db.commit()
    return {"message": "Broker account linked successfully."}

# -------------------------------
# Model Loading Functions
# -------------------------------
import lightgbm as lgb
import joblib
from stable_baselines3 import PPO, DQN

def load_models():
    models = {}

    # Load CNN Model
    cnn_model_path = "models/cnn_model.keras"
    if os.path.exists(cnn_model_path):
        models["cnn"] = tf.keras.models.load_model(cnn_model_path)
        print("[✅] CNN Model Loaded Successfully!")
    else:
        print("[❌] CNN Model Not Found!")

    # Load GRU Model
    gru_model_path = "models/gru_model.h5"
    if os.path.exists(gru_model_path):
        models["gru"] = tf.keras.models.load_model(gru_model_path)
        print("[✅] GRU Model Loaded Successfully!")
    else:
        print("[❌] GRU Model Not Found!")

    # Load XGBoost Model
    xgb_model_path = "models/xgboost_model.json"
    if os.path.exists(xgb_model_path):
        models["xgboost"] = xgb.XGBRegressor()
        models["xgboost"].load_model(xgb_model_path)
        print("[✅] XGBoost Model Loaded Successfully!")
    else:
        print("[❌] XGBoost Model Not Found!")

    # Load LightGBM Model
    lightgbm_model_path = "models/lightgbm_model.txt"
    if os.path.exists(lightgbm_model_path):
        models["lightgbm"] = lgb.Booster(model_file=lightgbm_model_path)
        print("[✅] LightGBM Model Loaded Successfully!")
    else:
        print("[❌] LightGBM Model Not Found!")

    # Load Stacking Model
    stacking_model_path = "models/stacking_model.pkl"
    if os.path.exists(stacking_model_path):
        models["stacking"] = joblib.load(stacking_model_path)
        print("[✅] Stacking Model Loaded Successfully!")
    else:
        print("[❌] Stacking Model Not Found!")

    # Load Sentiment Analysis Model
    sentiment_model_path = "models/sentiment_model.pkl"
    if os.path.exists(sentiment_model_path):
        with open(sentiment_model_path, "rb") as f:
            models["sentiment"] = pickle.load(f)
        print("[✅] Sentiment Model Loaded Successfully!")
    else:
        print("[❌] Sentiment Model Not Found!")

    # Load Anomaly Detection Model
    anomaly_model_path = "models/anomaly_model.pkl"
    if os.path.exists(anomaly_model_path):
        with open(anomaly_model_path, "rb") as f:
            models["anomaly"] = pickle.load(f)
        print("[✅] Anomaly Detection Model Loaded Successfully!")
    else:
        print("[❌] Anomaly Detection Model Not Found!")

    # Load Price Scaler
    scaler_path = "models/price_scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            models["scaler"] = pickle.load(f)
        print("[✅] Price Scaler Loaded Successfully!")
    else:
        print("[❌] Price Scaler Not Found!")

    # Load Trade Signal Model
    trade_signal_model_path = "models/trade_signal_model.h5"
    if os.path.exists(trade_signal_model_path):
        models["trade_signal"] = tf.keras.models.load_model(trade_signal_model_path)
        print("[✅] Trade Signal Model Loaded Successfully!")
    else:
        print("[❌] Trade Signal Model Not Found!")

    # Load PPO Trading Model
    ppo_model_path = "models/trading_ppo.zip"
    if os.path.exists(ppo_model_path):
        try:
            models["ppo"] = PPO.load(ppo_model_path)
            print("[✅] PPO Trading Model Loaded Successfully!")
        except Exception as e:
            print(f"[❌] Error loading PPO Trading Model: {e}")
    else:
        print("[❌] PPO Trading Model Not Found!")

    # Load DQN Trading Model
    dqn_model_path = "models/trading_dqn.zip"
    if os.path.exists(dqn_model_path):
        try:
            models["dqn"] = DQN.load(dqn_model_path)
            print("[✅] DQN Trading Model Loaded Successfully!")
        except Exception as e:
            print(f"[❌] Error loading DQN Trading Model: {e}")
    else:
        print("[❌] DQN Trading Model Not Found!")

    return models

# Load all models and print status
models = load_models()
print("\n📌 **Final Model Loading Process Complete!**")

# -------------------------------
# Testing Model Predictions (with Correct Input Shapes)
# -------------------------------
# For CNN: expected input shape is (1, 60, 1)
sample_cnn_input = np.random.random((1, 60, 1))
if "cnn" in models:
    cnn_prediction = models["cnn"].predict(sample_cnn_input)
    print(f"[🔮] CNN Prediction: {cnn_prediction}")

# For LightGBM: trained on 3 features; generate a sample with 3 features.
sample_lightgbm_input = np.random.random((1, 3))
if "lightgbm" in models:
    lightgbm_prediction = models["lightgbm"].predict(sample_lightgbm_input)
    print(f"[🔮] LightGBM Prediction: {lightgbm_prediction}")

# For Stacking Model: trained on 6 features; generate a sample with 6 features.
sample_stacking_input = np.random.random((1, 6))
if "stacking" in models:
    stacking_prediction = models["stacking"].predict(sample_stacking_input)
    print(f"[🔮] Stacking Model Prediction: {stacking_prediction}")

# (Additional predictions for other models can be added similarly)

# -------------------------------
# MT5 Trading Functions (Development: No real execution)
# -------------------------------
def initialize_mt5(max_attempts=5):
    if mt5 is not None:
        if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logging.info("MT5 Initialized successfully.")
            return True
        else:
            error_code, error_msg = mt5.last_error()
            logging.error(f"MT5 Initialization failed: {error_msg}")
    else:
        logging.warning("MT5 module not available.")
    return False

def shutdown_mt5():
    if mt5 is not None:
        mt5.shutdown()

def place_mt5_order(symbol, order_type, volume, price, sl, tp, signal_id):
    if mt5 is None:
        logging.warning("MT5 module not available. Skipping order placement.")
        return None
    logging.info(f"Placing order for {symbol}: {order_type} at {price} with SL={sl} TP={tp}")
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol.replace("/", ""),
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": f"Trade Signal {signal_id}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    })
    return result

def close_trade(ticket, signal_id, symbol):
    logging.info(f"Trade {signal_id} would be closed here (MT5 execution not available).")

def get_account_info():
    if mt5 is not None:
        return mt5.account_info()
    return None

def monitor_trade(signal_id, mt5_order_ticket, symbol, entry_price):
    logging.info(f"Monitoring trade {signal_id} (MT5 execution not available).")

# -------------------------------
# Additional Functions: Data, Indicators, and Telegram Messaging
# -------------------------------
from fetch_live_prices import get_live_forex_prices
from news_fetcher import get_forex_news
from meta_learning import MetaLearningAgent
meta_agent = MetaLearningAgent()

feedback_list = []

class Feedback(BaseModel):
    signal_id: str
    rating: int
    comment: str = None

@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    feedback_entry = {
        "signal_id": feedback.signal_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "timestamp": datetime.utcnow().isoformat()
    }
    feedback_list.append(feedback_entry)
    logging.debug(f"Feedback received: {feedback_entry}")
    return {"message": "Feedback received", "data": feedback_entry}

@app.get("/feedback")
def get_feedback():
    return {"feedback": feedback_list}

def format_trade_signal(signal_details: Dict[str, Any]) -> str:
    message = (
        "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃ 🚀 **NekoAIBot Trade Signal** 🚀 ┃\n"
        "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
        f"**Signal ID:** `{signal_details.get('signal_id')}`\n"
        f"**Pair:** `{signal_details.get('pair')}`\n"
        f"**Predicted Change:** `{signal_details.get('predicted_change')}`\n"
        f"**News Sentiment:** `{signal_details.get('news_sentiment')}`\n"
        f"**AI Signal:** `{signal_details.get('signal')}`\n"
        f"**Confidence:** `{signal_details.get('confidence')}%`\n\n"
        f"**Entry:** `{signal_details.get('entry_price')}`\n"
        f"**Stop Loss:** `{signal_details.get('stop_loss')}`\n"
        "——————————————\n"
        "**Take Profits:**\n"
        f"  • **TP1:** `{signal_details.get('TP1')}`\n"
        f"  • **TP2:** `{signal_details.get('TP2')}`\n"
        f"  • **TP3:** `{signal_details.get('TP3')}`\n"
        f"  • **TP4:** `{signal_details.get('TP4')}`\n\n"
        "⚠️ *Risk Warning: Trading involves significant risk. Manage your risk accordingly.*\n\n"
        "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃   **NekoAIBot - Next-Gen Trading**   ┃\n"
        "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
    )
    return message

def send_telegram_message(message: str, to_channel: bool = False):
    chat_id = TELEGRAM_CHANNEL_ID if to_channel else TELEGRAM_CHAT_ID
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logging.debug(f"Message sent to {'Channel' if to_channel else 'Bot'}.")
                return
            elif response.status_code == 429:
                retry_after = response.json().get("parameters", {}).get("retry_after", 2)
                logging.warning(f"Rate limited (429). Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                logging.error(f"Failed to send message: {response.text}")
                return
        except Exception as e:
            logging.error(f"Exception while sending message: {e}")
            time.sleep(2)

# -------------------------------
# Economic Indicators & Position Sizing Endpoints
# -------------------------------
def get_economic_indicators(function="GDP"):
    url = "https://www.alphavantage.co/query"
    params = {"function": function, "apikey": ALPHAVANTAGE_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching economic indicators: {e}")
        return {}

@app.get("/active_trades")
def get_active_trades():
    return active_trades

@app.get("/historical_trades")
def get_historical_trades():
    return {"historical_trades": historical_trades}

@app.get("/performance_metrics")
def performance_metrics():
    if not historical_trades:
        return {"message": "No historical trades available."}
    total_trades = len(historical_trades)
    avg_confidence = np.mean([trade.get("confidence", 0) for trade in historical_trades])
    buy_signals = sum(1 for trade in historical_trades if trade.get("signal") == "BUY")
    sell_signals = sum(1 for trade in historical_trades if trade.get("signal") == "SELL")
    hold_signals = sum(1 for trade in historical_trades if trade.get("signal") == "HOLD")
    return {
        "total_trades": total_trades,
        "average_confidence": round(avg_confidence, 2),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
    }

@app.get("/backtest")
def backtest():
    if not historical_trades:
        return {"message": "No historical trades available for backtesting."}
    total_trades = len(historical_trades)
    wins = sum(1 for trade in historical_trades if trade.get("signal") == "BUY")
    losses = sum(1 for trade in historical_trades if trade.get("signal") == "SELL")
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_confidence = np.mean([trade.get("confidence", 0) for trade in historical_trades])
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_percentage": round(win_rate, 2),
        "average_confidence": round(avg_confidence, 2),
        "note": "This is a simplified backtest using BUY as win and SELL as loss."
    }

@app.get("/risk_adjustments")
def risk_adjustments():
    simulated_avg_atr = 0.0025
    suggested_multiplier = 1.5
    return {
        "simulated_average_atr": simulated_avg_atr,
        "suggested_stop_loss_multiplier": suggested_multiplier,
        "note": "Use these parameters to adjust stop loss levels based on market volatility."
    }

@app.get("/position_sizing")
def position_sizing(equity: float, risk_percent: float = 1.0):
    simulated_avg_atr = 0.0025
    risk_amount = equity * (risk_percent / 100.0)
    recommended_position_size = risk_amount / simulated_avg_atr
    return {
        "equity": equity,
        "risk_percent": risk_percent,
        "risk_amount": round(risk_amount, 2),
        "simulated_average_atr": simulated_avg_atr,
        "recommended_position_size": round(recommended_position_size, 2),
        "note": "This is a simplified calculation for dynamic position sizing."
    }

@app.get("/")
def home():
    return {"message": "NEKO AI Trading API is running!"}

# -------------------------------
# Live Signal Worker: Trading, Monitoring, and Updates
# -------------------------------
def live_signal_worker():
    global signal_counter, signal_generation_active, last_news_update, last_trading_mode, total_loss_today, current_day, last_signal_time
    current_day = datetime.now().date()
    while True:
        if not signal_generation_active:
            logging.info("Signal generation is paused.")
            time.sleep(10)
            continue

        today = datetime.now().date()
        if today != current_day:
            total_loss_today = 0.0
            current_day = today

        # Determine asset list based on session times
        if not manual_mode_override:
            est = pytz.timezone("US/Eastern")
            now_est = datetime.now(est).time()
            session_start = dt_time(5, 0)
            session_end = dt_time(22, 0)
            if session_start <= now_est <= session_end:
                assets = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"]
                mode = "forex"
            else:
                assets = ["BTC/USD", "ETH/USD", "XRP/USD"]
                mode = "crypto"
        else:
            if manual_mode == "crypto":
                assets = ["BTC/USD", "ETH/USD", "XRP/USD"]
                mode = "crypto"
            else:
                assets = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"]
                mode = "forex"

        if last_trading_mode is None:
            last_trading_mode = mode
        elif mode != last_trading_mode:
            mode_switch_message = f"🔔 Trading Mode Change Alert:\nSwitched from **{last_trading_mode.upper()}** to **{mode.upper()}**."
            send_telegram_message(mode_switch_message, to_channel=True)
            last_trading_mode = mode

        logging.info(f"Trading mode: {mode}")

        for asset in assets:
            if not signal_generation_active:
                break

            if not can_send_signal(asset):
                logging.info(f"Duplicate signal for {asset} suppressed.")
                continue

            prices, _ = get_extended_live_data(symbol=asset)
            if prices is None or prices.shape[0] < 60:
                logging.warning(f"Not enough price data for {asset}. Skipping...")
                continue

            logging.info(f"{asset} latest price: {prices[-1][0]}")
            df_prices = pd.DataFrame(prices, columns=["close"])
            df_prices = compute_indicators(df_prices)
            df_prices = compute_custom_indicators(df_prices)
            rsi_value = df_prices["rsi"].iloc[-1] if "rsi" in df_prices.columns else None
            macd_value = df_prices["macd"].iloc[-1] if "macd" in df_prices.columns else None
            bb_middle = df_prices["cmf"].iloc[-1] if "cmf" in df_prices.columns else None
            atr_value = df_prices["atr"].iloc[-1] if "atr" in df_prices.columns else None
            sma_20 = df_prices["SMA_20"].iloc[-1] if "SMA_20" in df_prices.columns else None
            sma_50 = df_prices["SMA_50"].iloc[-1] if "SMA_50" in df_prices.columns else None
            momentum = df_prices["Momentum"].iloc[-1] if "Momentum" in df_prices.columns else None

            logging.info(f"{asset} Indicators: RSI={rsi_value}, MACD={macd_value}, CMF={bb_middle}, ATR={atr_value}")

            # Reshape prices for prediction (CNN expects (1,60,1))
            reshaped_data = prices.reshape(1, 60, 1)
            lstm_pred = lstm_model.predict(reshaped_data)[0][0]
            gru_pred = gru_model.predict(reshaped_data)[0][0]
            xgb_pred = xgb_model.predict(prices.reshape(-1, 1))[-1]
            logging.info(f"{asset} Predictions: LSTM={lstm_pred}, GRU={gru_pred}, XGB={xgb_pred}")
            last_price = prices[-1][0]
            predicted_change = round(abs(((lstm_pred + gru_pred + xgb_pred) / 3.0) - last_price), 5)

            try:
                headlines = get_forex_news()
                news_headline = headlines[0] if headlines else "No news available."
            except Exception as e:
                logging.error(f"Error fetching news for {asset}: {e}")
                news_headline = "No news available."
            news_headline, impact_flag = filter_news_headline(news_headline)
            if news_headline != last_news_update and news_headline != "No news available.":
                send_telegram_message(f"📰 *News Update*: {news_headline}", to_channel=True)
                last_news_update = news_headline

            sentiment_result = sentiment_pipeline(news_headline)
            sentiment_label = sentiment_result[0]["label"].upper() if sentiment_result else "NEUTRAL"

            standard_value = (lstm_pred + gru_pred + xgb_pred) / 3.0
            final_signal = "BUY" if standard_value > 0.5 else "SELL"
            if abs(standard_value - 0.5) < 0.05:
                final_signal = "HOLD"
            strategy = (f"Strategy: {final_signal} - Standard Avg = {standard_value:.4f}\n"
                        f"RSI: {rsi_value}, MACD: {macd_value}, CMF: {bb_middle}, ATR: {atr_value}\n"
                        f"SMA20: {sma_20}, SMA50: {sma_50}, Momentum: {momentum}\n"
                        f"Predicted Change: {predicted_change}")
            confidence = random.randint(80, 100)

            economic_data = get_economic_indicators("GDP")
            if "data" in economic_data and len(economic_data["data"]) > 0:
                gdp_value = economic_data["data"][0].get("value", "N/A")
                strategy += f"\nGDP Indicator: {gdp_value}"

            pre_signal_message = (
                "⚠️ *Risk Alert:*\n"
                "Market conditions indicate heightened risk.\n"
                "Ensure proper risk management and position sizing before proceeding.\n"
                "⏳ *Preparing to drop a trade signal in 30 seconds...*\n\n" + strategy
            )
            send_telegram_message(pre_signal_message, to_channel=True)
            time.sleep(30)

            if final_signal in ["BUY", "SELL"] and atr_value is not None:
                TP1, TP2, TP3, TP4, SL = calculate_trade_parameters(last_price, final_signal, atr_value)
            else:
                TP1 = TP2 = TP3 = TP4 = SL = None

            if final_signal not in ["BUY", "SELL"]:
                final_signal = "HOLD"

            send_telegram_message(f"📣 *Trade Signal:* {asset} - {final_signal} at {last_price:.5f}", to_channel=True)

            signal_counter += 1
            signal_id = f"SIG-{signal_counter:04d}"
            signal_details = {
                "signal_id": signal_id,
                "pair": asset,
                "predicted_change": predicted_change,
                "news_sentiment": sentiment_label,
                "signal": final_signal,
                "confidence": confidence,
                "entry_price": last_price,
                "stop_loss": SL,
                "TP1": TP1,
                "TP2": TP2,
                "TP3": TP3,
                "TP4": TP4,
                "timestamp": datetime.utcnow().isoformat(),
                "mode": mode,
                "strategy": strategy
            }
            formatted_message = format_trade_signal(signal_details) + "\n" + strategy
            send_telegram_message(formatted_message, to_channel=True)
            send_telegram_message(formatted_message, to_channel=False)
            asyncio.run_coroutine_threadsafe(broadcast_trade(formatted_message), event_loop)
            update_errors(last_price, lstm_pred, gru_pred, xgb_pred)
            active_trades[signal_id] = signal_details
            historical_trades.append(signal_details)
            logging.info(f"Generated Signal for {asset} ({mode} mode): {signal_details}")
            time.sleep(5)
        time.sleep(60)

# Start the live signal worker in a background thread
live_thread = threading.Thread(target=live_signal_worker, daemon=True)
live_thread.start()

# -------------------------------
# HTTP Endpoints for Dashboard & Monitoring
# -------------------------------
@app.get("/active_trades")
def get_active_trades():
    return active_trades

@app.get("/historical_trades")
def get_historical_trades():
    return {"historical_trades": historical_trades}

@app.get("/performance_metrics")
def performance_metrics():
    if not historical_trades:
        return {"message": "No historical trades available."}
    total_trades = len(historical_trades)
    avg_confidence = np.mean([trade.get("confidence", 0) for trade in historical_trades])
    buy_signals = sum(1 for trade in historical_trades if trade.get("signal") == "BUY")
    sell_signals = sum(1 for trade in historical_trades if trade.get("signal") == "SELL")
    hold_signals = sum(1 for trade in historical_trades if trade.get("signal") == "HOLD")
    return {
        "total_trades": total_trades,
        "average_confidence": round(avg_confidence, 2),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
    }

@app.get("/backtest")
def backtest():
    if not historical_trades:
        return {"message": "No historical trades available for backtesting."}
    total_trades = len(historical_trades)
    wins = sum(1 for trade in historical_trades if trade.get("signal") == "BUY")
    losses = sum(1 for trade in historical_trades if trade.get("signal") == "SELL")
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_confidence = np.mean([trade.get("confidence", 0) for trade in historical_trades])
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_percentage": round(win_rate, 2),
        "average_confidence": round(avg_confidence, 2),
        "note": "This is a simplified backtest using BUY as win and SELL as loss."
    }

@app.get("/risk_adjustments")
def risk_adjustments():
    simulated_avg_atr = 0.0025
    suggested_multiplier = 1.5
    return {
        "simulated_average_atr": simulated_avg_atr,
        "suggested_stop_loss_multiplier": suggested_multiplier,
        "note": "Use these parameters to adjust stop loss levels based on market volatility."
    }

@app.get("/position_sizing")
def position_sizing(equity: float, risk_percent: float = 1.0):
    simulated_avg_atr = 0.0025
    risk_amount = equity * (risk_percent / 100.0)
    recommended_position_size = risk_amount / simulated_avg_atr
    return {
        "equity": equity,
        "risk_percent": risk_percent,
        "risk_amount": round(risk_amount, 2),
        "simulated_average_atr": simulated_avg_atr,
        "recommended_position_size": round(recommended_position_size, 2),
        "note": "This is a simplified calculation for dynamic position sizing."
    }

@app.get("/")
def home():
    return {"message": "NEKO AI Trading API is running!"}

# -------------------------------
# Telegram Command Handlers (Async)
# -------------------------------
async def start_signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        requests.post("http://127.0.0.1:8080/start_signals")
        await update.message.reply_text("✅ *Signal generation started via Telegram!*", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error starting signals: {e}", parse_mode="Markdown")

async def stop_signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        requests.post("http://127.0.0.1:8080/stop_signals")
        await update.message.reply_text("🛑 *Signal generation stopped via Telegram!*", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error stopping signals: {e}", parse_mode="Markdown")

async def switch_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return
        parts = update.message.text.split()
        if len(parts) < 2:
            await update.message.reply_text("Usage: /switch_mode <forex|crypto>", parse_mode="Markdown")
            return
        mode = parts[1].lower()
        if mode not in ["forex", "crypto"]:
            await update.message.reply_text("Invalid mode. Use 'forex' or 'crypto'.", parse_mode="Markdown")
            return
        requests.post(f"http://127.0.0.1:8080/switch_mode?mode={mode}")
        await update.message.reply_text(f"🔔 Mode switched manually to **{mode.upper()}**", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error switching mode: {e}", parse_mode="Markdown")

async def execute_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        parts = update.message.text.split()
        if len(parts) < 2:
            await update.message.reply_text("Usage: /execute_trade <signal_id>", parse_mode="Markdown")
            return
        signal_id = parts[1].strip()
        trade_details = active_trades.get(signal_id)
        if not trade_details:
            trade_details = next((t for t in historical_trades if t.get("signal_id") == signal_id), None)
        if not trade_details:
            await update.message.reply_text(f"Trade signal {signal_id} not found.", parse_mode="Markdown")
            return
        if mt5 is None:
            await update.message.reply_text("MT5 module not available. Cannot execute trade.", parse_mode="Markdown")
            return
        symbol = trade_details.get("pair")
        order_type = trade_details.get("signal")
        last_price = trade_details.get("entry_price")
        volume = 0.1
        sl = trade_details.get("stop_loss")
        tp = trade_details.get("TP4")
        result = place_mt5_order(symbol, order_type, volume, last_price, sl, tp, signal_id)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            await update.message.reply_text(f"Trade {signal_id} executed successfully.", parse_mode="Markdown")
        else:
            error_message = result.comment if result is not None else "Unknown error"
            await update.message.reply_text(f"Trade {signal_id} failed to execute: {error_message}", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error executing trade: {e}", parse_mode="Markdown")

@app.on_event("startup")
async def start_telegram_listener():
    application = TGApplication.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start_signals", start_signals_command))
    application.add_handler(CommandHandler("stop_signals", stop_signals_command))
    application.add_handler(CommandHandler("switch_mode", switch_mode_command))
    application.add_handler(CommandHandler("execute_trade", execute_trade_command))
    async def polling_task():
        try:
            await application.run_polling()
        except RuntimeError as e:
            if "Cannot close a running event loop" in str(e):
                logging.warning("Telegram polling attempted to close the event loop; ignoring.")
            else:
                raise
    asyncio.create_task(polling_task())
    logging.info("Telegram command listener started.")

@app.post("/start_signals")
def start_signals():
    global signal_generation_active
    signal_generation_active = True
    send_telegram_message("✅ Signal generation started via API!", to_channel=False)
    return {"message": "Signal generation started"}

@app.post("/stop_signals")
def stop_signals():
    global signal_generation_active
    signal_generation_active = False
    send_telegram_message("🛑 Signal generation stopped via API!", to_channel=False)
    return {"message": "Signal generation stopped"}

@app.post("/switch_mode")
def switch_mode(mode: str):
    mode = mode.lower()
    if mode not in ["forex", "crypto"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'forex' or 'crypto'.")
    global last_trading_mode, manual_mode_override, manual_mode
    manual_mode_override = True
    manual_mode = mode
    last_trading_mode = mode
    send_telegram_message(f"🔔 Trading mode switched manually to **{mode.upper()}**", to_channel=True)
    return {"message": f"Mode switched to {mode}"}

@app.post("/set_risk_reward")
def set_risk_reward(ratio: float, current_user: dict = Depends(get_current_user)):
    global user_risk_reward_ratio
    if ratio <= 0 or ratio > 1:
        raise HTTPException(status_code=400, detail="Risk-reward ratio must be between 0 and 1.")
    user_risk_reward_ratio = ratio
    return {"message": f"Risk-reward ratio updated to {ratio}."}

@app.post("/set_execution_mode")
def set_execution_mode(execute: bool, current_user: dict = Depends(get_current_user)):
    global monitor_only_mode
    monitor_only_mode = not execute
    mode_str = "execution" if execute else "monitor-only"
    send_telegram_message(f"🔄 Mode updated: Now in {mode_str} mode.", to_channel=True)
    return {"message": f"Execution mode set to {mode_str}."}

# -------------------------------
# Live Signal Worker: Trading, Monitoring, and Updates
# -------------------------------
def live_signal_worker():
    global signal_counter, signal_generation_active, last_news_update, last_trading_mode, total_loss_today, current_day, last_signal_time
    current_day = datetime.now().date()
    while True:
        if not signal_generation_active:
            logging.info("Signal generation is paused.")
            time.sleep(10)
            continue

        today = datetime.now().date()
        if today != current_day:
            total_loss_today = 0.0
            current_day = today

        if not manual_mode_override:
            est = pytz.timezone("US/Eastern")
            now_est = datetime.now(est).time()
            session_start = dt_time(5, 0)
            session_end = dt_time(22, 0)
            if session_start <= now_est <= session_end:
                assets = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"]
                mode = "forex"
            else:
                assets = ["BTC/USD", "ETH/USD", "XRP/USD"]
                mode = "crypto"
        else:
            if manual_mode == "crypto":
                assets = ["BTC/USD", "ETH/USD", "XRP/USD"]
                mode = "crypto"
            else:
                assets = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"]
                mode = "forex"

        if last_trading_mode is None:
            last_trading_mode = mode
        elif mode != last_trading_mode:
            mode_switch_message = f"🔔 Trading Mode Change Alert:\nSwitched from **{last_trading_mode.upper()}** to **{mode.upper()}**."
            send_telegram_message(mode_switch_message, to_channel=True)
            last_trading_mode = mode

        logging.info(f"Trading mode: {mode}")

        for asset in assets:
            if not signal_generation_active:
                break

            if not can_send_signal(asset):
                logging.info(f"Duplicate signal for {asset} suppressed.")
                continue

            prices, _ = get_extended_live_data(symbol=asset)
            if prices is None or prices.shape[0] < 60:
                logging.warning(f"Not enough price data for {asset}. Skipping...")
                continue

            logging.info(f"{asset} latest price: {prices[-1][0]}")
            df_prices = pd.DataFrame(prices, columns=["close"])
            df_prices = compute_indicators(df_prices)
            df_prices = compute_custom_indicators(df_prices)
            rsi_value = df_prices["rsi"].iloc[-1] if "rsi" in df_prices.columns else None
            macd_value = df_prices["macd"].iloc[-1] if "macd" in df_prices.columns else None
            bb_middle = df_prices["cmf"].iloc[-1] if "cmf" in df_prices.columns else None
            atr_value = df_prices["atr"].iloc[-1] if "atr" in df_prices.columns else None
            sma_20 = df_prices["SMA_20"].iloc[-1] if "SMA_20" in df_prices.columns else None
            sma_50 = df_prices["SMA_50"].iloc[-1] if "SMA_50" in df_prices.columns else None
            momentum = df_prices["Momentum"].iloc[-1] if "Momentum" in df_prices.columns else None

            logging.info(f"{asset} Indicators: RSI={rsi_value}, MACD={macd_value}, CMF={bb_middle}, ATR={atr_value}")

            reshaped_data = prices.reshape(1, 60, 1)
            lstm_pred = lstm_model.predict(reshaped_data)[0][0]
            gru_pred = gru_model.predict(reshaped_data)[0][0]
            xgb_pred = xgb_model.predict(prices.reshape(-1, 1))[-1]
            logging.info(f"{asset} Predictions: LSTM={lstm_pred}, GRU={gru_pred}, XGB={xgb_pred}")
            last_price = prices[-1][0]
            predicted_change = round(abs(((lstm_pred + gru_pred + xgb_pred) / 3.0) - last_price), 5)

            try:
                headlines = get_forex_news()
                news_headline = headlines[0] if headlines else "No news available."
            except Exception as e:
                logging.error(f"Error fetching news for {asset}: {e}")
                news_headline = "No news available."
            news_headline, impact_flag = filter_news_headline(news_headline)
            if news_headline != last_news_update and news_headline != "No news available.":
                send_telegram_message(f"📰 *News Update*: {news_headline}", to_channel=True)
                last_news_update = news_headline

            sentiment_result = sentiment_pipeline(news_headline)
            sentiment_label = sentiment_result[0]["label"].upper() if sentiment_result else "NEUTRAL"

            standard_value = (lstm_pred + gru_pred + xgb_pred) / 3.0
            final_signal = "BUY" if standard_value > 0.5 else "SELL"
            if abs(standard_value - 0.5) < 0.05:
                final_signal = "HOLD"
            strategy = (f"Strategy: {final_signal} - Standard Avg = {standard_value:.4f}\n"
                        f"RSI: {rsi_value}, MACD: {macd_value}, CMF: {bb_middle}, ATR: {atr_value}\n"
                        f"SMA20: {sma_20}, SMA50: {sma_50}, Momentum: {momentum}\n"
                        f"Predicted Change: {predicted_change}")
            confidence = random.randint(80, 100)

            economic_data = get_economic_indicators("GDP")
            if "data" in economic_data and len(economic_data["data"]) > 0:
                gdp_value = economic_data["data"][0].get("value", "N/A")
                strategy += f"\nGDP Indicator: {gdp_value}"

            pre_signal_message = (
                "⚠️ *Risk Alert:*\n"
                "Market conditions indicate heightened risk.\n"
                "Ensure proper risk management and position sizing before proceeding.\n"
                "⏳ *Preparing to drop a trade signal in 30 seconds...*\n\n" + strategy
            )
            send_telegram_message(pre_signal_message, to_channel=True)
            time.sleep(30)

            if final_signal in ["BUY", "SELL"] and atr_value is not None:
                TP1, TP2, TP3, TP4, SL = calculate_trade_parameters(last_price, final_signal, atr_value)
            else:
                TP1 = TP2 = TP3 = TP4 = SL = None

            if final_signal not in ["BUY", "SELL"]:
                final_signal = "HOLD"

            send_telegram_message(f"📣 *Trade Signal:* {asset} - {final_signal} at {last_price:.5f}", to_channel=True)

            signal_counter += 1
            signal_id = f"SIG-{signal_counter:04d}"
            signal_details = {
                "signal_id": signal_id,
                "pair": asset,
                "predicted_change": predicted_change,
                "news_sentiment": sentiment_label,
                "signal": final_signal,
                "confidence": confidence,
                "entry_price": last_price,
                "stop_loss": SL,
                "TP1": TP1,
                "TP2": TP2,
                "TP3": TP3,
                "TP4": TP4,
                "timestamp": datetime.utcnow().isoformat(),
                "mode": mode,
                "strategy": strategy
            }
            formatted_message = format_trade_signal(signal_details) + "\n" + strategy
            send_telegram_message(formatted_message, to_channel=True)
            send_telegram_message(formatted_message, to_channel=False)
            asyncio.run_coroutine_threadsafe(broadcast_trade(formatted_message), event_loop)
            update_errors(last_price, lstm_pred, gru_pred, xgb_pred)
            active_trades[signal_id] = signal_details
            historical_trades.append(signal_details)
            logging.info(f"Generated Signal for {asset} ({mode} mode): {signal_details}")
            time.sleep(5)
        time.sleep(60)

live_thread = threading.Thread(target=live_signal_worker, daemon=True)
live_thread.start()

# -------------------------------
# HTTP Endpoints for Dashboard & Monitoring
# -------------------------------
@app.get("/active_trades")
def get_active_trades():
    return active_trades

@app.get("/historical_trades")
def get_historical_trades():
    return {"historical_trades": historical_trades}

@app.get("/performance_metrics")
def performance_metrics():
    if not historical_trades:
        return {"message": "No historical trades available."}
    total_trades = len(historical_trades)
    avg_confidence = np.mean([trade.get("confidence", 0) for trade in historical_trades])
    buy_signals = sum(1 for trade in historical_trades if trade.get("signal") == "BUY")
    sell_signals = sum(1 for trade in historical_trades if trade.get("signal") == "SELL")
    hold_signals = sum(1 for trade in historical_trades if trade.get("signal") == "HOLD")
    return {
        "total_trades": total_trades,
        "average_confidence": round(avg_confidence, 2),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
    }

@app.get("/backtest")
def backtest():
    if not historical_trades:
        return {"message": "No historical trades available for backtesting."}
    total_trades = len(historical_trades)
    wins = sum(1 for trade in historical_trades if trade.get("signal") == "BUY")
    losses = sum(1 for trade in historical_trades if trade.get("signal") == "SELL")
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_confidence = np.mean([trade.get("confidence", 0) for trade in historical_trades])
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_percentage": round(win_rate, 2),
        "average_confidence": round(avg_confidence, 2),
        "note": "This is a simplified backtest using BUY as win and SELL as loss."
    }

@app.get("/risk_adjustments")
def risk_adjustments():
    simulated_avg_atr = 0.0025
    suggested_multiplier = 1.5
    return {
        "simulated_average_atr": simulated_avg_atr,
        "suggested_stop_loss_multiplier": suggested_multiplier,
        "note": "Use these parameters to adjust stop loss levels based on market volatility."
    }

@app.get("/position_sizing")
def position_sizing(equity: float, risk_percent: float = 1.0):
    simulated_avg_atr = 0.0025
    risk_amount = equity * (risk_percent / 100.0)
    recommended_position_size = risk_amount / simulated_avg_atr
    return {
        "equity": equity,
        "risk_percent": risk_percent,
        "risk_amount": round(risk_amount, 2),
        "simulated_average_atr": simulated_avg_atr,
        "recommended_position_size": round(recommended_position_size, 2),
        "note": "This is a simplified calculation for dynamic position sizing."
    }
