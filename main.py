# main.py - NEKO AI Premium Trader (Complete & Enhanced with MT5 Trading)

import os
import time
import threading
import asyncio
import numpy as np
import pandas as pd
import requests
import logging
import pytz
import json
import aiohttp
import MetaTrader5 as mt5
from datetime import datetime, timedelta, time as dt_time
from fastapi import FastAPI, HTTPException, WebSocket, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet
import joblib
from passlib.context import CryptContext
from jose import JWTError, jwt
from queue import Queue
from dotenv import load_dotenv
from stable_baselines3 import DQN, PPO
from tensorflow.keras.losses import MeanSquaredError  # Preserved if needed
import nest_asyncio
nest_asyncio.apply()

# ---------------- FastAPI App Initialization ----------------
# Place the FastAPI app definition at the top so it is available to all code.
app = FastAPI(title="NEKO AI Premium Trader")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Environment Setup ----------------
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@yourdomain.com")

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"NEWS_API_KEY: {'set' if NEWSAPI_KEY else 'NOT set'}")
logger.info(f"TELEGRAM_BOT_TOKEN: {'set' if TELEGRAM_BOT_TOKEN else 'NOT set'}")
logger.info(f"TELEGRAM_CHAT_ID: {'set' if TELEGRAM_CHAT_ID else 'NOT set'}")
logger.info(f"TELEGRAM_CHANNEL_ID: {'set' if TELEGRAM_CHANNEL_ID else 'NOT set'}")

# ---------------- Disable GPU for TensorFlow ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
nest_asyncio.apply()

# ---------------- Global Variables ----------------
models = {}  # Global dictionary for all models
model_lock = threading.Lock()  # Lock for safe model loading
signal_generation_active = True
active_trades = {}
historical_trades = []
signal_queue = Queue()

# ---------------- Database Setup ----------------
Base = declarative_base()

class TradeSignal(Base):
    __tablename__ = "trade_signals"
    id = Column(Integer, primary_key=True, index=True)
    asset = Column(String)
    signal = Column(String)
    price = Column(Float)
    confidence = Column(Float)
    tp_levels = Column(JSON)
    sl_level = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    indicators = Column(JSON)
    predictions = Column(JSON)
    news_sentiment = Column(String)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    broker_info = Column(String, nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

class Feedback(BaseModel):
    signal_id: str
    rating: int
    comment: Optional[str] = None

# ---------------- Security Configuration ----------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ---------------- Email Sending Function ----------------
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
        logger.error(f"Error sending email: {e}")

# ---------------- API Dependency ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- User Endpoints ----------------
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    username: str
    email: EmailStr
    registered_at: datetime

from fastapi.security import OAuth2PasswordRequestForm

@app.post("/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
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

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

class BrokerLink(BaseModel):
    broker: str
    broker_username: str
    broker_password: str
    broker_server: str

@app.post("/link_broker")
def link_broker(broker_info: BrokerLink, db: Session = Depends(get_db)):
    current_user = db.query(User).first()  # In production, use proper authentication
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")
    current_user.broker_info = json.dumps(broker_info.dict())
    db.add(current_user)
    db.commit()
    return {"message": "Broker account linked successfully."}

# ---------------- Adjustable Trading & Risk Settings ----------------
TP_MULTIPLIER = 2.0
SL_MULTIPLIER = 1.5
MONITOR_INTERVAL = 30
PROFIT_THRESHOLD_CLOSE = 3.0
DECAY_FACTOR = 0.9

manual_mode_override = False
manual_mode: Optional[str] = None
monitor_only_mode = False

# ---------------- Additional Global Variables ----------------
connected_clients: List[WebSocket] = []
trade_queue = Queue()
active_trades: Dict[str, Any] = {}
historical_trades: List[Dict[str, Any]] = []
trading_mode_override: Optional[str] = None
last_signal_time: Dict[str, float] = {}
total_loss_today = 0.0
current_day = datetime.now().date()
user_risk_reward_ratio = 0.5
feedback_list: List[Dict[str, Any]] = []

# ---------------- Intelligent Rate Limiter ----------------
class IntelligentRateLimiter:
    def __init__(self, max_calls=8, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        
    async def acquire(self):
        while True:
            now = time.time()
            self.calls = [t for t in self.calls if t > now - self.period]
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            sleep_time = self.period - (now - self.calls[0])
            logger.warning(f"Rate limit exceeded. Sleeping {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)

rate_limiter = IntelligentRateLimiter()

# ---------------- MT5 Trading Functions ----------------
def initialize_mt5(max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.info("MT5 initialized successfully.")
            return True
        else:
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            attempts += 1
            time.sleep(5)
    return False

def shutdown_mt5():
    mt5.shutdown()

def place_mt5_order(symbol, order_type, volume, price, sl, tp, signal_id):
    request = {
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
    }
    result = mt5.order_send(request)
    return result

async def monitor_trades():
    """Continuously monitor open MT5 trades and update Telegram if SL/TP adjustments are needed."""
    while True:
        open_trades = mt5.positions_get()
        if open_trades:
            for trade in open_trades:
                profit = trade.profit
                symbol = trade.symbol
                ticket = trade.ticket
                if profit > 10:  # Adjust threshold as needed
                    message = f"Trade {ticket} on {symbol} has profit {profit:.2f}. Consider reviewing SL/TP."
                    send_telegram_message(message, to_channel=False)
        await asyncio.sleep(60)

# ---------------- Model Loading System ----------------
def load_all_models():
    global models
    model_paths = {
        "lstm": "models/lstm_model.keras",
        "gru": "models/gru_model.keras",
        "cnn": "models/cnn_model.keras",
        "transformer": "models/transformer_model.keras",
        "xgb": "models/xgboost_model.json",
        "lightgbm": "models/lightgbm_model.txt",
        "catboost": "models/catboost_model.cbm",
        "stacking": "models/stacking_model.pkl",
        "gaussian": "models/gaussian_process_model.pkl",
        "prophet": "models/prophet_model.pkl",
        "sentiment": "models/sentiment_model.keras",
        "sentiment_tokenizer": "models/sentiment_tokenizer.pkl",
        "anomaly": "models/anomaly_model.keras",
        "scaler": "models/price_scaler.pkl",
        "dqn": "models/dqn_model.zip",
        "ppo": "models/ppo_model.zip",
        "svr": "models/svr_model.pkl"
    }
    with model_lock:
        for name, path in model_paths.items():
            try:
                if not os.path.exists(path):
                    logger.warning(f"Model file missing: {path}")
                    continue
                if name.endswith(".keras"):
                    models[name] = tf.keras.models.load_model(path)
                elif name == "xgb":
                    models[name] = xgb.XGBRegressor()
                    models[name].load_model(path)
                elif name == "lightgbm":
                    models[name] = lgb.Booster(model_file=path, params={'predict_disable_shape_check': True})
                elif name == "catboost":
                    models[name] = CatBoostRegressor()
                    models[name].load_model(path)
                elif name.endswith(".pkl"):
                    with open(path, "rb") as f:
                        models[name] = joblib.load(f)
                elif name in ["dqn", "ppo"]:
                    models[name] = DQN.load(path) if name == "dqn" else PPO.load(path)
                elif name == "svr":
                    with open(path, "rb") as f:
                        models[name] = joblib.load(f)
                logger.info(f"Successfully loaded {name}")
            except Exception as e:
                logger.error(f"Error loading {name}: {str(e)}")
    return models

# ---------------- Data Fetching Function ----------------
async def fetch_market_data(symbol: str, asset_type: str):
    await rate_limiter.acquire()
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "outputsize": 200,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON"
    }
    if asset_type == "crypto":
        params.update({"exchange": "Binance", "type": "digital_currency"})
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"API error {response.status}: {await response.text()}")
                    return None
                data = await response.json()
                if "values" not in data:
                    logger.error(f"Invalid data format for {symbol}: {data}")
                    return None
                df = pd.DataFrame(data["values"])
                df = df.rename(columns={"datetime": "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp", ascending=True)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df.dropna()
    except Exception as e:
        logger.error(f"Data fetch failed for {symbol}: {str(e)}")
        return None

# ---------------- News & Sentiment ----------------
async def fetch_market_news(asset_type: str):
    if not NEWSAPI_KEY:
        logger.warning("NEWS_API_KEY is not set; skipping news fetch.")
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": "business", "language": "en", "apiKey": NEWSAPI_KEY, "pageSize": 5}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"News API error {response.status}: {await response.text()}")
                    return []
                data = await response.json()
                return data.get("articles", [])
    except Exception as e:
        logger.error(f"News fetch failed: {str(e)}")
        return []

def analyze_sentiment(news_articles: List[dict]) -> str:
    positive_words = ["up", "gain", "bull", "soar", "surge", "improve", "rally", "optimism"]
    negative_words = ["down", "loss", "bear", "plunge", "drop", "decline", "pessimism"]
    if not news_articles:
        return "NEUTRAL"
    pos_count = 0
    neg_count = 0
    for article in news_articles:
        text = (article.get("title", "") + " " + article.get("description", "")).lower()
        for word in positive_words:
            if word in text:
                pos_count += 1
        for word in negative_words:
            if word in text:
                neg_count += 1
    if pos_count > neg_count:
        return "POSITIVE"
    elif neg_count > pos_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# ---------------- Professional Indicators ----------------
def compute_professional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['RSI'] = ta.rsi(df['close'], length=14).fillna(50)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = df.join(macd, how='left')
        bb = ta.bbands(df['close'], length=20, std=2)
        df = df.join(bb, how='left')
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx['ADX_14']
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        df['ICHIMOKU'] = ichimoku[0]['ITS_9']
        volume = df['volume'] if 'volume' in df.columns else df['close']
        df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], volume)
        df['OBV'] = ta.obv(df['close'], volume)
        df['TREND_STRENGTH'] = df['ADX'].fillna(0)
        df['MOMENTUM_SCORE'] = df['RSI'].fillna(50) / 100
    except Exception as e:
        logger.error(f"Indicator error: {str(e)}")
    df = df.ffill().bfill()
    return df

# ---------------- Feature Engineering ----------------
def prepare_features(df: pd.DataFrame, model_type: str):
    try:
        features = df[['close', 'RSI', 'MACD_12_26_9', 'BBU_20_2.0', 
                       'BBL_20_2.0', 'ATR', 'ADX', 'TREND_STRENGTH']].dropna()
        if model_type == "ml":
            return features.tail(10).values.flatten().reshape(1, -1)
        elif model_type == "dl":
            return features.tail(100).values.reshape(1, 100, 8)
        elif model_type == "rl":
            return features.tail(1).values.reshape(1, -1)
    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}")
        return None

# ---------------- Prediction Generation ----------------
def generate_institutional_predictions(df: pd.DataFrame) -> dict:
    predictions = {}
    try:
        ml_data = prepare_features(df, "ml")
        dl_data = prepare_features(df, "dl")
        rl_data = prepare_features(df, "rl")
        for model in ["lstm", "gru", "transformer", "cnn"]:
            if model in models and dl_data is not None:
                try:
                    predictions[model] = float(models[model].predict(dl_data)[0][0])
                except Exception as e:
                    logger.error(f"{model} prediction failed: {str(e)}")
        for model in ["xgb", "lightgbm", "catboost", "svr", "stacking", "gaussian"]:
            if model in models and ml_data is not None:
                try:
                    if model == "lightgbm":
                        pred = models[model].predict(ml_data, predict_disable_shape_check=True)
                    else:
                        pred = models[model].predict(ml_data)
                    predictions[model] = float(pred[0])
                except Exception as e:
                    if model == "xgb":
                        logger.error(f"{model} prediction skipped due to error: {str(e)}")
                    else:
                        logger.error(f"{model} prediction failed: {str(e)}")
        for model in ["dqn", "ppo"]:
            if model in models and rl_data is not None:
                try:
                    action, _ = models[model].predict(rl_data)
                    predictions[model] = float(action[0])
                except Exception as e:
                    logger.error(f"{model} prediction failed: {str(e)}")
        if "prophet" in models and not df.empty:
            try:
                prophet_df = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
                future = models["prophet"].make_future_dataframe(periods=1, freq='min')
                forecast = models["prophet"].predict(future)
                predictions["prophet"] = forecast['yhat'].iloc[-1]
            except Exception as e:
                logger.error(f"Prophet prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction generation error: {str(e)}")
    return predictions

# ---------------- Trade Level Computation ----------------
def compute_trade_levels(price: float, atr: float, side: str) -> Tuple[float, List[float]]:
    if side == "BUY":
        stop_loss = price - 1.5 * atr
        tps = [price + 1.5 * atr, price + 3 * atr, price + 4.5 * atr]
    else:
        stop_loss = price + 1.5 * atr
        tps = [price - 1.5 * atr, price - 3 * atr, price - 4.5 * atr]
    return stop_loss, tps

# ---------------- Signal Generation ----------------
def generate_institutional_signal(df: pd.DataFrame, predictions: dict, asset: str) -> dict:
    price = df["close"].iloc[-1] if not df.empty else 0
    try:
        atr = df["ATRr_14"].iloc[-1] if "ATRr_14" in df.columns else 0.0
    except Exception:
        atr = 0.0
    signal = {
        "asset": asset,
        "action": "HOLD",
        "price": price,
        "confidence": 0,
        "timestamp": datetime.utcnow().isoformat(),
        "indicators": {
            "RSI": df["RSI_14"].iloc[-1] if "RSI_14" in df.columns else None,
            "ATR": atr,
            "Trend Strength": df["TREND_STRENGTH"].iloc[-1] if "TREND_STRENGTH" in df.columns else None
        },
        "predictions": predictions,
        "news_sentiment": "NEUTRAL",
        "tp_levels": [],
        "sl_level": None,
        "trade_mode": None
    }
    try:
        valid_preds = [p for p in predictions.values() if not np.isnan(p)]
        if len(valid_preds) < 3:
            return signal
        prediction_avg = sum(valid_preds) / len(valid_preds)
        price_diff = prediction_avg - price
        risk_ratio = abs(price_diff) / atr if atr > 0 else 0
        base_confidence = risk_ratio * 100 if risk_ratio < 0.5 else 50 + (risk_ratio - 0.5) * 100
        consensus_std = np.std(valid_preds)
        consensus_factor = 1 - min(consensus_std / (abs(prediction_avg) + 1e-5), 1)
        confidence = min(100, base_confidence * consensus_factor)
        if price_diff > 0:
            signal["action"] = "BUY"
            signal["confidence"] = confidence
        elif price_diff < 0:
            signal["action"] = "SELL"
            signal["confidence"] = confidence
        if signal["action"] in ["BUY", "SELL"] and atr > 0:
            sl, tps = compute_trade_levels(price, atr, signal["action"])
            signal["sl_level"] = sl
            signal["tp_levels"] = tps
    except Exception as e:
        logger.error(f"Signal generation error: {str(e)}")
    return signal

# ---------------- Trading Hours Logic ----------------
def is_forex_trading_hours() -> bool:
    now = datetime.utcnow().time()
    start = dt_time(8, 0, 0)
    end = dt_time(17, 0, 0)
    return start <= now <= end

def get_asset_universe() -> Dict[str, List[str]]:
    global trading_mode_override
    if trading_mode_override:
        if trading_mode_override.upper() == "FOREX":
            return {"forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"]}
        elif trading_mode_override.upper() == "CRYPTO":
            return {"crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD", "ADA/USD"]}
    if is_forex_trading_hours():
        return {"forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"]}
    else:
        return {"crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD", "ADA/USD"]}

# ---------------- Telegram Messaging Helper ----------------
def send_telegram_message(message: str, to_channel: bool = False):
    chat_id = TELEGRAM_CHANNEL_ID if to_channel else TELEGRAM_CHAT_ID
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logging.debug(f"Message sent to {'Channel' if to_channel else 'Group'}.")
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

# ---------------- Telegram Command Handlers ----------------
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

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
            await update.message.reply_text("Usage: /mode <forex|crypto|auto>", parse_mode="Markdown")
            return
        mode = parts[1].lower()
        if mode not in ["forex", "crypto", "auto"]:
            await update.message.reply_text("Invalid mode. Use 'forex', 'crypto', or 'auto'.", parse_mode="Markdown")
            return
        requests.post(f"http://127.0.0.1:8080/switch_mode?mode={mode}")
        await update.message.reply_text(f"🔔 Trading mode switched manually to **{mode.upper()}**", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error switching mode: {e}", parse_mode="Markdown")

# ---------------- Telegram Webhook for Commands ----------------
@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    global signal_generation_active, trading_mode_override, manual_mode
    data = await request.json()
    message = data.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id", "")
    logger.info(f"Received Telegram command from chat {chat_id}: {text}")
    response_text = "Command not recognized. Available commands: /start_signals, /stop_signals, /mode <forex|crypto|auto>"
    if text.startswith("/start_signals"):
        signal_generation_active = True
        response_text = "✅ Signal generation started via Telegram!"
    elif text.startswith("/stop_signals"):
        signal_generation_active = False
        response_text = "🛑 Signal generation stopped via Telegram!"
    elif text.startswith("/mode"):
        parts = text.split()
        if len(parts) >= 2:
            mode = parts[1].lower()
            if mode in ["forex", "crypto"]:
                trading_mode_override = mode
                manual_mode = mode
                response_text = f"🔔 Trading mode overridden to {mode.upper()}."
            elif mode == "auto":
                trading_mode_override = None
                manual_mode = None
                response_text = "🔔 Trading mode set to automatic (time-based)."
            else:
                response_text = "Invalid mode. Use '/mode forex', '/mode crypto', or '/mode auto'."
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": response_text, "parse_mode": "Markdown"}
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload)
    except Exception as e:
        logger.error(f"Error sending Telegram command response: {str(e)}")
    return {"status": "ok"}

# ---------------- Telegram Polling Listener ----------------
async def start_telegram_listener():
    from telegram.ext import Application as TGApplication, CommandHandler
    application = TGApplication.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start_signals", start_signals_command))
    application.add_handler(CommandHandler("stop_signals", stop_signals_command))
    application.add_handler(CommandHandler("mode", switch_mode_command))
    try:
        await application.run_polling()
    except RuntimeError as e:
        if "Cannot close a running event loop" in str(e):
            logger.warning("Telegram polling attempted to close the event loop; ignoring.")
        else:
            raise

# ---------------- Trading Execution Worker ----------------
async def premium_trading_worker():
    asset_universe = {
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"],
        "crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD", "ADA/USD"]
    }
    
    if not initialize_mt5():
        logger.error("MT5 initialization failed. Trading will not proceed.")
    
    asyncio.create_task(monitor_trades())
    
    while signal_generation_active:
        start_time = time.time()
        for asset_type, assets in asset_universe.items():
            for asset in assets:
                try:
                    logger.info(f"Fetching market data for {asset} ({asset_type})")
                    df = await fetch_market_data(asset, asset_type)
                    if df is None or df.empty:
                        continue
                    df = compute_professional_indicators(df)
                    if df.empty:
                        continue
                    predictions = generate_institutional_predictions(df)
                    signal = generate_institutional_signal(df, predictions, asset)
                    signal["trade_mode"] = "CRYPTO" if asset_type == "crypto" else "FOREX"
                    news = await fetch_market_news(asset_type)
                    signal["news_sentiment"] = analyze_sentiment(news)
                    
                    if signal["action"] != "HOLD" and signal["confidence"] > 65:
                        await send_pre_signal_message(signal)
                        await asyncio.sleep(30)
                        order_volume = 0.1
                        mt5_result = place_mt5_order(
                            symbol=asset,
                            order_type=signal["action"],
                            volume=order_volume,
                            price=signal["price"],
                            sl=signal["sl_level"],
                            tp=signal["tp_levels"][0],
                            signal_id="Auto"
                        )
                        logger.info(f"MT5 order result for {asset}: {mt5_result}")
                        with SessionLocal() as db:
                            db_signal = TradeSignal(
                                asset=signal["asset"],
                                signal=signal["action"],
                                price=signal["price"],
                                confidence=signal["confidence"],
                                tp_levels=signal["tp_levels"],
                                sl_level=signal["sl_level"],
                                indicators=signal["indicators"],
                                predictions=signal["predictions"],
                                news_sentiment=signal["news_sentiment"]
                            )
                            db.add(db_signal)
                            db.commit()
                            db.refresh(db_signal)
                            signal["id"] = db_signal.id
                        logger.info(f"Stored signal for {asset} with ID {signal['id']}.")
                        await send_telegram_alert(signal)
                    else:
                        logger.info(f"No valid signal for {asset} (action: {signal['action']}, confidence: {signal['confidence']}).")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Asset processing error for {asset}: {str(e)}")
                    continue
        elapsed = time.time() - start_time
        logger.info(f"Worker cycle completed in {elapsed:.2f} seconds.")
        await asyncio.sleep(max(0, 60 - elapsed))

# ---------------- Telegram Messaging: Pre-Signal and Final Signal ----------------
async def send_pre_signal_message(signal: dict):
    message = (
        "🤖✨ **AlphaTrader Pre-Signal Notification** ✨🤖\n"
        f"Trade Mode: {signal.get('trade_mode', 'N/A')}\n"
        f"Asset: {signal['asset']}\n"
        f"Action: {signal['action']}\n"
        f"Price: {signal['price']:.5f}\n"
        "⚠️ **Risk Warning:** Extreme market volatility ahead. Final confirmation in 30 seconds...\n"
        "— NekoAITrader"
    )
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHANNEL_ID, "text": message, "parse_mode": "Markdown"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Pre-signal Telegram API error {response.status}: {await response.text()}")
                else:
                    logger.info(f"Pre-signal message sent to channel {TELEGRAM_CHANNEL_ID}.")
    except Exception as e:
        logger.error(f"Pre-signal Telegram error: {str(e)}")

async def send_telegram_alert(signal: dict):
    if signal["confidence"] < 65:
        logger.info("Signal confidence below threshold; alert not sent.")
        return
    if not TELEGRAM_BOT_TOKEN or not (TELEGRAM_CHAT_ID and TELEGRAM_CHANNEL_ID):
        logger.error("Telegram credentials not set; cannot send alert.")
        return
    predicted_move = 0.0
    if "lightgbm" in signal.get("predictions", {}):
        predicted_move = signal["predictions"]["lightgbm"] - signal["price"]
    message = (
        "🚀 **Premium Trade Signal** 🚀\n"
        f"Signal ID: {signal.get('id', 'N/A')}\n"
        f"Trade Mode: {signal.get('trade_mode', 'N/A')}\n"
        f"📈 {signal['asset']} | {signal['action']} | Confidence: {signal['confidence']:.1f}%\n"
        f"💰 Price: {signal['price']:.5f}\n"
        f"🔮 Predicted Move: {predicted_move:.5f}\n"
        f"🛑 Stop Loss: {signal['sl_level']:.5f}\n"
        f"🎯 Take Profits: {', '.join(map(lambda x: f'{x:.5f}', signal['tp_levels']))}\n"
        "⚠️ **Risk Management Warning:** Please manage your risk appropriately.\n"
        f"📰 Sentiment: {signal['news_sentiment']}"
    )
    for dest in [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID]:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": dest, "text": message, "parse_mode": "Markdown"}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Telegram API error {response.status}: {await response.text()}")
                    else:
                        logger.info(f"Alert sent to destination: {dest}")
        except Exception as e:
            logger.error(f"Telegram error: {str(e)}")

# ---------------- WebSocket Endpoint ----------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            signal = signal_queue.get()
            await websocket.send_json(signal)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            break

# ---------------- Additional Endpoints ----------------
@app.get("/signals", response_model=List[dict])
def get_signals(limit: int = 10):
    with SessionLocal() as db:
        signals = db.query(TradeSignal).order_by(TradeSignal.timestamp.desc()).limit(limit).all()
        return [{
            "asset": s.asset,
            "signal": s.signal,
            "price": s.price,
            "confidence": s.confidence,
            "timestamp": s.timestamp.isoformat(),
            "tp_levels": s.tp_levels,
            "sl_level": s.sl_level,
            "indicators": s.indicators,
            "predictions": s.predictions,
            "news_sentiment": s.news_sentiment
        } for s in signals]

@app.get("/feedback")
def get_feedback():
    return {"feedback": feedback_list}

@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    feedback_entry = {
        "signal_id": feedback.signal_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "timestamp": datetime.utcnow().isoformat()
    }
    feedback_list.append(feedback_entry)
    logger.debug(f"Feedback received: {feedback_entry}")
    return {"message": "Feedback received", "data": feedback_entry}

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
        "note": "Simplified backtest: BUY is win, SELL is loss."
    }

@app.get("/risk_adjustments")
def risk_adjustments():
    simulated_avg_atr = 0.0025
    suggested_multiplier = SL_MULTIPLIER
    return {
        "simulated_average_atr": simulated_avg_atr,
        "suggested_stop_loss_multiplier": suggested_multiplier,
        "note": "Adjust stop loss based on volatility."
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
        "note": "Simplified position sizing."
    }

@app.get("/predict")
def predict():
    try:
        sample_input = np.random.random((1, 3))
        predictions = {}
        if "cnn" in models:
            predictions["cnn"] = models["cnn"].predict(sample_input).tolist()
        if "lightgbm" in models:
            predictions["lightgbm"] = models["lightgbm"].predict(sample_input.reshape(1, -1)).tolist()
        if "stacking" in models:
            predictions["stacking"] = models["stacking"].predict(sample_input).tolist()
        return {"status": "success", "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Startup & Shutdown ----------------
@app.on_event("startup")
async def startup_event():
    load_all_models()
    asyncio.create_task(premium_trading_worker())
    asyncio.create_task(start_telegram_listener())
    asyncio.create_task(monitor_trades())
    logger.info("NekoAI Premium Trader Active and Telegram command listener started.")

@app.on_event("shutdown")
async def shutdown_event():
    global signal_generation_active
    signal_generation_active = False
    shutdown_mt5()
    logger.info("NekoAI Premium Trader Shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
