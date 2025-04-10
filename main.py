# main.py - NEKO AIBot Premium Trader (Multi‑User & Dynamic Risk Management Version)
import sys
import random
import asyncio
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import time
import threading
import numpy as np
import pandas as pd
# Add to imports
import traceback
import requests
import logging
import json
import aiohttp
from datetime import datetime, timedelta, time as dt_time
from fastapi import FastAPI, HTTPException, WebSocket, Request, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, joinedload
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet
import joblib
import pandas_ta as ta
import yfinance as yf 
from passlib.context import CryptContext
from jose import JWTError, jwt
from diskcache import Cache
from dotenv import load_dotenv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import DummyVecEnv
from diskcache import Cache
import threading
import nest_asyncio
nest_asyncio.apply()
from tenacity import retry, stop_after_attempt, wait_exponential
from cryptography.fernet import Fernet
# Add these with your other imports
from telegram import Update
from telegram.ext import (
    Application as TGApplication,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

# Create a synchronous lock for model loading
model_sync_lock = threading.Lock()

# Try to import MetaTrader5; if unavailable, disable trading functions.
import os
from dotenv import load_dotenv
load_dotenv()

USE_MOCK_MT5 = os.getenv("USE_MOCK_MT5", "true").lower() == "true"

try:
    if USE_MOCK_MT5:
        print("[INFO] Running in MOCK mode. Using mock_mt5 module.")
        from mock_mt5 import MockMT5, MockOrderSendResult
        mt5 = MockMT5()
        mt5.OrderSendResult = MockOrderSendResult
        MT5_AVAILABLE = False  # Optional: you can make this True if mocks should count
    else:
        import MetaTrader5 as mt5
        print("[INFO] MetaTrader5 successfully imported.")
        MT5_AVAILABLE = True
except ImportError:
    print("[WARN] MetaTrader5 package not available. Trading functionality will be disabled.")
    MT5_AVAILABLE = False
    mt5 = None

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
encryption_key = os.getenv("ENCRYPTION_KEY")
if not encryption_key:
    encryption_key = Fernet.generate_key().decode()
    print("Please set ENCRYPTION_KEY in your environment to:", encryption_key)
cipher_suite = Fernet(encryption_key)

# Replace the existing cache initialization with:
cache = Cache("./.api_cache")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize market data store and other attributes
app.market_data_store = cache  # assuming you're using the same cache instance
app.data_lock = asyncio.Lock()
app.shutdown_flag = asyncio.Event()
app.models = {}  # This will store your RL models

nest_asyncio.apply()

# Helper functions for data access
def get_historical_data():
    data = app.market_data_store.get("historical_data")
    return data if data is not None else pd.DataFrame()

def get_recent_data():
    data = app.market_data_store.get("recent_data")
    return data if data is not None else pd.DataFrame()

# Attach helper methods to app instance
app.get_historical_data = get_historical_data
app.get_recent_data = get_recent_data

# During startup, after initializing the app

async def reset_daily_counters():
    global twelvedata_calls_today, alphavantage_calls_today
    while True:
        now = datetime.utcnow()
        next_reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        await asyncio.sleep((next_reset - now).total_seconds())
        twelvedata_calls_today = 0
        alphavantage_calls_today = 0
        logger.info("Daily API counters reset")

@app.on_event("startup")
async def startup_event():
    load_all_models()
    asyncio.create_task(start_telegram_listener())
    asyncio.create_task(refresh_market_data())
    asyncio.create_task(premium_trading_worker())
    asyncio.create_task(monitor_trades())
    
    # Initialize retrain system AFTER setting up market_data_store
    retrain_system = RetrainRLSystem(app)
    app.retrain_rl_system = retrain_system  # Attach to app instance
    asyncio.create_task(retrain_system.continuous_retraining_scheduler())
    asyncio.create_task(retrain_system.monitor_performance())
    
    # Start other tasks
    asyncio.create_task(reset_daily_counters())
    async def process_signals_and_trade():
        """
        This function processes trading signals and executes trades.
        Add your logic here to handle signals and trade execution.
        """
        while not app.shutdown_flag.is_set():
            try:
                # Example logic for processing signals
                signal = await signal_queue.get()
                await process_trade_signal(signal)
            except Exception as e:
                logger.error(f"Error in process_signals_and_trade: {str(e)}")
            await asyncio.sleep(1)
    
        asyncio.create_task(process_signals_and_trade())
    
    logger.info("NekoAIBot Active with Enhanced API Management and Trade Execution Integration")

async def monitor_rate_limits():
    while True:
        logger.info(f"TwelveData: {twelvedata_limiter.remaining} calls remaining")
        logger.info(f"AlphaVantage: {alphavantage_limiter.remaining} calls remaining")
        await asyncio.sleep(30)

NEWSAPI_KEY = os.getenv("NEWS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")  # Not used now
try:
    MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
except Exception:
    MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
TWELVEDATA_DAILY_LIMIT = int(os.getenv("TWELVEDATA_DAILY_LIMIT", "800"))
ALPHAVANTAGE_DAILY_LIMIT = int(os.getenv("ALPHAVANTAGE_DAILY_LIMIT", "500"))
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@yourdomain.com")

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs.
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"NEWS_API_KEY: {'set' if NEWSAPI_KEY else 'NOT set'}")
logger.info(f"TELEGRAM_BOT_TOKEN: {'set' if TELEGRAM_BOT_TOKEN else 'NOT set'}")
logger.info(f"TELEGRAM_CHAT_ID: {'set' if TELEGRAM_CHAT_ID else 'NOT set'}")
logger.info(f"TELEGRAM_CHANNEL_ID: {'set' if TELEGRAM_CHANNEL_ID else 'NOT set'}")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Add after imports but before app initialization
class IntelligentRateLimiter:
    def __init__(self, max_calls: int = 7, period: int = 65):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        async with self.lock:
            while True:
                now = time.time()
                self.calls = [t for t in self.calls if t > now - self.period]
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                sleep_time = self.period - (now - self.calls[0])
                logger.warning(f"Rate limit exceeded. Sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                
    @property
    def remaining(self):
        now = time.time()
        return self.max_calls - len([t for t in self.calls if t > now - self.period])

# Initialize after class definition
twelvedata_limiter = IntelligentRateLimiter(max_calls=45, period=60)
alphavantage_limiter = IntelligentRateLimiter(max_calls=5, period=60)
       # Ensure these globals are defined at the top of your file
MAX_CONCURRENT_TRADES = 5
RISK_PER_TRADE = 0.02
MIN_RISK_REWARD = 3.0     
# Replace existing signal_counter_lock definition
signal_counter = 1
signal_counter_lock = asyncio.Lock()  # Changed from threading.Lock
trading_mode_override = None
models = {}
model_lock = threading.Lock()
signal_generation_active = True
api_calls_today = 0
shutdown_flag = asyncio.Event()
signal_queue = asyncio.Queue()
sent_signal_ids = set()
sent_alert_ids = set()
historical_trades = []
# Set these to True to force trade execution and market open override
FORCE_TRADE_OVERRIDE = True
FORCE_MARKET_OPEN_OVERRIDE = True
# Add with other global variables
PROFIT_THRESHOLD_PERCENT = 0.02  # Close trade when profit reaches 2%
TRAILING_PROFIT_THRESHOLD_PERCENT = 0.01  # Adjust SL if profit is above 1%
twelvedata_calls_today = 0
alphavantage_calls_today = 0
twelvedata_limiter = IntelligentRateLimiter(max_calls=45, period=60)
alphavantage_limiter = IntelligentRateLimiter(max_calls=5, period=60)

# Add to switch_mode function
if not TELEGRAM_BOT_TOKEN:
    raise HTTPException(status_code=500, detail="Telegram bot token not configured")

if not TELEGRAM_CHANNEL_ID:
    raise HTTPException(status_code=500, detail="Telegram channel ID not set")

# Add this near your configuration checks
if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN not set - bot will not function")
if not TELEGRAM_CHANNEL_ID:
    logger.warning("TELEGRAM_CHANNEL_ID not set - channel notifications disabled")

# Verify MT5 connection
if MT5_AVAILABLE and not mt5.initialize():
    logger.critical("MT5 connection failed - trades won't execute")

global_market_data: Dict[str, pd.DataFrame] = {}

Base = declarative_base()
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./trading.db"), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TradeSignal(Base):
    __tablename__ = "trade_signals"
    id = Column(Integer, primary_key=True, index=True)
    asset = Column(String)
    signal = Column(String)
    price = Column(Float)
    confidence = Column(Float)
    tp_levels = Column(String)
    sl_level = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    indicators = Column(String)
    predictions = Column(String)
    news_sentiment = Column(String)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    registered_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Integer, default=0)
    broker_credentials = relationship("BrokerCredentials", back_populates="user", uselist=False)

class BrokerCredentials(Base):
    __tablename__ = "broker_credentials"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    broker = Column(String)
    encrypted_username = Column(String)
    encrypted_password = Column(String)
    broker_server = Column(String)
    user = relationship("User", back_populates="broker_credentials")
    
    def set_credentials(self, username, password):
        self.encrypted_username = cipher_suite.encrypt(username.encode()).decode()
        self.encrypted_password = cipher_suite.encrypt(password.encode()).decode()
        
    def get_credentials(self):
        username = cipher_suite.decrypt(self.encrypted_username.encode()).decode()
        password = cipher_suite.decrypt(self.encrypted_password.encode()).decode()
        return username, password

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__default_rounds=12,
    bcrypt__ident="2b",
    bcrypt__min_rounds=10,
    bcrypt__max_rounds=14
)

def get_password_hash(password: str):
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hash generation error: {str(e)}")
        raise

def verify_password(plain_password: str, hashed_password: str):
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {str(e)}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

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
            port=int(SMTP_PORT),
            username=SMTP_USER,
            password=SMTP_PASSWORD,
            start_tls=True,
        )
    except Exception as e:
        logger.error(f"Error sending email: {e}")

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    username: str
    email: EmailStr
    registered_at: datetime

@app.post("/register", response_model=UserOut)
async def register(user: UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    token = create_access_token({"sub": new_user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    background_tasks.add_task(send_email, new_user.email, "Your Login Token", f"Your token is: {token}")
    return new_user

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

class BrokerInfo(BaseModel):
    broker: str
    username: str
    password: str
    broker_server: str

@app.post("/update_broker", response_model=None)
def update_broker(broker_info: BrokerInfo, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    creds = db.query(BrokerCredentials).filter(BrokerCredentials.user_id == current_user.id).first()
    if not creds:
        creds = BrokerCredentials(user_id=current_user.id)
    creds.broker = broker_info.broker
    creds.broker_server = broker_info.broker_server
    creds.set_credentials(broker_info.username, broker_info.password)
    db.add(creds)
    db.commit()
    return {"message": "Broker credentials updated successfully."}

user_trading_tasks: Dict[int, asyncio.Task] = {}

async def start_user_trading(user: User):
    if user.id in user_trading_tasks:
        logger.info(f"Trading task already running for user {user.username} (ID: {user.id}).")
        return
    task = asyncio.create_task(premium_trading_worker())
    user_trading_tasks[user.id] = task
    logger.info(f"Started trading task for user {user.username} (ID: {user.id}).")

async def stop_user_trading(user: User):
    task = user_trading_tasks.get(user.id)
    if task:
        task.cancel()
        del user_trading_tasks[user.id]
        logger.info(f"Stopped trading task for user {user.username} (ID: {user.id}).")

@app.post("/start_trading", response_model=None)
async def start_trading(current_user: User = Depends(get_current_user)):
    await start_user_trading(current_user)
    return {"message": f"Trading task started for user {current_user.username}"}

@app.post("/stop_trading", response_model=None)
async def stop_trading(current_user: User = Depends(get_current_user)):
    await stop_user_trading(current_user)
    return {"message": f"Trading task stopped for user {current_user.username}"}

from fastapi import Request, HTTPException, Depends
from datetime import datetime
import aiohttp

@app.post("/switch_mode")
async def switch_mode(mode: str, request: Request, current_user: User = Depends(get_current_user)):
    """
    API endpoint to switch trading mode.
    When the mode is changed, it forces an immediate refresh of the asset universe and notifies connected
    clients as well as the Telegram channel. It also enqueues a mode change event into the signal_queue
    so that trade signals are generated immediately under the new mode.
    """
    global trading_mode_override

    # Validate and normalize mode input
    mode = mode.lower().strip()
    if mode not in ["forex", "crypto", "auto"]:
        logger.error(f"Invalid mode attempt: {mode}")
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'forex', 'crypto', or 'auto'")

    old_mode = trading_mode_override
    trading_mode_override = None if mode == "auto" else mode
    logger.info(f"Mode switched from {old_mode} to {trading_mode_override or 'auto'}")

    # Force an immediate refresh of the asset universe.
    assets = get_asset_universe(force_refresh=True)
    logger.info(f"New asset universe: {assets}")

    # Trigger immediate market data refresh for new assets
    asyncio.create_task(refresh_assets_immediately(assets))

    # Synchronously send a Telegram notification.
    try:
        msg = f"⚡ [NekoAIBot] Mode FORCE-CHANGED to {mode.upper()}\nActive assets: {', '.join(assets)}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            await session.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHANNEL_ID, "text": msg, "parse_mode": "HTML"}
            )
    except Exception as e:
        logger.error(f"Telegram notification error: {str(e)}")

    # Notify connected WebSocket clients.
    try:
        notification = {
            "event": "mode_change",
            "new_mode": mode,
            "active_assets": assets,
            "timestamp": datetime.utcnow().isoformat()
        }
        await broadcast_to_websockets(notification)
    except Exception as e:
        logger.error(f"WebSocket broadcast error: {str(e)}")

    # Enqueue a mode change event into the global signal_queue.
    try:
        await signal_queue.put({
            "event": "mode_change",
            "new_mode": mode,
            "active_assets": assets,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error enqueuing mode change event: {str(e)}")

    return {
        "status": "overridden",
        "new_mode": mode,
        "active_assets": assets,
        "timestamp": datetime.utcnow().isoformat()
    }

# PASTE THIS ENTIRE CODE BLOCK INTO YOUR TRADING SYSTEM FILE
# (Replace previous versions of the TradingSystem class)

import pandas as pd
import numpy as np
import time
from datetime import datetime

if __name__ == "__main__":
    # Add the necessary code to run the application
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
async def refresh_assets_immediately(assets: List[str]):
    """Forceful mode-specific data refresh"""
    logger.info(f"Mode switch initiated - refreshing {len(assets)} assets")
    
    valid_assets = []
    for asset in assets:
        is_crypto = "/USD" in asset and asset not in ["EUR/USD", "GBP/USD"]
        if (trading_mode_override == "forex" and not is_crypto) or \
           (trading_mode_override == "crypto" and is_crypto) or \
           trading_mode_override is None:
            valid_assets.append(asset)

    for asset in valid_assets:
        try:
            asset_type = "crypto" if "/USD" in asset else "forex"
            data = await fetch_market_data(asset, asset_type)
            if not data.empty:
                async with app.data_lock:
                    global_market_data[asset] = data
                    logger.info(f"Immediate refresh: {asset}")
            await asyncio.sleep(1)  # Rate limit protection
        except Exception as e:
            logger.error(f"Immediate refresh failed: {str(e)}")

# Helper function to broadcast a message to all connected WebSocket clients.
async def broadcast_to_websockets(message: dict):
    # This assumes you have a global asyncio.Queue called 'signal_queue'
    # which is read by your WebSocket endpoint to push updates to clients.
    await signal_queue.put(message)

async def _send_telegram_with_retry(payload: dict, max_retries: int = 2) -> bool:
    """Guaranteed delivery telegram sender with tight timeouts"""
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    logger.warning(f"Telegram attempt {attempt+1} failed: {await response.text()}")
        except Exception as e:
            logger.warning(f"Telegram send error (attempt {attempt+1}): {str(e)}")
        
        if attempt < max_retries:
            await asyncio.sleep(0.5)  # Brief delay between retries
    
    logger.error(f"Failed to send Telegram notification after {max_retries} retries")
    return False

def track_api_usage(source: str):
    global twelvedata_calls_today, alphavantage_calls_today
    if source == "TwelveData":
        twelvedata_calls_today += 1
        logger.info(f"TwelveData calls: {twelvedata_calls_today}/{TWELVEDATA_DAILY_LIMIT}")
        if twelvedata_calls_today >= TWELVEDATA_DAILY_LIMIT * 0.9:
            logger.warning("Approaching TwelveData daily limit!")
    elif source == "AlphaVantage":
        alphavantage_calls_today += 1
        logger.info(f"AlphaVantage calls: {alphavantage_calls_today}/{ALPHAVANTAGE_DAILY_LIMIT}")
        if alphavantage_calls_today >= ALPHAVANTAGE_DAILY_LIMIT * 0.9:
            logger.warning("Approaching AlphaVantage daily limit!")

def initialize_user_mt5(user: User, db: Session):
    creds = db.query(BrokerCredentials).filter(BrokerCredentials.user_id == user.id).first()
    if not creds:
        logger.error("No broker credentials found for user")
        return False
    username, password = creds.get_credentials()
    try:
        login_val = int(username)
    except Exception:
        login_val = username
    if MT5_AVAILABLE and mt5.initialize(login=login_val, password=password, server=creds.broker_server):
        logger.info("User-specific MT5 initialized successfully.")
        return True
    else:
        logger.error(f"User-specific MT5 initialization failed: {mt5.last_error() if MT5_AVAILABLE else 'MT5 not available'}")
        return False

def initialize_mt5_admin(max_attempts=3):
    """Initialize MT5 connection with version-safe checks"""
    if not MT5_AVAILABLE:
        return False

    # Check if already connected
    try:
        if mt5.terminal_info() is not None:
            return True
    except AttributeError as e:
        logger.debug(f"Terminal info check failed: {str(e)}")

    attempts = 0
    while attempts < max_attempts:
        try:
            login_val = int(MT5_LOGIN) if str(MT5_LOGIN).isdigit() else MT5_LOGIN
            if mt5.initialize(login=login_val, 
                            password=MT5_PASSWORD, 
                            server=MT5_SERVER):
                logger.info("Global MT5 (admin) initialized successfully.")
                return True
        except Exception as e:
            logger.error(f"Connection attempt {attempts+1} failed: {str(e)}")
            attempts += 1
            time.sleep(5)
    
    logger.error("MT5 initialization failed after maximum attempts")
    return False

async def monitor_trades():
    """
    Monitors open trades, adjusts SL/TP to secure profits, and closes trades once profit thresholds are reached.
    Sends Telegram updates when trades are closed or updated.
    """
    while not shutdown_flag.is_set():
        try:
            if MT5_AVAILABLE:
                try:
                    connected = mt5.terminal_info() is not None
                except Exception:
                    connected = False

                if not connected and not initialize_mt5_admin():
                    logger.error("Global MT5 init failed for monitoring")
                    await asyncio.sleep(60)
                    continue

                positions = mt5.positions_get()
                if positions is None:
                    logger.debug("No positions found")
                    await asyncio.sleep(5)
                    continue

                for pos in positions:
                    try:
                        tick = mt5.symbol_info_tick(pos.symbol)
                        if not tick:
                            continue
                        current_price = tick.ask if pos.type == mt5.ORDER_TYPE_BUY else tick.bid
                        open_price = pos.price_open
                        profit_pct = (current_price - open_price) / open_price if pos.type == mt5.ORDER_TYPE_BUY else (open_price - current_price) / open_price

                        # Close trade if profit threshold reached
                        if profit_pct >= PROFIT_THRESHOLD_PERCENT:
                            result = close_trade(pos)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                logger.info(f"Trade {pos.ticket} for {pos.symbol} closed at profit {profit_pct*100:.2f}%.")
                                close_signal = {
                                    "chat_id": TELEGRAM_CHANNEL_ID,
                                    "text": f"Trade {pos.ticket} for {pos.symbol} closed at profit {profit_pct*100:.2f}%.",
                                    "parse_mode": "HTML"
                                }
                                await _send_telegram_with_retry(close_signal)
                            else:
                                logger.error(f"Failed to close trade {pos.ticket} for {pos.symbol}: {result.comment if result else 'No result'}")
                        else:
                            # Adjust stop loss using trailing stop logic
                            buffer_threshold = current_price * 0.005  # 0.5% buffer
                            new_sl = pos.sl
                            symbol_info = mt5.symbol_info(pos.symbol)
                            if not symbol_info:
                                continue

                            if pos.type == mt5.ORDER_TYPE_BUY:
                                if profit_pct >= TRAILING_PROFIT_THRESHOLD_PERCENT:
                                    new_sl = round(current_price - buffer_threshold, symbol_info.digits)
                            else:
                                if profit_pct >= TRAILING_PROFIT_THRESHOLD_PERCENT:
                                    new_sl = round(current_price + buffer_threshold, symbol_info.digits)

                            if new_sl != pos.sl and new_sl > 0:
                                result = update_mt5_trade(pos, new_sl, pos.tp)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(f"Updated trade {pos.ticket} for {pos.symbol} with new SL: {new_sl}")
                                    update_signal = {
                                        "chat_id": TELEGRAM_CHANNEL_ID,
                                        "text": f"Trade {pos.ticket} for {pos.symbol} updated with new SL: {new_sl}",
                                        "parse_mode": "HTML"
                                    }
                                    await _send_telegram_with_retry(update_signal)
                                else:
                                    logger.error(f"Failed to update trade {pos.ticket}: {result.comment}")
                    except Exception as e:
                        logger.error(f"Error processing position {pos.ticket}: {str(e)}")
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Trade monitoring error: {str(e)}")
            await asyncio.sleep(30)

def close_trade(position):
    """
    Attempt to close the given position by sending an opposite order.
    """
    symbol = position.symbol
    mt5_symbol = symbol.replace("/", "")
    tick = mt5.symbol_info_tick(mt5_symbol)
    if not tick:
        logger.error(f"No tick data available for {symbol} when trying to close trade.")
        return None

    # Determine closing price based on order type.
    close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": close_price,
        "deviation": 10,
        "magic": getattr(position, "magic", 234000),
        "comment": f"Close Trade (Ticket: {position.ticket})",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    return result

def shutdown_mt5_admin():
    if MT5_AVAILABLE:
        mt5.shutdown()

def process_data(data: dict) -> pd.DataFrame:
    """Data processing with explicit datetime formats"""
    try:
        if "values" in data:  # TwelveData format
            df = pd.DataFrame(data["values"])
            df["timestamp"] = pd.to_datetime(df["datetime"], format='%Y-%m-%d %H:%M:%S')
            df = df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })
        else:  # AlphaVantage format
            ts_key = "Time Series (Digital Currency Daily)"
            df = pd.DataFrame.from_dict(data[ts_key], orient="index")
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
            df = df.rename(columns={
                "1a. open (USD)": "open",
                "2a. high (USD)": "high",
                "3a. low (USD)": "low", 
                "4a. close (USD)": "close"
            })

        numeric_cols = ["open", "high", "low", "close"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df.dropna().sort_index()
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return pd.DataFrame()
    
async def place_mt5_order(
    symbol: str,
    order_type: str,
    volume: float,
    price: float,
    signal_id: str,
    user: Optional[User] = None,
    sl_level: Optional[float] = None,
    tp_levels: Optional[List[float]] = None
):
    """Enhanced MT5 order placement with comprehensive validation"""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Initialize MT5 connection
            if user:
                with SessionLocal() as db:
                    if not initialize_user_mt5(user, db):
                        logger.error(f"User MT5 init failed for {user.username}")
                        return None
            elif not initialize_mt5_admin():
                logger.error("Global MT5 init failed")
                return None

            # Symbol validation
            mt5_symbol = symbol.replace("/", "")
            if not mt5.symbol_select(mt5_symbol, True):
                logger.error(f"Symbol {mt5_symbol} not available")
                return None

            symbol_info = mt5.symbol_info(mt5_symbol)
            if not symbol_info:
                logger.error("No symbol info available")
                return None

            # Validate required attributes
            required_attrs = ['stops_level', 'point', 'digits', 'volume_min']
            missing = [attr for attr in required_attrs if not hasattr(symbol_info, attr)]
            if missing:
                logger.error(f"Missing critical attributes for {symbol}: {missing}")
                return None

            # Get current market price
            tick = mt5.symbol_info_tick(mt5_symbol)
            if not tick:
                logger.error("No tick data available")
                raise Exception("No tick data")

            current_price = tick.ask if order_type.upper() == "BUY" else tick.bid
            price_diff = abs(current_price - price) / price
            if price_diff > 0.01:
                logger.warning(f"Price deviation too large: {price_diff:.2%}")
                return None

            # Calculate minimum stop distance
            min_stop_points = symbol_info.stops_level
            point_value = symbol_info.point
            min_stop_distance = min_stop_points * point_value

            # Adjust SL/TP levels
            if sl_level is not None:
                if order_type.upper() == "BUY":
                    sl_level = max(sl_level, current_price - min_stop_distance)
                else:
                    sl_level = min(sl_level, current_price + min_stop_distance)
                sl_level = round(sl_level, symbol_info.digits)

            if tp_levels:
                adjusted_tps = []
                for tp in tp_levels:
                    if order_type.upper() == "BUY":
                        tp = max(tp, current_price + min_stop_distance)
                    else:
                        tp = min(tp, current_price - min_stop_distance)
                    adjusted_tps.append(round(tp, symbol_info.digits))
                tp_levels = adjusted_tps

            # Validate volume
            if volume <= 0 or volume < symbol_info.volume_min:
                volume = max(calculate_lot_size(symbol, current_price, sl_level), symbol_info.volume_min)
                logger.info(f"Adjusted volume to {volume}")

            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "deviation": 10,
                "magic": 234000,
                "comment": f"NekoAITrader_{signal_id}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            if sl_level is not None:
                request["sl"] = sl_level
            if tp_levels:
                request["tp"] = tp_levels[0]

            # Execute order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                attempt += 1
                await asyncio.sleep(2)
                continue
                
            return result

        except Exception as e:
            logger.error(f"Order attempt {attempt+1} failed: {str(e)}")
            attempt += 1
            await asyncio.sleep(2)
            
    logger.error(f"Failed all {max_attempts} order attempts for {symbol}")
    return None

def update_mt5_trade(position, new_sl, new_tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol": position.symbol,
        "sl": new_sl,
        "tp": new_tp,
        "magic": getattr(position, "magic", 234000),
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    return result

class EnhancedCircuitBreaker:
    def __init__(self, max_failures=5, reset_time=300):
        self.failures = 0
        self.max_failures = max_failures
        self.reset_time = reset_time
        self.last_failure = None
        
    async def check(self):
        if self.failures >= self.max_failures:
            if (datetime.utcnow() - self.last_failure).total_seconds() < self.reset_time:
                logger.error("Circuit Breaker Tripped - All requests paused for 5 minutes")
                raise Exception("Service unavailable")
            self.reset()
        return True
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.utcnow()
        
    def reset(self):
        self.failures = 0
        self.last_failure = None

app.state.circuit_breaker = EnhancedCircuitBreaker()

async def fetch_yahoo_data(symbol: str, asset_type: str) -> pd.DataFrame:
    try:
        if asset_type == "forex":
            yf_symbol = f"{symbol.replace('/', '')}=X"  # Add =X for forex pairs
        elif "/" in symbol:
            yf_symbol = symbol.replace("/", "-")
        elif "-" not in symbol and len(symbol) > 3:
            yf_symbol = symbol[:-3] + "-" + symbol[-3:]
        else:
            yf_symbol = symbol
        # Explicitly set auto_adjust=True to handle dividends/splits properly
        data = yf.download(yf_symbol, period="1d", interval="1m", auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index().rename(columns={
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp", ascending=True).set_index("timestamp")
        return data
    except Exception as e:
        logger.error(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")
        return pd.DataFrame()

# ====================== UPDATED DATA FETCHING ======================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
async def fetch_market_data(symbol: str, asset_type: str) -> pd.DataFrame:
    """
    Fetch market data for the specified symbol and asset type.
    Tries TwelveData first, then falls back to Yahoo Finance.
    Always returns a DataFrame with indicators by processing through compute_professional_indicators.
    If no data is available, returns a DataFrame with default values.
    """
    try:
        current_mode = globals().get("trading_mode_override", None) or "auto"
        logger.info(f"Fetching data for {symbol} in mode {current_mode}")

        # Attempt TwelveData data source.
        twelve_data = await fetch_twelvedata(symbol, asset_type)
        if not twelve_data.empty:
            df_ind = compute_professional_indicators(twelve_data)
            if not df_ind.empty:
                return df_ind

        # Fallback: Yahoo Finance.
        yahoo_data = await fetch_yahoo_data(symbol, asset_type)
        if not yahoo_data.empty:
            df_ind = compute_professional_indicators(yahoo_data)
            if not df_ind.empty:
                return df_ind

        logger.error(f"No market data available for {symbol} from any source; using default values.")
        default_price = 1.0
        default_row = {
            'open': default_price, 'high': default_price, 'low': default_price, 'close': default_price,
            'SMA_50': default_price, 'RSI_14': 50, 'ADX_14': 25, '+DI_14': 25, '-DI_14': 25,
            'BB_MIDDLE': default_price
        }
        return pd.DataFrame([default_row])
    
    except Exception as e:
        logger.error(f"Data fetch failed for {symbol}: {e}")
        default_price = 1.0
        default_row = {
            'open': default_price, 'high': default_price, 'low': default_price, 'close': default_price,
            'SMA_50': default_price, 'RSI_14': 50, 'ADX_14': 25, '+DI_14': 25, '-DI_14': 25,
            'BB_MIDDLE': default_price
        }
        return pd.DataFrame([default_row])

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame structure with type checking"""
    required = ['open', 'high', 'low', 'close']
    return isinstance(df, pd.DataFrame) and not df.empty and all(col in df.columns for col in required)
    
async def fetch_alphavantage(symbol: str, asset_type: str):
    function = "DIGITAL_CURRENCY_DAILY" if asset_type == "crypto" else "FX_DAILY"
    params = {
        "function": function,
        "symbol": symbol.split("/")[0],
        "market": "USD" if asset_type == "crypto" else None,
        "apikey": ALPHAVANTAGE_API_KEY,
        "datatype": "json"
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
            async with session.get("https://www.alphavantage.co/query", params=params) as response:
                if response.status != 200:
                    logger.error(f"AlphaVantage HTTP error: {response.status}")
                    return None
                    
                data = await response.json()
                
                # Check for API error messages
                if "Error Message" in data:
                    logger.error(f"AlphaVantage API error: {data['Error Message']}")
                    return None
                if "Note" in data:  # Rate limit message
                    logger.warning(f"AlphaVantage rate limit: {data['Note']}")
                    return None
                    
                return data
                
    except asyncio.TimeoutError:
        logger.warning("AlphaVantage request timed out")
        return None
    except Exception as e:
        logger.error(f"AlphaVantage fetch failed: {str(e)}")
        return None
    
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
async def fetch_twelvedata(symbol: str, asset_type: str):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            params = {
                "symbol": symbol,
                "interval": "1min",
                "outputsize": 200,
                "apikey": TWELVEDATA_API_KEY,
                "format": "JSON"
            }
            if asset_type == "crypto":
                params.update({"exchange": "Binance", "type": "digital_currency"})

            async with session.get("https://api.twelvedata.com/time_series", 
                                 params=params,
                                 headers={"User-Agent": "NekoAIBot/1.0"}) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 300))  # 5min default
                    logger.warning(f"TwelveData rate limited. Sleeping {retry_after}s")
                    app.state.circuit_breaker.record_failure()
                    await asyncio.sleep(retry_after)
                    return pd.DataFrame()
                    
                data = await response.json()
                if data.get("code") == 429:
                    raise Exception("API limit exceeded")
                    
                return process_data(data)
                
    except Exception as e:
        logger.error(f"TwelveData fetch failed: {str(e)}")
        return pd.DataFrame()

def process_data(data: dict) -> pd.DataFrame:
    """Data processing with explicit datetime formats and column validation"""
    required_cols = ["open", "high", "low", "close"]
    
    try:
        if "values" in data:  # TwelveData format
            df = pd.DataFrame(data["values"])
            if not all(col in df.columns for col in ["datetime"] + required_cols):
                return pd.DataFrame()
            # Rest of processing...
        else:  # AlphaVantage format
            ts_key = "Time Series (Digital Currency Daily)"
            if ts_key not in data or not isinstance(data[ts_key], dict):
                return pd.DataFrame()
            
        # After processing, validate final columns
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns in processed data")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return pd.DataFrame()

def process_alpha_data(data: dict, asset_type: str) -> pd.DataFrame:
    ts_key = "Time Series (Digital Currency Daily)" if asset_type == "crypto" else "Time Series FX (Daily)"
    
    # Add null checks and type validation
    if not data.get(ts_key) or not isinstance(data[ts_key], dict):
        logger.warning(f"Invalid/missing {ts_key} in AlphaVantage response")
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame.from_dict(data[ts_key], orient="index").reset_index().rename(columns={"index": "timestamp"})
        
        # Handle different crypto/forex column mappings
        col_map = {
            "crypto": {
                "1a. open (USD)": "open",
                "2a. high (USD)": "high",
                "3a. low (USD)": "low", 
                "4a. close (USD)": "close"
            },
            "forex": {
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close"
            }
        }
        
        df = df.rename(columns=col_map[asset_type])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=True).set_index("timestamp")
        
        # Numeric conversion with error tracking
        numeric_cols = ["open", "high", "low", "close"]
        conversion_errors = 0
        
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].ffill().bfill()  # Forward/backward fill
            except Exception as e:
                logger.error(f"Error converting {col}: {str(e)}")
                conversion_errors += 1
                df[col] = np.nan
                
        if conversion_errors > 0:
            logger.warning(f"Failed to convert {conversion_errors} columns")
            
        return df.dropna(how="all").ffill().bfill()
        
    except Exception as e:
        logger.error(f"AlphaVantage processing failed: {str(e)}")
        return pd.DataFrame()

async def fetch_market_news(asset_type: str):
    if not NEWSAPI_KEY:
        logger.warning("NEWS_API_KEY not set; skipping news fetch.")
        return []
    params = {"category": "business", "language": "en", "apiKey": NEWSAPI_KEY, "pageSize": 5}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get("https://newsapi.org/v2/top-headlines", params=params) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return data.get("articles", [])
    except Exception as e:
        logger.error(f"News fetch failed: {str(e)}")
        return []

def analyze_sentiment(news_articles: List[dict]) -> str:
    positive_words = ["up", "gain", "bull", "soar", "surge", "improve", "rally", "optimism", "strong", "growth", "positive", "increase", "recovery", "breakout", "profit", "success", "win", "boom", "expansion", "outperform"]
    negative_words = ["down", "loss", "bear", "plunge", "drop", "decline", "pessimism", "weak", "crash", "slump", "risk", "volatile", "sell-off", "warning", "crisis", "trouble", "failure", "cut", "reduce", "bankrupt"]
    pos_count = 0
    neg_count = 0
    for article in news_articles:
        text = (article.get("title") or "") + " " + (article.get("description") or "")
        for word in positive_words:
            if word in text.lower():
                pos_count += 1
        for word in negative_words:
            if word in text.lower():
                neg_count += 1
    if pos_count > neg_count + 2:
        return "STRONG POSITIVE"
    if pos_count > neg_count:
        return "POSITIVE"
    if neg_count > pos_count + 2:
        return "STRONG NEGATIVE"
    if neg_count > pos_count:
        return "NEGATIVE"
    return "NEUTRAL"

def get_last_data(market_data_store, window_hours, resolution):
    """
    Retrieve the latest market data from the cache.
    Data is expected to be stored under a key formatted as:
    "market_data_last_{window_hours}_{resolution}".
    """
    key = f"market_data_last_{window_hours}_{resolution}"
    data = market_data_store.get(key)
    if data is None:
        logger.warning(f"No market data found for key: {key}")
    return data


# ====================== FEATURE ENGINEERING ======================
def prepare_features(df: pd.DataFrame, model_type: str):
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
            # Extract scalars using .item() to avoid FutureWarnings
            atr_val = features['ATR_14'].iloc[0].item() if hasattr(features['ATR_14'].iloc[0], 'item') else float(features['ATR_14'].iloc[0])
            close_val = features['close'].iloc[0].item() if hasattr(features['close'].iloc[0], 'item') else float(features['close'].iloc[0])
            # Use .loc to assign the new column without shape conflicts
            features.loc[:, 'ATR_Ratio'] = atr_val / close_val
            feature_array = features.values  # shape (1, n)
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
        logger.error(f"Feature error: {str(e)}")
        return None

# ====================== CORE INDICATOR CALCULATION ======================
# ----- Fix 1: Attach a get_historical_data method to the app -----
def get_historical_data():
    """
    Retrieve historical data from the cache.
    If no data exists, fetch default historical data for all forex-major and crypto assets,
    combine them into a single DataFrame (with an 'asset' column), cache the result, and return it.
    """
    data = app.market_data_store.get("historical_data")
    if data is None or data.empty:
        import yfinance as yf
        import pandas as pd

        # Define asset lists
        forex_assets = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]
        crypto_assets = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD"]
        all_assets = forex_assets + crypto_assets

        dfs = []
        for asset in all_assets:
            # Convert asset to appropriate yfinance ticker:
            if asset in forex_assets:
                # For forex pairs, Yahoo Finance typically uses the format "EURUSD=X"
                yf_symbol = asset.replace("/", "") + "=X"
            else:
                # For crypto, use the format "BTC-USD"
                yf_symbol = asset.replace("/", "-")
            try:
                df_asset = yf.download(yf_symbol, period="5d", interval="1h", progress=False)
                if not df_asset.empty:
                    df_asset["asset"] = asset
                    dfs.append(df_asset)
                    logger.info(f"Fetched historical data for {asset} ({yf_symbol}).")
                else:
                    logger.warning(f"No data returned for {asset} ({yf_symbol}).")
            except Exception as e:
                logger.error(f"Error fetching data for {asset} ({yf_symbol}): {e}")

        if dfs:
            combined = pd.concat(dfs)
            app.market_data_store.set("historical_data", combined)
            logger.info("Loaded default historical data for all forex-major and crypto assets.")
            return combined
        else:
            logger.error("Failed to load default historical data for any asset.")
            return pd.DataFrame()
    return data

def get_recent_data():
    import pandas as pd
    data = app.market_data_store.get("recent_data")
    if data is None:
        data = pd.DataFrame()  # Fallback if no data available
    return data

# Attach both methods so retraining can call either one.
app.get_historical_data = get_historical_data
app.get_recent_data = get_recent_data


def is_market_open(symbol: str) -> bool:
    """
    Determines whether the market is open for a given symbol.
    For forex pairs, it uses a time window (8:00 to 17:00 UTC) unless an override
    (FORCE_MARKET_OPEN_OVERRIDE) is set to True, in which case trading is allowed.
    Crypto pairs are assumed to be always open.
    """
    # If override is enabled, allow trades regardless of market hours.
    if globals().get("FORCE_MARKET_OPEN_OVERRIDE", False):
        return True

    # Crypto markets are always open (excluding specified forex pairs)
    if "/USD" in symbol and symbol not in ["EUR/USD", "GBP/USD"]:
        return True

    # Validate Forex hours
    now = datetime.utcnow().time()
    is_forex_hours = dt_time(8, 0) <= now <= dt_time(17, 0)

    # Check if symbol is Forex
    is_forex_pair = symbol in ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]

    # Enforce mode-specific trading: if mode override is set, only allow appropriate assets.
    if trading_mode_override == "forex" and not is_forex_pair:
        return False
    if trading_mode_override == "crypto" and is_forex_pair:
        return False

    return is_forex_hours if is_forex_pair else True

def get_validation_data():
    """
    Retrieve performance validation data for the trading system.
    This function calculates metrics based on historical trades such as total trades,
    win rate, average profit/loss, maximum profit/loss, and average trade duration.
    
    It expects the global variable 'historical_trades' to be a list of dictionaries
    with keys:
        - 'entry_price': float
        - 'exit_price': float
        - 'profit': float
        - 'entry_time': string or datetime (ISO formatted)
        - 'exit_time': string or datetime (ISO formatted)
    
    Returns:
        pd.DataFrame: A DataFrame containing performance metrics.
    """
    # Check if there are any historical trades available.
    if not historical_trades:
        logger.warning("No historical trades available for validation.")
        return pd.DataFrame()

    try:
        # Convert the list of trade dictionaries into a DataFrame.
        df = pd.DataFrame(historical_trades)
        
        total_trades = len(df)
        wins = df[df['profit'] > 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_profit = df['profit'].mean() if total_trades > 0 else 0.0
        max_profit = df['profit'].max() if total_trades > 0 else 0.0
        max_loss = df['profit'].min() if total_trades > 0 else 0.0
        
        # If timestamps are provided, compute trade durations in minutes.
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
            df['duration_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60.0
            avg_duration = df['duration_minutes'].mean()
        else:
            avg_duration = None

        # Collect all metrics into a dictionary.
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_duration_minutes': avg_duration
        }
        return pd.DataFrame([metrics])
    except Exception as e:
        logger.error(f"Validation data retrieval failed: {str(e)}")
        return pd.DataFrame()

# Attach the function to your FastAPI app so it can be accessed elsewhere.
app.get_validation_data = get_validation_data

# ----- Updated compute_professional_indicators to avoid column overlap -----
import numpy as np
import pandas_ta as ta

# ====================== UPDATED INDICATOR CALCULATION ======================
def compute_professional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators from price data and guarantee that all required columns are present.
    If data is missing or an indicator cannot be computed, default values are substituted.
    
    Required columns: 'open', 'high', 'low', 'close'
    Computed indicators:
      - SMA_50: 50-period SMA (fallback: 20-period SMA)
      - RSI_14: 14-period RSI (default: 50)
      - ADX_14: 14-period ADX (default: 25)
      - +DI_14: Positive DI (default: 25)
      - -DI_14: Negative DI (default: 25)
      - BB_MIDDLE: Bollinger Bands middle band (fallback: 20-period SMA)
    """
    try:
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Convert required columns to numeric and fill missing values
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        if df.empty:
            default_price = 1.0
            df = pd.DataFrame({'open': [default_price],
                               'high': [default_price],
                               'low': [default_price],
                               'close': [default_price]})
        
        closes = df['close']

        # SMA_50
        sma_50 = closes.rolling(window=50, min_periods=1).mean()
        if sma_50.isna().all():
            sma_50 = closes.rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = sma_50.ffill().bfill()

        # RSI_14 with fallback default 50
        rsi = ta.rsi(closes, length=14)
        if rsi is None or rsi.empty:
            df['RSI_14'] = 50
        else:
            df['RSI_14'] = rsi.fillna(50)

        # ADX and directional indicators; default value 25 if missing.
        adx_data = ta.adx(df['high'], df['low'], closes, length=14)
        if adx_data is None or adx_data.empty:
            df['ADX_14'] = 25
            df['+DI_14'] = 25
            df['-DI_14'] = 25
        else:
            for col_name, default_val in [('ADX_14', 25), ('DMP_14', 25), ('DMN_14', 25)]:
                if col_name not in adx_data.columns or adx_data[col_name].isna().all():
                    adx_data[col_name] = default_val
            df['ADX_14'] = adx_data['ADX_14'].fillna(25)
            df['+DI_14'] = adx_data['DMP_14'].fillna(25)
            df['-DI_14'] = adx_data['DMN_14'].fillna(25)

        # Bollinger Bands middle band; fallback to 20-period SMA.
        bb = ta.bbands(closes, length=20)
        if bb is None or bb.empty:
            df['BB_MIDDLE'] = closes.rolling(window=20, min_periods=1).mean()
        else:
            # Use the middle band (column index 1)
            middle = bb.iloc[:, 1]
            if middle.isna().all():
                middle = closes.rolling(window=20, min_periods=1).mean()
            df['BB_MIDDLE'] = middle.ffill().bfill()

        # Final cleanup: if any indicator column is entirely missing, set to default.
        for col, default in [('RSI_14', 50), ('ADX_14', 25), ('+DI_14', 25), ('-DI_14', 25)]:
            if df[col].isna().all():
                df[col] = default

        return df.ffill().bfill()

    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}")
        default_price = 1.0
        default_data = {
            'open': default_price, 'high': default_price, 'low': default_price, 'close': default_price,
            'SMA_50': default_price, 'RSI_14': 50, 'ADX_14': 25, '+DI_14': 25, '-DI_14': 25,
            'BB_MIDDLE': default_price
        }
        return pd.DataFrame([default_data])

# ====================== UPDATED TRADE LEVEL CALCULATION ======================
def compute_trade_levels(price: float, atr: float, side: str, symbol: str) -> dict:
    """Calculates trade levels with broker-specific validation"""
    try:
        if not MT5_AVAILABLE or not mt5.initialize():
            initialize_mt5_admin()

        mt5_symbol = symbol.replace("/", "")
        symbol_info = mt5.symbol_info(mt5_symbol)
        digits = 5
        min_stop_distance = 0
        point_value = 0.00001

        if symbol_info:
            digits = symbol_info.digits
            point_value = symbol_info.point
            min_stop_points = symbol_info.trade_stops_level
            min_stop_distance = min_stop_points * point_value * 1.2  # 20% buffer
            trade_allowed = symbol_info.trade_allowed
            if not trade_allowed:
                logger.error(f"Trading disabled for {symbol}")
                raise ValueError("Symbol trading disabled")

        # Validate ATR against minimum stop distance
        atr = max(atr, min_stop_distance * 1.5)
        price = round(price, digits)

        if side.upper() == "BUY":
            raw_sl = price - atr
            sl = max(raw_sl, price - (price * 0.05))  # Max 5% risk
            sl = round(sl - (min_stop_distance * 0.5), digits)  # Add buffer
            tp_levels = [round(price + (atr * mult), digits) for mult in [1, 2, 3]]
        else:
            raw_sl = price + atr
            sl = min(raw_sl, price + (price * 0.05))
            sl = round(sl + (min_stop_distance * 0.5), digits)
            tp_levels = [round(price - (atr * mult), digits) for mult in [1, 2, 3]]

        # Final validation against current price
        if side == "BUY" and sl >= price - min_stop_distance:
            sl = round(price - min_stop_distance * 1.5, digits)
        elif side == "SELL" and sl <= price + min_stop_distance:
            sl = round(price + min_stop_distance * 1.5, digits)

        return {
            "sl_level": sl,
            "tp_levels": tp_levels,
            "min_stop": min_stop_distance
        }

    except Exception as e:
        logger.error(f"Level calculation critical error: {str(e)}")
        return {
            "sl_level": round(price * 0.99, 5),
            "tp_levels": [round(price * 1.01, 5), round(price * 1.02, 5)],
            "min_stop": 0
        }
    
def generate_institutional_predictions(df: pd.DataFrame) -> dict:
    predictions = {}
    try:
        ml_data = prepare_features(df, "ml")
        dl_data = prepare_features(df, "dl")
        rl_data = prepare_features(df, "rl")

        if dl_data is not None:
            for model in ["lstm", "gru", "transformer", "cnn"]:
                if model in models:
                    try:
                        pred_val = models[model].predict(dl_data)
                        predictions[model] = float(pred_val[0][0])
                    except Exception as e:
                        logger.error(f"{model} prediction failed: {str(e)}")
                        predictions[model] = 0

        if ml_data is not None:
            for model in ["xgb", "lightgbm", "catboost", "svr", "stacking", "gaussian"]:
                if model in models:
                    try:
                        pred_val = models[model].predict(ml_data)
                        predictions[model] = float(pred_val[0])
                    except Exception as e:
                        logger.error(f"{model} prediction failed: {str(e)}")
                        predictions[model] = 0

        if rl_data is not None:
            for model in ["dqn", "ppo"]:
                if model in models:
                    try:
                        action, _ = models[model].predict(rl_data)
                        predictions[model] = float(action[0])
                    except Exception as e:
                        logger.error(f"{model} RL prediction failed: {str(e)}")
                        predictions[model] = 0

        if "prophet" in models and not df.empty:
            try:
                if 'timestamp' not in df.columns:
                    df = df.reset_index()
                prophet_df = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
                future = models["prophet"].make_future_dataframe(periods=1, freq='min')
                forecast = models["prophet"].predict(future)
                predictions["prophet"] = float(forecast['yhat'].iloc[-1])
            except Exception as e:
                logger.error(f"Prophet prediction failed: {str(e)}")
                predictions["prophet"] = 0

    except Exception as e:
        logger.error(f"Prediction generation error: {str(e)}")

    return predictions

def get_all_users():
    """Query the database to return all registered users."""
    with SessionLocal() as db:
        return db.query(User).all()

# Revised compute_trade_levels function to handle disabled trading gracefully
from static_config import STATIC_INSTRUMENT_PARAMS

def compute_trade_levels(price: float, atr: float, side: str, symbol: str) -> dict:
    """
    Calculates trade levels using a hybrid approach:
    - First, attempt to retrieve parameters dynamically via MT5.
    - If MT5 data is missing or indicates trading is disabled (and no override is active),
      fall back to a static configuration.
    
    Parameters:
      price (float): Current market price.
      atr (float): Average True Range value.
      side (str): "BUY" or "SELL".
      symbol (str): The asset symbol (e.g., "EUR/USD" or "BTC/USD").
      
    Returns:
      dict: Contains 'sl_level', 'tp_levels', and 'min_stop'.
    """
    try:
        price = float(price)
        atr = float(atr)
        side = str(side).upper()
        
        # Start with static defaults.
        params = STATIC_INSTRUMENT_PARAMS.get(symbol, {"digits": 5, "point_value": 0.00001, "trade_stops_level": 10})
        digits = params["digits"]
        point_value = params["point_value"]
        min_stop_points = params["trade_stops_level"]
        trading_allowed = True
        
        # If MT5 is available, try to pull dynamic data.
        if globals().get("MT5_AVAILABLE", False):
            mt5_symbol = symbol.replace("/", "")
            if not mt5.symbol_select(mt5_symbol, True):
                logger.warning(f"Symbol {mt5_symbol} could not be selected. Using static config.")
            symbol_info = mt5.symbol_info(mt5_symbol)
            if symbol_info:
                # Use dynamic values if available.
                digits = getattr(symbol_info, 'digits', digits)
                point_value = getattr(symbol_info, 'point', point_value)
                min_stop_points = getattr(symbol_info, 'trade_stops_level', min_stop_points)
                # Check trading_allowed flag from MT5.
                trading_allowed = getattr(symbol_info, 'trade_allowed', True)
                # If FORCE_TRADE_OVERRIDE is set, ignore MT5's flag.
                if globals().get("FORCE_TRADE_OVERRIDE", False):
                    trading_allowed = True
                    logger.info(f"Force override active: Forcing trading allowed for {symbol}.")
                elif not trading_allowed:
                    logger.error(f"MT5 indicates trading disabled for {symbol}; using static config fallback.")
                    fallback_sl = round(price * 0.99, digits)
                    fallback_tp = [round(price * 1.01, digits), round(price * 1.02, digits)]
                    return {"sl_level": fallback_sl, "tp_levels": fallback_tp, "min_stop": 0}
            else:
                logger.warning(f"No dynamic symbol info for {symbol}; using static config.")
        
        # Calculate minimum stop distance.
        min_stop_distance = (min_stop_points * point_value) * 1.2
        atr = max(atr, min_stop_distance * 1.5) if atr else min_stop_distance * 1.5
        price = round(price, digits)
        
        if side == "BUY":
            raw_sl = price - atr
            sl = max(raw_sl, price - (price * 0.05))
            sl = round(sl - (min_stop_distance * 0.5), digits)
            tp_levels = [round(price + (atr * m), digits) for m in [1, 2, 3]]
        elif side == "SELL":
            raw_sl = price + atr
            sl = min(raw_sl, price + (price * 0.05))
            sl = round(sl + (min_stop_distance * 0.5), digits)
            tp_levels = [round(price - (atr * m), digits) for m in [1, 2, 3]]
        else:
            fallback_sl = round(price * 0.99, digits)
            fallback_tp = [round(price * 1.01, digits), round(price * 1.02, digits)]
            return {"sl_level": fallback_sl, "tp_levels": fallback_tp, "min_stop": min_stop_distance}
        
        # Final adjustment.
        if side == "BUY" and sl >= price - min_stop_distance:
            sl = round(price - min_stop_distance * 1.5, digits)
        elif side == "SELL" and sl <= price + min_stop_distance:
            sl = round(price + min_stop_distance * 1.5, digits)
        
        return {"sl_level": sl, "tp_levels": tp_levels[:3], "min_stop": min_stop_distance}
    except Exception as e:
        logger.error(f"Level calculation error: {e}")
        fallback_sl = round(price * 0.99, digits)
        fallback_tp = [round(price * 1.01, digits), round(price * 1.02, digits)]
        return {"sl_level": fallback_sl, "tp_levels": fallback_tp, "min_stop": 0}

def safe_predictions(predictions: dict) -> dict:
    """
    Ensures that all model prediction values are valid numbers.
    If any value is invalid or NaN, it falls back to 0.
    """
    fallback = {
        "lstm": 0, "gru": 0, "transformer": 0, "cnn": 0,
        "xgb": 0, "lightgbm": 0, "catboost": 0, "svr": 0,
        "stacking": 0, "gaussian": 0, "prophet": 0
    }
    safe_preds = {}
    for key, fb in fallback.items():
        try:
            val = predictions.get(key, fb)
            # If the value is a pandas Series, extract the last element.
            if isinstance(val, pd.Series):
                val = val.iloc[-1]
            if not isinstance(val, (int, float)) or (isinstance(val, float) and pd.isna(val)):
                safe_preds[key] = fb
            else:
                safe_preds[key] = float(val)
        except Exception:
            safe_preds[key] = fb
    return safe_preds

# ====================== ENHANCED SIGNAL GENERATION ======================
def generate_institutional_signal(df: pd.DataFrame, asset: str) -> dict:
    from datetime import datetime
    signal = {
        "asset": asset,
        "action": "HOLD",
        "price": 0.0,
        "confidence": 0,
        "timestamp": datetime.utcnow().isoformat(),
        "tp_levels": [],
        "sl_level": None
    }
    try:
        # Check mode and asset type compatibility.
        current_mode = globals().get("trading_mode_override", None) or "auto"
        is_crypto = "/USD" in asset and asset not in ["EUR/USD", "GBP/USD"]
        if (current_mode == "forex" and is_crypto) or (current_mode == "crypto" and not is_crypto):
            return signal

        # Use live data regardless of DataFrame length.
        if df.empty:
            fallback_price = 1.0
            signal["price"] = fallback_price
            levels = compute_trade_levels(fallback_price, fallback_price * 0.01, "HOLD", asset)
            signal.update(levels)
            return signal

        latest = df.iloc[-1]
        price_val = float(latest['close'])
        # For live data, if ATR is missing, use a fallback calculation.
        atr_val = float(latest.get('ATR_14', price_val * 0.01))
        signal["price"] = price_val

        # Retrieve or set default values for indicators.
        adx_val = float(latest.get('ADX_14', 25))
        rsi_val = float(latest.get('RSI_14', 50))

        # Adjusted signal logic: lower ADX threshold and tighter RSI ranges for live data.
        if adx_val > 20:
            if rsi_val < 45:
                signal.update({"action": "BUY", "confidence": 80})
            elif rsi_val > 55:
                signal.update({"action": "SELL", "confidence": 80})
            else:
                signal.update({"action": "HOLD", "confidence": 50})
        else:
            signal.update({"action": "HOLD", "confidence": 50})

        # Compute stop loss and take profit levels based on live price and ATR.
        levels = compute_trade_levels(price_val, atr_val, signal["action"], asset)
        signal.update(levels)
        return signal

    except Exception as e:
        logger.error(f"Signal generation error: {str(e)}")
        return signal

def is_forex_trading_hours() -> bool:
    now = datetime.utcnow().time()
    return dt_time(8, 0) <= now <= dt_time(17, 0)

def get_asset_universe(force_refresh=False):
    global trading_mode_override
    
    forex_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]
    crypto_pairs = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD"]
    
    if trading_mode_override == "forex":
        logger.info("Active Mode: Forex - Returning major currency pairs")
        return forex_pairs
    elif trading_mode_override == "crypto":
        logger.info("Active Mode: Crypto - Returning top crypto pairs")
        return crypto_pairs
    
    # Auto mode - time-based selection
    now = datetime.utcnow().time()
    if dt_time(8, 0) <= now <= dt_time(17, 0):
        logger.info("Auto Mode: Forex hours active")
        return forex_pairs
    logger.info("Auto Mode: Crypto hours active")
    return crypto_pairs

async def start_signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        async with aiohttp.ClientSession() as session:
            await session.post("http://127.0.0.1:8080/start_trading")
        await update.message.reply_text(
            "🚀 [NekoAIBot] Signal generation initiated.\nYour AI system is now live.\n— NekoAIBot",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ [NekoAIBot] Error starting signals: {e}", parse_mode="Markdown")

async def stop_signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        async with aiohttp.ClientSession() as session:
            await session.post("http://127.0.0.1:8080/stop_trading")
        await update.message.reply_text(
            "🛑 [NekoAIBot] Signal generation halted.\n— NekoAIBot",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ [NekoAIBot] Error stopping signals: {e}", parse_mode="Markdown")

async def switch_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /mode command with immediate response and notification."""
    try:
        # Validate input
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "Usage: /mode <forex|crypto|auto>\nExample: /mode crypto",
                parse_mode="Markdown"
            )
            return

        mode = context.args[0].lower().strip()
        if mode not in ["forex", "crypto", "auto"]:
            await update.message.reply_text(
                "❌ Invalid mode. Please use 'forex', 'crypto', or 'auto'",
                parse_mode="Markdown"
            )
            return

        # Immediately update the mode
        global trading_mode_override
        trading_mode_override = None if mode == "auto" else mode
        logger.info(f"Telegram command switched mode to: {trading_mode_override or 'auto'}")

        # Force immediate asset universe refresh (this function can use force_refresh=True to re-read the trading hours)
        assets = get_asset_universe(force_refresh=True)
        logger.info(f"Updated asset universe: {assets}")

        # Immediately reply to the user to confirm the change
        reply_message = (
            f"✅ Mode switched to *{mode.upper()}*\n"
            f"Active assets: {', '.join(assets)}\n"
            "Changes take effect immediately."
        )
        await update.message.reply_text(reply_message, parse_mode="Markdown")
        
        # Synchronously notify the Telegram channel with the update
        if TELEGRAM_CHANNEL_ID:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    payload = {
                        "chat_id": TELEGRAM_CHANNEL_ID,
                        "text": f"🔔 Mode changed to {mode.upper()} via user command\nActive assets: {', '.join(assets)}",
                        "parse_mode": "HTML"
                    }
                    async with session.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json=payload
                    ) as response:
                        if response.status != 200:
                            logger.error(f"Channel notification failed with status: {response.status}")
                        else:
                            logger.info("Channel notification sent successfully.")
            except Exception as e:
                logger.error(f"Channel notification exception: {str(e)}")
    except Exception as e:
        logger.error(f"Telegram command error: {str(e)}")
        await update.message.reply_text("❌ Failed to switch mode. Please try again.", parse_mode="Markdown")

from fastapi import Request
import aiohttp

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    """
    Telegram webhook endpoint.
    This version first checks if the request body is empty before attempting to parse JSON.
    It then processes commands such as /start_signals, /stop_signals, and /mode <forex|crypto|auto>
    and sends a response via Telegram.
    """
    # Check for an empty request body to prevent JSONDecodeError.
    body = await request.body()
    if not body:
        logger.error("Empty request body in Telegram webhook")
        return {"status": "error", "detail": "Empty request body"}

    data = await request.json()
    message = data.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id", "")
    response_text = "Command not recognized. Available commands: /start_signals, /stop_signals, /mode <forex|crypto|auto>"

    if text.startswith("/start_signals"):
        global signal_generation_active
        signal_generation_active = True
        response_text = "🚀 [NekoAIBot] Signal generation started!"
    elif text.startswith("/stop_signals"):
        signal_generation_active = False
        response_text = "🛑 [NekoAIBot] Signal generation stopped!"
    elif text.startswith("/mode"):
        parts = text.split()
        if len(parts) >= 2:
            mode = parts[1].lower()
            global trading_mode_override
            trading_mode_override = None if mode == "auto" else mode
            response_text = f"🔔 [NekoAIBot] Mode set to {mode.upper()}"

    # Send the response message via Telegram.
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": chat_id, "text": response_text, "parse_mode": "HTML"}
            await session.post(url, json=payload)
    except Exception as e:
        logger.error(f"Telegram response error: {str(e)}")

    return {"status": "ok"}

def setup_telegram_handlers(application: TGApplication):
    """Register all command handlers"""
    application.add_handler(CommandHandler("mode", switch_mode_command))
    application.add_handler(CommandHandler("start_signals", start_signals_command))
    application.add_handler(CommandHandler("stop_signals", stop_signals_command))

async def start_telegram_listener():
    """Initialize and start the Telegram bot"""
    global telegram_app
    
    try:
        telegram_app = TGApplication.builder().token(TELEGRAM_BOT_TOKEN).build()
        setup_telegram_handlers(telegram_app)
        
        await telegram_app.initialize()
        await telegram_app.start()
        logger.info("Telegram listener started successfully")
        
        # Verify bot is reachable
        async with aiohttp.ClientSession() as session:
            resp = await session.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe")
            if resp.status != 200:
                logger.error("Telegram bot token validation failed")
    except Exception as e:
        logger.error(f"Telegram startup failed: {str(e)}")
        telegram_app = None

def run_telegram_listener():
    if not hasattr(app, 'telegram_running'):
        asyncio.run(start_telegram_listener())
        app.telegram_running = True
    else:
        logger.warning("Telegram listener already running")

telegram_app = None

async def start_telegram_listener():
    global telegram_app
    if telegram_app:
        return
    telegram_app = TGApplication.builder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start_signals", start_signals_command))
    telegram_app.add_handler(CommandHandler("stop_signals", stop_signals_command))
    telegram_app.add_handler(CommandHandler("mode", switch_mode_command))
    try:
        await telegram_app.initialize()
        await telegram_app.start()
        logger.info("Telegram listener started successfully")
    except Exception as e:
        logger.error(f"Telegram startup failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_telegram():
    global telegram_app
    if telegram_app:
        try:
            await telegram_app.stop()
        except RuntimeError as e:
            logger.warning(f"Telegram app stop error: {str(e)}")
        try:
            await telegram_app.shutdown()
        except Exception as e:
            logger.warning(f"Telegram app shutdown error: {str(e)}")
        telegram_app = None
        logger.info("Telegram listener stopped")
    else:
        logger.info("Telegram listener was not running")

async def send_telegram_message(dest: str, payload: dict):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with session.post(url, json=payload) as response:
                response_text = await response.text()
                if response.status != 200:
                    logger.error(f"Failed to send message to {dest}: {response_text}")
                else:
                    logger.debug(f"Message sent to {dest} successfully.")
    except Exception as e:
        logger.error(f"Error sending message to {dest}: {str(e)}")

import aiohttp

async def _send_telegram_with_retry(payload: dict, max_retries: int = 2) -> bool:
    """
    Sends a Telegram message with retries using HTML parse mode to avoid markdown entity issues.
    """
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                # It's recommended to format your message text for HTML if switching parse_mode.
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    logger.warning(f"Telegram attempt {attempt+1} failed: {await response.text()}")
        except Exception as e:
            logger.warning(f"Telegram send error (attempt {attempt+1}): {str(e)}")
        if attempt < max_retries:
            await asyncio.sleep(0.5)
    logger.error(f"Failed to send Telegram notification after {max_retries} retries")
    return False

async def send_pre_signal_message(signal: dict):
    signal_id = signal.get("id")
    if signal_id in sent_signal_ids:
        return
    sent_signal_ids.add(signal_id)
    message = (
        "⚠️ Risk Alert:\n"
        "Market conditions indicate heightened risk.\n"
        "Ensure proper risk management before proceeding.\n"
        "⏳ Preparing to drop a trade signal..."
    )
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_notification": False
    }
    await send_telegram_message(TELEGRAM_CHANNEL_ID, payload)

async def send_telegram_alert(signal: dict):
    """
    Sends a formatted trade signal alert via Telegram.
    Ensures duplicate signals are not re-sent and logs any errors.
    """
    signal_id = signal.get("id")
    if signal_id in sent_alert_ids:
        logger.debug(f"Signal {signal_id} already sent; skipping duplicate.")
        return
    sent_alert_ids.add(signal_id)
    try:
        price_str = f"{signal.get('price'):.5f}" if signal.get("price") is not None else "N/A"
        sl_str = f"{signal.get('sl_level'):.5f}" if signal.get("sl_level") is not None else "N/A"
        tp_list = signal.get("tp_levels", [])
        tp1 = f"{tp_list[0]:.5f}" if len(tp_list) > 0 else "N/A"
        tp2 = f"{tp_list[1]:.5f}" if len(tp_list) > 1 else "N/A"
        tp3 = f"{tp_list[2]:.5f}" if len(tp_list) > 2 else "N/A"
        message = (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃ 🚀 NekoAIBot Trade Signal 🚀 ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
            f"Signal ID: {signal_id}\n"
            f"Pair/Asset: {signal.get('asset', 'N/A')}\n"
            f"Predicted Change: {signal.get('predicted_change', 'N/A')}\n"
            f"News Sentiment: {signal.get('news_sentiment', 'N/A')}\n"
            f"AI Signal: {signal.get('action', 'N/A')}\n"
            f"Confidence: {signal.get('confidence', 0):.1f}%\n\n"
            f"Entry: {price_str}\n"
            f"Stop Loss: {sl_str}\n"
            "——————————————\n"
            "Take Profits:\n"
            f"  • TP1: {tp1}\n"
            f"  • TP2: {tp2}\n"
            f"  • TP3: {tp3}\n\n"
            "⚠️ Risk Warning: Trading involves significant risk.\n\n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃   NekoAIBot - Next-Gen Trading   ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        )
        # Send to both chat and channel
        destinations = [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID]
        for dest in destinations:
            if not dest:
                logger.debug("Destination not set; skipping.")
                continue
            payload = {
                "chat_id": dest,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            logger.debug(f"Sending Telegram alert to {dest}: {message}")
            await send_telegram_message(dest, payload)
    except Exception as e:
        logger.error(f"Alert system error: {str(e)}")
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            signal = await signal_queue.get()
            await websocket.send_json(signal)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            break

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

@app.get("/performance_metrics")
def performance_metrics():
    total = len(historical_trades)
    avg_conf = np.mean([t.get("confidence", 0) for t in historical_trades]) if total > 0 else 0
    return {
        "total_trades": total,
        "average_confidence": round(avg_conf, 2),
        "buy_signals": sum(1 for t in historical_trades if t.get("signal") == "BUY") if total > 0 else 0,
        "sell_signals": sum(1 for t in historical_trades if t.get("signal") == "SELL") if total > 0 else 0
    }

@app.get("/test_telegram", response_model=None)
async def test_telegram():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": "🔧 System Test Message", "parse_mode": "HTML"}
            async with session.post(url, json=payload) as response:
                return {"status": response.status, "response": await response.text()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test_mt5", response_model=None)
def test_mt5():
    try:
        if not initialize_mt5_admin():
            return {"status": "error", "message": "MT5 initialization failed"}
        symbol = "BTCUSD"
        info = mt5.symbol_info(symbol)
        if info is None:
            return {"status": "error", "message": f"Symbol {symbol} not found"}
        return {
            "status": "success",
            "symbol": symbol,
            "point_size": info.point,
            "digits": info.digits,
            "volume_min": info.volume_min
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        shutdown_mt5_admin()

async def refresh_market_data():
    """
    Continuously fetch market data for each asset only once per cycle.
    Cached data is stored in global_market_data and updated using an async lock.
    """
    while not app.shutdown_flag.is_set():
        assets = get_asset_universe()
        for asset in assets:
            # Determine asset type based on symbol naming conventions.
            asset_type = "crypto" if "/USD" in asset and asset not in ["EUR/USD", "GBP/USD"] else "forex"
            try:
                # Fetch data from APIs (using fetch_market_data).
                data = await fetch_market_data(asset, asset_type)
                if data is not None and not data.empty:
                    # Process and cache the indicators.
                    processed = compute_professional_indicators(data)
                    async with app.data_lock:
                        global_market_data[asset] = processed
                    logger.info(f"Updated market data for {asset}")
                else:
                    logger.warning(f"Failed to update market data for {asset}")
            except Exception as e:
                logger.error(f"Error fetching market data for {asset}: {str(e)}")
            await asyncio.sleep(1)  # Brief pause between asset updates.
        logger.info("Completed market scan cycle")
        await asyncio.sleep(300)  # Wait before starting the next full cycle.

class AsyncPositionTracker:
    def __init__(self):
        self.active_positions = {}
        self.lock = asyncio.Lock()
        
    async def add_position(self, result: mt5.OrderSendResult):
        async with self.lock:
            self.active_positions[result.order] = {
                'symbol': result.request.symbol,
                'volume': result.volume,
                'price': result.price,
                'timestamp': datetime.utcnow(),
                'status': 'open'
            }
            
    async def update_position(self, ticket: int, updates: dict):
        async with self.lock:
            if ticket in self.active_positions:
                self.active_positions[ticket].update(updates)

# Initialize global tracker
position_tracker = AsyncPositionTracker()

# =====================================================================
# AUTOMATIC LOT SIZE CALCULATION
# =====================================================================
# ================= Order Placement Function =================
def calculate_lot_size(symbol: str, entry_price: float, sl_level: float) -> float:
    """
    Calculates lot size based on account balance, a fixed risk per trade,
    and the distance between entry price and stop loss.
    """
    try:
        if not MT5_AVAILABLE:
            return 0.01
        account = mt5.account_info()
        if account is None:
            return 0.01
        balance = account.balance
        risk_amount = balance * RISK_PER_TRADE
        risk_per_lot = abs(entry_price - sl_level)
        if risk_per_lot == 0:
            return 0.01
        raw_volume = risk_amount / risk_per_lot
        symbol_info = mt5.symbol_info(symbol.replace("/", ""))
        if symbol_info:
            volume = max(raw_volume, symbol_info.volume_min)
            step = symbol_info.volume_step
            volume = round(volume / step) * step
            return volume
        return raw_volume
    except Exception as e:
        logger.error(f"Lot size calculation error: {str(e)}")
        return 0.01

import asyncio

# ====================== UPDATED ORDER EXECUTION ======================
async def place_mt5_order(
    symbol: str,
    order_type: str,
    volume: float,
    price: float,
    signal_id: str,
    user: Optional[User] = None,
    sl_level: Optional[float] = None,
    tp_levels: Optional[List[float]] = None
):
    """Enhanced MT5 order placement with fallbacks and validation"""
    max_attempts = 3
    attempt = 0
    
    try:
        if not is_market_open(symbol):
            logger.error(f"Market closed for {symbol}")
            return None

        while attempt < max_attempts:
            try:
                # Initialize connection with fallback
                if user:
                    with SessionLocal() as db:
                        if not initialize_user_mt5(user, db):
                            raise Exception("User MT5 init failed")
                elif not initialize_mt5_admin():
                    raise Exception("Global MT5 init failed")

                # Symbol validation and preparation
                mt5_symbol = symbol.replace("/", "")
                if not mt5.symbol_select(mt5_symbol, True):
                    logger.error(f"Symbol {mt5_symbol} not available")
                    return None

                symbol_info = mt5.symbol_info(mt5_symbol)
                if not symbol_info or not getattr(symbol_info, 'trade_allowed', False):
                    logger.error(f"Trading disabled for {symbol}")
                    return None

                # Get critical parameters with fallbacks
                digits = getattr(symbol_info, 'digits', 5)
                point = getattr(symbol_info, 'point', 0.00001)
                min_stop_points = getattr(symbol_info, 'trade_stops_level', 10)
                min_stop_distance = min_stop_points * point * 1.2  # 20% buffer
                min_volume = getattr(symbol_info, 'volume_min', 0.01)

                # Price validation
                tick = mt5.symbol_info_tick(mt5_symbol)
                if not tick:
                    logger.error("No tick data available")
                    raise Exception("No market data")

                current_price = tick.ask if order_type.upper() == "BUY" else tick.bid
                price_diff = abs(current_price - price) / price
                if price_diff > 0.01:
                    logger.warning(f"Large price deviation: {price_diff:.2%}, using current price")
                    price = current_price

                # Calculate levels if not provided
                if not sl_level or not tp_levels:
                    levels = compute_trade_levels(
                        current_price, 
                        getattr(symbol_info, 'atr', current_price * 0.01),
                        order_type, 
                        symbol
                    )
                    sl_level = levels["sl_level"]
                    tp_levels = levels["tp_levels"]

                # Validate stop levels
                if order_type == "BUY":
                    stop_diff = current_price - sl_level
                else:
                    stop_diff = sl_level - current_price

                if stop_diff < min_stop_distance:
                    logger.warning(f"Stop too close: {stop_diff:.5f} < {min_stop_distance:.5f}")
                    sl_level = round(current_price - (min_stop_distance * 1.5), digits) if order_type == "BUY" \
                              else round(current_price + (min_stop_distance * 1.5), digits)

                # Calculate proper lot size
                volume = max(
                    calculate_lot_size(symbol, current_price, sl_level),
                    min_volume
                )
                volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step

                # Prepare order request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": mt5_symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": current_price,
                    "sl": round(sl_level, digits),
                    "tp": round(tp_levels[0], digits) if tp_levels else None,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": f"NEKO_{signal_id}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }

                # Execute order with validation
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Order executed: {result.order}")
                    return result
                
                logger.warning(f"Order failed: {result.comment}")
                attempt += 1
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Order attempt {attempt+1} failed: {str(e)}")
                attempt += 1
                await asyncio.sleep(2)

        logger.error(f"Failed all order attempts for {symbol}")
        return None

    except Exception as e:
        logger.error(f"Critical order error: {str(e)}")
        return None
    finally:
        if MT5_AVAILABLE:
            mt5.shutdown()

# Dictionary to track active signals per asset (to allow one signal at a time)
active_signals = {}

import asyncio
import logging

# New function to process signals and trigger trade execution
async def process_trade_signal(signal: dict):
    """Processes trade signals with adjusted volatility threshold"""
    try:
        logger.info(f"Processing {signal['asset']} {signal['action']} signal")
        
        # Adjusted volatility check (0.1% instead of 0.15%)
        atr = signal["indicators"]["ATR_14"]
        if atr < signal["price"] * 0.001:
            logger.warning(f"Ignoring {signal['asset']} - low volatility (ATR: {atr:.5f})")
            return

        # Generate unique signal ID
        async with signal_counter_lock:
            global signal_counter
            signal_id = f"NEKO_{signal_counter}"
            signal_counter += 1
        signal["id"] = signal_id

        # Pre-trade alert
        await send_pre_signal_message(signal)
        logger.info(f"Signal {signal_id} confirmed, executing in 30s")
        await asyncio.sleep(30)

        # Execute trade
        levels = compute_trade_levels(
            signal["price"],
            signal["indicators"]["ATR_14"],
            signal["action"],
            signal["asset"]
        )
        
        result = await place_mt5_order(
            symbol=signal["asset"],
            order_type=signal["action"],
            volume=0.1,
            price=signal["price"],
            signal_id=signal_id,
            sl_level=levels["sl_level"],
            tp_levels=levels["tp_levels"]
        )

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            await send_telegram_alert(signal)
            logger.info(f"Trade executed: {signal_id}")
            # Track position
            await position_tracker.add_position(result)
        else:
            logger.error(f"Execution failed for {signal_id}")

    except Exception as e:
        logger.error(f"Trade processing failed: {str(e)}")

def adjust_stop_loss(position, signal):
    """
    Adjusts the stop loss for a position based on trailing stop parameters.
    Validates that necessary values exist before computing the new stop loss.
    
    Parameters:
        position: The trading position object (must have attributes like price_current, price_open, ticket, tp).
        signal: A dictionary containing the action ("BUY" or "SELL") and trailing_stop parameters.
                Expected format: 
                  {
                    "action": "BUY" or "SELL",
                    "trailing_stop": {"activation": <value>, "step": <value>}
                  }
    """
    try:
        trailing = signal.get("trailing_stop")
        if trailing is None:
            logger.error("Trailing stop parameters missing. Aborting stop loss adjustment.")
            return

        activation = trailing.get("activation")
        step = trailing.get("step")
        if activation is None or step is None:
            logger.error("Trailing stop activation or step missing. Aborting stop loss adjustment.")
            return

        current_price = position.price_current
        if current_price is None or position.price_open is None:
            logger.error("Missing current or open price on position.")
            return

        if signal["action"] == "BUY":
            if current_price > (position.price_open + step * activation):
                new_sl = current_price - step
                if new_sl < current_price:
                    # Update order: replace with your actual order API call
                    mt5.order_send({
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp
                    })
        else:
            if current_price < (position.price_open - step * activation):
                new_sl = current_price + step
                if new_sl > current_price:
                    mt5.order_send({
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp
                    })
    except Exception as e:
        logger.error(f"Profit optimization error for position {position.ticket}: {str(e)}")

def position_exists(ticket):
    """
    Check if the position with the given ticket exists in MT5.
    This function queries current active positions and returns True if found.
    Replace or extend this logic if you maintain a separate tracking system.
    """
    positions = mt5.positions_get()
    if positions:
        return any(pos.ticket == ticket for pos in positions)
    return False


def profit_optimization():
    """
    Optimize profit for active positions.
    For each position, verify it exists before attempting any profit adjustments.
    If a position is not found, log a warning and skip it.
    Extend the inner loop with your profit optimization logic as needed.
    """
    try:
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            logger.warning("No positions found during profit optimization.")
            return

        for pos in positions:
            # Verify that the position exists in MT5 (or your internal system)
            if not position_exists(pos.ticket):
                logger.warning(f"Position {pos.ticket} not found. Exiting profit optimizer for this position.")
                continue

            # Place your profit optimization logic here.
            # For example, update stop loss/take profit if conditions are met:
            try:
                # Get current tick for the symbol
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    logger.warning(f"No tick data for symbol {pos.symbol}. Skipping optimization.")
                    continue

                # Example logic: Adjust SL and TP if the market moved significantly
                current_price = tick.ask if pos.type == mt5.ORDER_TYPE_BUY else tick.bid
                # This is a dummy example: update SL to a value based on current price
                new_sl = current_price * 0.99 if pos.type == mt5.ORDER_TYPE_BUY else current_price * 1.01
                new_tp = current_price * 1.02 if pos.type == mt5.ORDER_TYPE_BUY else current_price * 0.98

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "symbol": pos.symbol,
                    "sl": new_sl,
                    "tp": new_tp,
                    "magic": getattr(pos, "magic", 234000),
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Profit optimization update failed for position {pos.ticket}: {result.comment}")
                else:
                    logger.info(f"Profit optimization updated for position {pos.ticket}")
            except Exception as opt_e:
                logger.error(f"Error optimizing position {pos.ticket}: {str(opt_e)}")

    except Exception as e:
        logger.error(f"Profit optimization error: {str(e)}")

async def calculate_volatility_adjusted_sizes(assets):
    """Calculate position sizes based on ATR (volatility)"""
    sizes = []
    for asset in assets:
        try:
            if asset in global_market_data and not global_market_data[asset].empty:
                if 'ATR_14' not in global_market_data[asset].columns:
                    global_market_data[asset] = compute_professional_indicators(global_market_data[asset])
                atr = global_market_data[asset]['ATR_14'].iloc[-1]
                price = global_market_data[asset]['close'].iloc[-1]
                size = max(0.01, min(1.0, (RISK_PER_TRADE * 10000) / ((atr / price) * 10000)))
                sizes.append(round(size, 2))
            else:
                sizes.append(0.1)
        except Exception as e:
            logger.error(f"Size calculation error for {asset}: {str(e)}")
            sizes.append(0.1)
    return sizes

# ====================== IMPROVED TRADING WORKER ======================
async def premium_trading_worker():
    """
    Active trading loop that uses cached data to generate trading signals.
    Signals are generated only if the cached data is valid and sufficient.
    """
    while not app.shutdown_flag.is_set():
        assets = get_asset_universe()
        logger.info(f"Analyzing {len(assets)} assets")
        for asset in assets:
            asset_type = "crypto" if "/USD" in asset else "forex"
            try:
                # Retrieve cached data for the asset.
                async with app.data_lock:
                    cached_data = global_market_data.get(asset)
                if cached_data is None or cached_data.empty or len(cached_data) < 30:
                    logger.debug(f"Insufficient cached data for {asset}, skipping...")
                    continue
                # Generate trading signal using the cached indicators.
                signal = generate_institutional_signal(cached_data, asset)
                if signal["action"] != "HOLD":
                    logger.info(f"Processing {signal['action']} signal for {asset}")
                    # Enrich the signal with news sentiment data.
                    news = await fetch_market_news(asset_type)
                    signal["news_sentiment"] = analyze_sentiment(news)
                    # Enqueue the signal for processing.
                    await signal_queue.put(signal)
                    asyncio.create_task(process_trade_signal(signal))
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error processing asset {asset}: {str(e)}")
                await asyncio.sleep(1)
        logger.info("Completed market scan cycle")
        await asyncio.sleep(10)

# ====================== ENHANCED LOGGING ======================
async def process_trade_signal(signal: dict):
    """
    Processes a trade signal by generating a unique signal ID,
    waiting for market confirmation, and then placing a trade order.
    """
    try:
        logger.info(f"Processing signal for {signal['asset']} with action {signal['action']}")
        async with signal_counter_lock:
            global signal_counter
            signal_id = f"NekoAITrader_{signal_counter}"
            signal_counter += 1
        signal["id"] = signal_id
        logger.info(f"Executing trade {signal_id} after waiting for market confirmation...")
        # Optional wait to allow market conditions to stabilize.
        await asyncio.sleep(30)
        result = await place_mt5_order(
            symbol=signal["asset"],
            order_type=signal["action"],
            volume=0.1,
            price=signal["price"],
            signal_id=signal_id,
            sl_level=signal["sl_level"],
            tp_levels=signal["tp_levels"]
        )
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trade executed successfully: {signal_id}")
            await send_telegram_alert(signal)
        else:
            logger.error(f"Trade execution failed for {signal_id}: {result.comment if result else 'No result'}")
    except Exception as e:
        logger.error(f"Error processing trade signal {signal.get('id', 'unknown')}: {str(e)}")

@app.on_event("shutdown")
async def shutdown():
    shutdown_flag.set()
    # Cancel all tasks except the current one.
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    logger.info("All tasks have been cancelled. Shutting down gracefully.")

async def process_asset(asset: str, position_size: float, processing_lock: asyncio.Lock, signal_counter_lock: asyncio.Lock):
    """Process a single asset: generate signal, send pre-alert with risk warnings, wait 30 seconds, execute trade, and send main signal alert."""
    global signal_counter
    try:
        async with processing_lock:
            now = datetime.utcnow()
            # Prevent duplicate signals within a 5-minute window.
            if asset in active_signals and (now - active_signals[asset]).total_seconds() < 300:
                return

            logger.info(f"Processing {asset} with position size {position_size}...")
            asset_type = "crypto" if "/USD" in asset and asset not in ["EUR/USD", "GBP/USD"] else "forex"
            if twelvedata_limiter.remaining <= 1:
                logger.warning("Approaching TwelveData limit, adding safety delay")
                await asyncio.sleep(30)
            data = await fetch_market_data(asset, asset_type)
            if data.empty:
                logger.warning(f"No data for {asset}, skipping")
                return

            data = compute_professional_indicators(data)
            predictions = generate_institutional_predictions(data)
            # Calculate the trading signal using both AI and rule-based logic.
            signal = generate_institutional_signal(data, predictions, asset)
            
            # Calculate news sentiment in real time.
            news_articles = await fetch_market_news(asset)
            signal["news_sentiment"] = analyze_sentiment(news_articles)  # Now dynamically set instead of "NEUTRAL"

            # Recalculate confidence level using the latest price/ATR values.
            price = data["close"].iloc[-1]
            atr = data["ATR_14"].iloc[-1] if "ATR_14" in data.columns else price * 0.01
            price_diff = (predictions.get("lstm", 0) * price) - price  # Example: using lstm prediction scaled by price
            risk_ratio = abs(price_diff) / atr if atr > 0 else 0
            signal["confidence"] = min(100, risk_ratio * 100 * (data["ADX_14"].iloc[-1] / 100 + 0.5))

            async with signal_counter_lock:
                signal_id = str(signal_counter)
                signal_counter += 1
            signal["id"] = signal_id
            signal["size"] = position_size  # Trade volume

            # --- Send Pre-Signal Alert with Risk Warning ---
            await send_pre_signal_message(signal)
            logger.info(f"Pre-signal alert sent for signal {signal_id}. Waiting 30 seconds before executing trade...")
            await asyncio.sleep(30)

            # --- Execute the Order ---
            result = await place_mt5_order(
                symbol=asset,
                order_type=signal.get("action"),
                volume=signal["size"],
                price=signal.get("price"),
                signal_id=signal_id,
                sl_level=signal.get("sl_level"),
                tp_levels=signal.get("tp_levels")
            )

            if result:
                logger.info(f"Trade executed for {asset} with result: {result}")
            else:
                logger.error(f"Trade execution failed for {asset}")

            # --- Send Main Signal Alert ---
            await send_telegram_alert(signal)
    except Exception as e:
        logger.error(f"Error processing asset {asset}: {str(e)}")

async def execute_trades(signal: dict):
    """Enhanced order execution with trailing stops (original + new logic)"""
    executions = []
    
    try:
        if MT5_AVAILABLE and signal["action"] in ["BUY", "SELL"] and signal["confidence"] >= 70:
            mt5_symbol = signal["asset"].replace("/", "")
            admin_result = await place_mt5_order(
                symbol=signal["asset"],
                order_type=signal["action"],
                volume=signal.get("size", 0.1),
                price=signal["price"],
                signal_id=signal["id"],
                user=None,
                sl_level=signal.get("sl_level"),
                tp_levels=signal.get("tp_levels", [])
            )
            executions.append(("Admin", admin_result))
            if admin_result and admin_result.retcode == mt5.TRADE_RETCODE_DONE:
                trailing_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": admin_result.order,
                    "sl": signal["sl_level"],
                    "tp": signal["tp_levels"][0],
                    "deviation": 5,
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                mt5.order_send(trailing_request)
                threading.Thread(target=optimize_trade_profit, args=(admin_result.order, signal)).start()
            with SessionLocal() as db:
                users = db.query(User).options(joinedload(User.broker_credentials)).all()
                for user in users:
                    if user.broker_credentials:
                        user_result = await place_mt5_order(
                            symbol=signal["asset"],
                            order_type=signal["action"],
                            volume=signal.get("size", 0.1),
                            price=signal["price"],
                            signal_id=signal["id"],
                            user=user,
                            sl_level=signal.get("sl_level"),
                            tp_levels=signal.get("tp_levels", [])
                        )
                        executions.append((f"User {user.username}", user_result))
        asyncio.create_task(monitor_executed_trades(executions, signal))
        return executions
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        return []

def optimize_trade_profit(position_id: int, signal: dict):
    """
    Continuously monitor an open position to adjust the stop loss (profit optimization).
    If the position data is missing, exits gracefully.
    """
    try:
        while True:
            positions = mt5.positions_get(ticket=position_id)
            if not positions or len(positions) == 0:
                logger.warning(f"Position {position_id} not found. Exiting profit optimizer.")
                break
            position = positions[0]
            current_price = position.price_current
            activation = signal.get("trailing_stop", {}).get("activation")
            step = signal.get("trailing_stop", {}).get("step")
            if activation is None or step is None:
                logger.error("Trailing stop parameters missing. Exiting profit optimizer.")
                break

            if signal["action"] == "BUY":
                if current_price > (position.price_open + step * activation):
                    new_sl = current_price - step
                    # Validate new stop: must be below current price
                    if new_sl < current_price:
                        mt5.order_send({
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_id,
                            "sl": new_sl,
                            "tp": position.tp
                        })
            else:
                if current_price < (position.price_open - step * activation):
                    new_sl = current_price + step
                    if new_sl > current_price:
                        mt5.order_send({
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_id,
                            "sl": new_sl,
                            "tp": position.tp
                        })
            time.sleep(15)
    except Exception as e:
        logger.error(f"Profit optimization error for position {position_id}: {str(e)}")

async def monitor_open_trades():
    """
    New real-time trade monitoring routine.
    Checks open positions, verifies if stop levels need adjustment,
    and sends update alerts via Telegram.
    """
    while not shutdown_flag.is_set():
        try:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    # Example: if profit exceeds a threshold, update stop loss and send an alert.
                    profit_pct = (pos.price_current - pos.price_open) / pos.price_open * 100
                    # If profit > 2%, adjust stop loss to lock in profit.
                    if pos.type == mt5.ORDER_TYPE_BUY and profit_pct > 2:
                        new_sl = pos.price_current - 0.5 * (pos.price_current - pos.price_open)
                        # Update order if new stop is higher than current stop
                        if new_sl > pos.sl:
                            update_result = mt5.order_send({
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": new_sl,
                                "tp": pos.tp
                            })
                            if update_result.retcode == mt5.TRADE_RETCODE_DONE:
                                await send_telegram_alert({
                                    "update": f"Updated BUY position {pos.ticket}: New SL {new_sl}"
                                })
                    elif pos.type == mt5.ORDER_TYPE_SELL and profit_pct < -2:
                        new_sl = pos.price_current + 0.5 * (pos.price_open - pos.price_current)
                        if new_sl < pos.sl:
                            update_result = mt5.order_send({
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": new_sl,
                                "tp": pos.tp
                            })
                            if update_result.retcode == mt5.TRADE_RETCODE_DONE:
                                await send_telegram_alert({
                                    "update": f"Updated SELL position {pos.ticket}: New SL {new_sl}"
                                })
            await asyncio.sleep(30)  # adjust the frequency as needed
        except Exception as e:
            logger.error(f"Trade monitoring error: {str(e)}")
            await asyncio.sleep(30)

async def monitor_executed_trades(executions: list, original_signal: dict):
    asset = original_signal["asset"]
    update_interval = 60
    while True:
        status_updates = []
        for account, result in executions:
            if not result:
                continue
            try:
                if MT5_AVAILABLE:
                    positions = mt5.positions_get(symbol=asset.replace("/",""))
                    for pos in positions:
                        if pos.comment == f"NEKO_{original_signal['id']}":
                            current_tick = mt5.symbol_info_tick(pos.symbol)
                            if not current_tick:
                                continue
                            current_price = current_tick.ask if pos.type == 0 else current_tick.bid
                            pnl = (current_price - pos.price_open) * pos.volume * (1 if pos.type == 0 else -1)
                            status_updates.append(f"{account} Position: {pos.volume} lots | PnL: ${pnl:.2f}")
            except Exception as e:
                logger.error(f"Monitoring error for {account}: {str(e)}")
        if status_updates:
            message = (
                f"🔍 Trade Monitoring: {asset}\n"
                f"Signal ID: {original_signal['id']}\n"
                "——————————————\n" +
                "\n".join(status_updates) +
                "\n\nNext update in 1 minute..."
            )
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={"chat_id": TELEGRAM_CHANNEL_ID, "text": message, "parse_mode": "HTML"}
                    )
            except Exception as e:
                logger.error(f"Monitoring update failed: {str(e)}")
        await asyncio.sleep(update_interval)

import os
import threading
import asyncio
import logging
import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from datetime import datetime
import pandas as pd

# Global dictionary for models
models = {}

# Create a synchronous lock for model loading
model_sync_lock = threading.Lock()

# Global shutdown flag for the scheduler (must be set on startup/shutdown)
shutdown_flag = asyncio.Event()

# Logger setup (customize as needed)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_all_models():
    """
    Loads all trading models from disk.
    Note: Your RL models (dqn and ppo) will error out if their input dimensions don't match.
    Either update your environment to produce 90 features or retrain the models with 30 features.
    """
    from trading_env import TradingEnv  # Ensure TradingEnv is available
    env = TradingEnv()

    # Custom objects for RL models.
    # (This configuration builds a model that expects 30 features.)
    custom_objects = {
        "lr_schedule": lambda _: 0.001,
        "exploration_schedule": lambda _: 0.1,
        "clip_range": lambda _: 0.2,
        "n_envs": 1,
        "device": "cpu",
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "_vec_normalize_env": DummyVecEnv([lambda: TradingEnv()]),
        "policy_kwargs": {"net_arch": [64, 64]},
        "optimizer_class": RMSpropTFLike
    }

    model_paths = {
        # Deep Learning Models (Keras)
        "lstm": "models/lstm_model.keras",
        "gru": "models/gru_model.keras",
        "cnn": "models/cnn_model.keras",
        "transformer": "models/transformer_model.keras",
        # Machine Learning Models
        "xgb": "models/xgboost_model.json",
        "lightgbm": "models/lightgbm_model.txt",
        "catboost": "models/catboost_model.cbm",
        "stacking": "models/stacking_model.pkl",
        "gaussian": "models/gaussian_process_model.pkl",
        # Time Series Model
        "prophet": "models/prophet_model.pkl",
        # NLP / Anomaly Models
        "sentiment": "models/sentiment_model.keras",
        "sentiment_tokenizer": "models/sentiment_tokenizer.pkl",
        "anomaly": "models/anomaly_model.keras",
        # Preprocessing and Regression
        "scaler": "models/price_scaler.pkl",
        "svr": "models/svr_model.pkl",
        # RL Models
        "dqn": ("models/dqn_model.zip", custom_objects),
        "ppo": ("models/ppo_model.zip", custom_objects),
    }

    with model_sync_lock:
        for name, path_info in model_paths.items():
            if isinstance(path_info, tuple):
                path, custom_obj = path_info
            else:
                path, custom_obj = path_info, {}
            try:
                if not os.path.exists(path):
                    logger.warning(f"Model file missing: {path}")
                    continue
                if name in ["lstm", "gru", "cnn", "transformer", "sentiment", "anomaly"]:
                    models[name] = tf.keras.models.load_model(path)
                elif name == "xgb":
                    models[name] = xgb.XGBRegressor()
                    models[name].load_model(path)
                elif name == "lightgbm":
                    models[name] = lgb.Booster(model_file=path)
                elif name == "catboost":
                    models[name] = CatBoostRegressor()
                    models[name].load_model(path)
                elif name in ["stacking", "gaussian", "prophet", "sentiment_tokenizer", "scaler", "svr"]:
                    with open(path, "rb") as f:
                        models[name] = joblib.load(f)
                elif name in ["dqn", "ppo"]:
                    if name == "dqn":
                        models[name] = DQN.load(path, custom_objects=custom_obj)
                    elif name == "ppo":
                        models[name] = PPO.load(path, custom_objects=custom_obj)
                else:
                    logger.warning(f"Unknown model type for {name}")
                logger.info(f"Successfully loaded {name}")
            except Exception as e:
                logger.error(f"Error loading {name}: {str(e)}")
                if "rl" in name:
                    logger.error("Note: RL models may require retraining with the current feature set.")
    return models

import gym
from stable_baselines3.common.monitor import Monitor

###############################################################################
# Updated RetrainRLSystem Class
###############################################################################
# ================= RetrainRLSystem Class =================
import time
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import yfinance as yf
import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)

# ---------- RetrainRLSystem CLASS ----------
class RetrainRLSystem:
    def __init__(self, app, retraining_interval: int = 3600, monitoring_interval: int = 300):
        """
        Initializes the retraining system.
        
        :param app: The FastAPI app instance (or similar) where market data, validation data, and models are stored.
        :param retraining_interval: Interval in seconds between retraining cycles.
        :param monitoring_interval: Interval in seconds between performance monitoring cycles.
        """
        self.app = app
        self.retraining_interval = retraining_interval
        self.monitoring_interval = monitoring_interval

    def preprocess_training_data(self) -> pd.DataFrame:
        """
        Collects and preprocesses the training data for the RL models.
        Replace this with your actual preprocessing logic.
        
        :return: A preprocessed pandas DataFrame with training data.
        """
        data = self.app.get_recent_data()
        if data is None or data.empty:
            logger.error("Preprocessing error: training data is empty.")
            return pd.DataFrame()
        
        if 'close' not in data.columns:
            logger.error("Preprocessing error: 'close' column missing from training data.")
            return pd.DataFrame()

        # Example preprocessing: drop rows with missing values
        data = data.dropna()
        return data

    def retrain_models(self, data: pd.DataFrame):
        """
        Retrains each RL model using the provided training data.
        Replace the placeholder retraining logic with your actual model retraining code.
        
        :param data: Preprocessed training data.
        """
        rl_models = self.app.models.get("rl", {})
        if not rl_models:
            logger.warning("No RL models available for retraining.")
            return
        
        for model_name, model in rl_models.items():
            try:
                # Replace model.retrain(data) with your actual retraining method.
                model.retrain(data)
                logger.info(f"Retrained model '{model_name}' successfully.")
            except Exception as e:
                logger.error(f"Failed to retrain model '{model_name}': {str(e)}")

    async def continuous_retraining_scheduler(self):
        """
        Continuously runs the retraining cycle on a fixed interval.
        Checks that the preprocessed training data is non-empty before retraining.
        """
        while True:
            try:
                data = self.preprocess_training_data()
                if data.empty:
                    logger.error("No training data available after preprocessing; skipping retraining cycle.")
                else:
                    self.retrain_models(data)
                    logger.info("Retraining cycle completed.")
            except Exception as e:
                logger.error(f"Retraining error: {str(e)}")
            await asyncio.sleep(self.retraining_interval)

    async def monitor_performance(self):
        """
        Monitors the performance of the RL models using a validation dataset.
        This method retrieves validation data from the app (via a get_validation_data method) and
        then evaluates each model using its evaluate_performance method.
        
        Replace or adjust the validation data retrieval and performance evaluation logic as needed.
        """
        while True:
            try:
                # Retrieve validation data from your app. Ensure that get_validation_data exists.
                if hasattr(self.app, "get_validation_data"):
                    val_data = self.app.get_validation_data()
                else:
                    logger.error("No validation data retrieval method found in app.")
                    val_data = pd.DataFrame()

                if val_data is None or val_data.empty:
                    logger.warning("No validation data available for performance monitoring.")
                else:
                    rl_models = self.app.models.get("rl", {})
                    for model_name, model in rl_models.items():
                        try:
                            # Replace evaluate_performance with your actual performance evaluation method.
                            performance_metric = model.evaluate_performance(val_data)
                            logger.info(f"Performance of '{model_name}': {performance_metric}")
                        except Exception as e:
                            logger.error(f"Error evaluating performance of '{model_name}': {str(e)}")
            except Exception as e:
                logger.error(f"Performance monitoring error: {str(e)}")
            await asyncio.sleep(self.monitoring_interval)

# In startup_event()
retrain_system = RetrainRLSystem(app)
app.retrain_rl_system = retrain_system  # Attach to app instance
app.retrain_rl_system.market_data_store = app.market_data_store  # Grant access
asyncio.create_task(retrain_system.continuous_retraining_scheduler())
asyncio.create_task(retrain_system.monitor_performance())

@app.post("/set_overrides")
async def set_overrides(force_trade: bool = False, force_market: bool = False):
    globals()["FORCE_TRADE_OVERRIDE"] = force_trade
    globals()["FORCE_MARKET_OPEN_OVERRIDE"] = force_market
    logger.info(f"Overrides set: FORCE_TRADE_OVERRIDE={force_trade}, FORCE_MARKET_OPEN_OVERRIDE={force_market}")
    return {"FORCE_TRADE_OVERRIDE": force_trade, "FORCE_MARKET_OPEN_OVERRIDE": force_market}

@app.on_event("shutdown")
async def shutdown_event():
    global signal_generation_active
    signal_generation_active = False
    shutdown_flag.set()
    shutdown_mt5_admin()
    logger.info("NekoAIBot Shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=True)