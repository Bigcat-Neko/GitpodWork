# main.py - NEKO AIBot Premium Trader (Multi‑User & Dynamic Risk Management Version)
import sys
import random
import asyncio
import os
import time
import threading
import numpy as np
import pandas as pd
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
from passlib.context import CryptContext
from jose import JWTError, jwt
from diskcache import Cache
from dotenv import load_dotenv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import DummyVecEnv
# Try to import MetaTrader5; if unavailable, disable trading functions.
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 package not available. Trading functionality will be disabled.")
import nest_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from cryptography.fernet import Fernet

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()
encryption_key = os.getenv("ENCRYPTION_KEY")
if not encryption_key:
    encryption_key = Fernet.generate_key().decode()
    print("Please set ENCRYPTION_KEY in your environment to:", encryption_key)
cipher_suite = Fernet(encryption_key)

# ---------------- FastAPI App Initialization ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
nest_asyncio.apply()
cache = Cache("./.api_cache")

# ---------------- Environment Variables ----------------
NEWSAPI_KEY = os.getenv("NEWS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
try:
    MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
except Exception:
    MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
TWELVEDATA_DAILY_LIMIT = int(os.getenv("TWELVEDATA_DAILY_LIMIT", "800"))
ALPHAVANTAGE_DAILY_LIMIT = int(os.getenv("ALPHAVANTAGE_DAILY_LIMIT", "500"))
# SMTP & Email settings – obtain these from your email service provider.
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
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

# ---------------- Global Variables ----------------
# Global variable for incremental signal IDs
signal_counter = 1
signal_counter_lock = threading.Lock()  # To ensure thread safety if needed

trading_mode_override = None
models = {}
model_lock = threading.Lock()
signal_generation_active = True
api_calls_today = 0
shutdown_flag = asyncio.Event()
signal_queue = asyncio.Queue()  # For WebSocket notifications
sent_signal_ids = set()         # To avoid duplicate pre-signal messages
sent_alert_ids = set()          # To avoid duplicate main signal messages
historical_trades = []          # For trade history if needed

# ---------------- Database Setup ----------------
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
    tp_levels = Column(String)  # JSON stored as string
    sl_level = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    indicators = Column(String)  # JSON string
    predictions = Column(String)  # JSON string
    news_sentiment = Column(String)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    registered_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Integer, default=0)  # 1 for admin, 0 for regular
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

# ---------------- Security & Authentication ----------------
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

# ---------------- Email Sending ----------------
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

# ---------------- User Endpoints ----------------
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

# ---------------- Multi‑User Trading Task Management ----------------
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

@app.post("/switch_mode", response_model=None)
async def switch_mode(mode: str, current_user: User = Depends(get_current_user)):
    global trading_mode_override
    if mode.lower() not in ["forex", "crypto", "auto"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use forex, crypto, or auto.")
    trading_mode_override = None if mode.lower() == "auto" else mode.lower()
    # Notify Telegram about mode switch
    async with aiohttp.ClientSession() as session:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHANNEL_ID, "text": f"🔔 [NekoAIBot] Mode switched to {mode.upper()}", "parse_mode": "Markdown"}
        await session.post(url, json=payload)
    return {"message": f"Trading mode switched to {mode.upper()}"}

# ---------------- Database Migration Note ----------------
# Make sure to run your Alembic migrations so your database schema is up-to-date.

# ---------------- Track API Usage ----------------
def track_api_usage(source: str):
    global api_calls_today
    api_calls_today += 1
    logger.info(f"API call to {source}. Total today: {api_calls_today}/{TWELVEDATA_DAILY_LIMIT}")
    if api_calls_today >= TWELVEDATA_DAILY_LIMIT * 0.9:
        logger.warning(f"Approaching daily API limit ({TWELVEDATA_DAILY_LIMIT})")

# ---------------- User-Specific MT5 Session Initialization ----------------
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

# ---------------- Global Trading Functions ----------------
def initialize_mt5_admin(max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        try:
            login_val = int(MT5_LOGIN)
        except Exception:
            login_val = MT5_LOGIN
        if MT5_AVAILABLE and mt5.initialize(login=login_val, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.info("Global MT5 (admin) initialized successfully.")
            return True
        else:
            logger.error(f"Global MT5 initialization failed: {mt5.last_error() if MT5_AVAILABLE else 'MT5 not available'}")
            attempts += 1
            time.sleep(5)
    return False

def shutdown_mt5_admin():
    if MT5_AVAILABLE:
        mt5.shutdown()

async def place_mt5_order(symbol: str, order_type: str, volume: float, price: float, 
                         signal_id: str, user: Optional[User] = None,
                         sl_level: Optional[float] = None, 
                         tp_levels: Optional[List[float]] = None):
    """Enhanced order placement with proper signal handling"""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Initialize connection
            if user:
                with SessionLocal() as db:
                    if not initialize_user_mt5(user, db):
                        logger.error(f"User MT5 init failed for {user.username}")
                        return None
            elif not initialize_mt5_admin():
                logger.error("Global MT5 init failed")
                return None
                
            # Prepare order request
            mt5_symbol = symbol.replace("/", "")
            if not mt5.symbol_select(mt5_symbol, True):
                logger.error(f"Symbol {mt5_symbol} not available")
                return None
                
            # Get current price
            tick = mt5.symbol_info_tick(mt5_symbol)
            if not tick:
                logger.error("No tick data available")
                raise Exception("No tick data")
                
            current_price = tick.ask if order_type == "BUY" else tick.bid
            price_diff = abs(current_price - price)/price
            if price_diff > 0.01:  # 1% price deviation
                logger.warning(f"Price deviation too large: {price_diff:.2%}")
                return None
                
            # Prepare order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "deviation": 10,
                "magic": 234000,
                "comment": f"NEKO_{signal_id}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Add SL/TP if provided
            if sl_level is not None:
                request["sl"] = sl_level
            if tp_levels is not None and len(tp_levels) > 0:
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

class CircuitBreaker:
    def __init__(self, max_failures=5, reset_timeout=300):
        self.failures = 0
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.last_failure = None
        
    async def check(self):
        if self.failures >= self.max_failures:
            if (datetime.now() - self.last_failure).seconds < self.reset_timeout:
                raise Exception("Circuit breaker tripped")
            self.failures = 0
        return True
        
    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()

# Initialize in your app
app.state.circuit_breaker = CircuitBreaker()

# Usage in data fetching
async def fetch_market_data(symbol: str, asset_type: str):
    try:
        await app.state.circuit_breaker.check()
        # ... existing code ...
    except Exception as e:
        app.state.circuit_breaker.record_failure()
        raise

# ---------------- Global Trade Monitoring (Enhanced) ----------------
async def execute_trades(signal: dict):
    """Centralized trade execution function"""
    asset = signal["asset"]
    signal_id = signal["id"]
    executions = []

    # Execute Admin Trade
    if MT5_AVAILABLE:
        admin_result = await place_mt5_order(
            symbol=asset,
            order_type=signal["action"],
            volume=0.1,
            price=signal["price"],
            signal_id=signal_id,
            user=None,
            sl_level=signal.get("sl_level"),
            tp_levels=signal.get("tp_levels", [])
        )
        executions.append(("Admin", admin_result))

    # Execute User Trades
    with SessionLocal() as db:
        users = db.query(User).options(joinedload(User.broker_credentials)).all()
        for user in users:
            if user.broker_credentials:
                user_result = await place_mt5_order(
                    symbol=asset,
                    order_type=signal["action"],
                    volume=0.1,
                    price=signal["price"],
                    signal_id=signal_id,
                    user=user,
                    sl_level=signal.get("sl_level"),
                    tp_levels=signal.get("tp_levels", [])
                )
                executions.append((f"User {user.username}", user_result))

    # Start monitoring for these trades
    asyncio.create_task(monitor_executed_trades(executions, signal))
    
    return executions

async def monitor_trades():
    """Global trade monitoring function"""
    while not shutdown_flag.is_set():
        if MT5_AVAILABLE and not initialize_mt5_admin():
            logger.error("Global MT5 initialization failed for monitoring")
            await asyncio.sleep(60)
            continue
        
        try:
            positions = mt5.positions_get() if MT5_AVAILABLE else []
            if positions:
                for pos in positions:
                    try:
                        entry_price = pos.price_open
                        current_tick = mt5.symbol_info_tick(pos.symbol)
                        if not current_tick:
                            continue
                        
                        current_price = current_tick.ask if pos.type == 0 else current_tick.bid
                        asset = pos.symbol
                        asset_type = "forex" if "USD" in asset else "crypto"
                        
                        # Get fresh market data for dynamic adjustment
                        data = await fetch_market_data(asset, asset_type)
                        if data.empty:
                            continue
                            
                        data = compute_professional_indicators(data)
                        atr = data["ATR"].iloc[-1] if "ATR" in data.columns else current_price * 0.005
                        
                        # Dynamic SL/TP adjustment
                        if pos.type == 0 and current_price > entry_price * 1.01:  # BUY in profit
                            new_sl = current_price - 1.5 * atr
                            new_tp = current_price + 3 * atr
                        elif pos.type == 1 and current_price < entry_price * 0.99:  # SELL in profit
                            new_sl = current_price + 1.5 * atr
                            new_tp = current_price - 3 * atr
                        else:
                            continue
                            
                        result = update_mt5_trade(pos, new_sl, new_tp)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"Trade {pos.ticket} updated: SL={new_sl:.5f}, TP={new_tp:.5f}")
                    except Exception as e:
                        logger.error(f"Error adjusting trade {pos.ticket}: {str(e)}")
        except Exception as e:
            logger.error(f"Trade monitoring error: {str(e)}")
            
        await asyncio.sleep(30)  # Check every 30 seconds

async def monitor_executed_trades(executions: list, original_signal: dict):
    """Enhanced trade monitoring with Telegram updates"""
    asset = original_signal["asset"]
    update_interval = 60  # seconds
    
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
                            current_price = mt5.symbol_info_tick(pos.symbol).ask if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).bid
                            pnl = (current_price - pos.price_open) * pos.volume * (1 if pos.type == 0 else -1)
                            status_updates.append(
                                f"{account} Position: {pos.volume} lots | PnL: ${pnl:.2f}"
                            )
            except Exception as e:
                logger.error(f"Monitoring error for {account}: {str(e)}")
        
        # Send consolidated update
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
                        json={
                            "chat_id": TELEGRAM_CHANNEL_ID,
                            "text": message,
                            "parse_mode": "Markdown"
                        }
                    )
            except Exception as e:
                logger.error(f"Monitoring update failed: {str(e)}")
        
        await asyncio.sleep(update_interval)

# ---------------- RL Retraining Stub ----------------
def retrain_rl_models():
    logger.info("Starting retraining of RL models (stub)...")
    # RL backtesting/retraining code can be integrated here in the future.
    logger.info("RL models retraining stub complete.")

async def schedule_retraining():
    while not shutdown_flag.is_set():
        retrain_rl_models()
        await asyncio.sleep(21600)

# ---------------- RL Model Loading ----------------
def load_all_models():
    global models
    from trading_env import TradingEnv  # Ensure trading_env.py is in your project root.
    env = TradingEnv()
    custom_objects = {
        "lr_schedule": lambda _: 0.001,
        "exploration_schedule": lambda _: 0.1,
        "clip_range": lambda _: 0.2,
        "n_envs": 1,
        "device": "cpu",
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "_vec_normalize_env": DummyVecEnv([lambda: TradingEnv()]),
        "policy_kwargs": {},
        "optimizer_class": RMSpropTFLike
    }
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
        "dqn": ("models/dqn_model.zip", custom_objects),
        "ppo": ("models/ppo_model.zip", custom_objects),
        "svr": "models/svr_model.pkl"
    }
    with model_lock:
        for name, path_info in model_paths.items():
            path, custom_obj = path_info if isinstance(path_info, tuple) else (path_info, {})
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
                    models[name] = DQN.load(path, custom_objects=custom_obj) if name == "dqn" else PPO.load(path, custom_objects=custom_obj)
                elif name == "svr":
                    with open(path, "rb") as f:
                        models[name] = joblib.load(f)
                logger.info(f"Successfully loaded {name}")
            except Exception as e:
                logger.error(f"Error loading {name}: {str(e)}")
    return models

# ---------------- Data Fetching Functions ----------------
# Add this to your imports
from tenacity import RetryError

# Modify the fetch_market_data function
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_market_data(symbol: str, asset_type: str):
    cache_key = f"{symbol}-{asset_type}-{datetime.now().hour}"
    cached_data = cache.get(cache_key)
    
    # Prioritize cached data when approaching limits
    if api_calls_today >= TWELVEDATA_DAILY_LIMIT * 0.9:
        if cached_data is not None:
            logger.warning("Approaching API limit - using cached data")
            return cached_data
            
    try:
        async with aiohttp.ClientSession() as session:
            # Try TwelveData first
            twelve_data = await fetch_twelvedata(session, symbol, asset_type)
            if twelve_data is not None:
                track_api_usage("TwelveData")
                cache.set(cache_key, twelve_data, expire=300)  # 5 minute cache
                return twelve_data
                
            # Fallback to AlphaVantage
            alpha_data = await fetch_alphavantage(session, symbol, asset_type)
            if alpha_data is not None:
                track_api_usage("AlphaVantage")
                cache.set(cache_key, alpha_data, expire=300)
                return alpha_data
                
        logger.warning("All data sources failed, using cached data if available")
        return cached_data or pd.DataFrame()
        
    except RetryError:
        logger.error(f"Max retries exceeded for {symbol}")
        return cached_data or pd.DataFrame()
    except Exception as e:
        logger.error(f"Data fetch error: {str(e)}")
        return cached_data or pd.DataFrame()

async def fetch_twelvedata(session, symbol: str, asset_type: str):
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
        async with session.get("https://api.twelvedata.com/time_series", params=params) as response:
            data = await response.json()
            if "code" in data and data["code"] == 429:
                logger.error("TwelveData API limit exceeded, waiting for 60 seconds before retrying.")
                await asyncio.sleep(60)
                return None
            return process_data(data)
    except Exception as e:
        logger.error(f"TwelveData fetch failed: {str(e)}")
        return None

async def fetch_alphavantage(session, symbol: str, asset_type: str):
    if not ALPHAVANTAGE_API_KEY:
        return None
    function = "DIGITAL_CURRENCY_DAILY" if asset_type == "crypto" else "FX_DAILY"
    params = {
        "function": function,
        "symbol": symbol.split("/")[0] if asset_type == "crypto" else symbol,
        "market": "USD",
        "apikey": ALPHAVANTAGE_API_KEY,
        "datatype": "json"
    }
    try:
        async with session.get("https://www.alphavantage.co/query", params=params) as response:
            if response.status != 200:
                return None
            data = await response.json()
            return process_alpha_data(data, asset_type)
    except Exception as e:
        logger.error(f"AlphaVantage fetch failed: {str(e)}")
        return None

def process_data(data: dict) -> pd.DataFrame:
    if "values" not in data and "Time Series (Digital Currency Daily)" not in data:
        return pd.DataFrame()
    if "values" in data:
        df = pd.DataFrame(data["values"]).rename(columns={"datetime": "timestamp"})
    else:
        ts_key = "Time Series (Digital Currency Daily)"
        df = pd.DataFrame.from_dict(data[ts_key], orient="index").reset_index().rename(columns={"index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=True).set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

def process_alpha_data(data: dict, asset_type: str) -> pd.DataFrame:
    ts_key = "Time Series (Digital Currency Daily)" if asset_type == "crypto" else "Time Series FX (Daily)"
    if ts_key not in data:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data[ts_key], orient="index").reset_index().rename(columns={"index": "timestamp"})
    if asset_type == "crypto":
        df = df.rename(columns={
            "1a. open (USD)": "open",
            "2a. high (USD)": "high",
            "3a. low (USD)": "low",
            "4a. close (USD)": "close"
        })
    else:
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=True).set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

# ---------------- Enhanced News & Sentiment ----------------
async def fetch_market_news(asset_type: str):
    if not NEWSAPI_KEY:
        logger.warning("NEWS_API_KEY not set; skipping news fetch.")
        return []
    params = {"category": "business", "language": "en", "apiKey": NEWSAPI_KEY, "pageSize": 5}
    try:
        async with aiohttp.ClientSession() as session:
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

# ---------------- Technical Indicators & Feature Engineering ----------------
def compute_professional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # ===== Core Trend Indicators =====
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku is not None:
            ichimoku_df = ichimoku[0]
            df = df.join(ichimoku_df.rename(columns={
                'ISA_9': 'ICHIMOKU_TENKAN',
                'ISB_26': 'ICHIMOKU_KIJUN',
                'ITS_9': 'ICHIMOKU_SENKOU_A',
                'IKS_26': 'ICHIMOKU_SENKOU_B',
                'ICS_26': 'ICHIMOKU_CHIKOU'
            }), how='left')

        # ===== Momentum Oscillators =====
        df['RSI_14'] = ta.rsi(df['close'], length=14).fillna(50)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df = df.join(stoch.rename(columns={
                'STOCHk_14_3_3': 'STOCH_%K',
                'STOCHd_14_3_3': 'STOCH_%D'
            }), how='left')

        # ===== Volatility Analysis =====
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df = df.join(bb.rename(columns={
                'BBU_20_2.0': 'BB_UPPER',
                'BBM_20_2.0': 'BB_MIDDLE',
                'BBL_20_2.0': 'BB_LOWER'
            }), how='left')
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # ===== Volume Analysis =====
        if 'volume' in df.columns:
            df['OBV'] = ta.obv(df['close'], df['volume'])
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            if vwap is not None:
                df['VWAP'] = vwap

        # ===== Trend Strength =====
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['ADX_14'] = adx['ADX_14']
            df['+DI_14'] = adx['DMP_14']
            df['-DI_14'] = adx['DMN_14']
        else:
            df['ADX_14'] = df['+DI_14'] = df['-DI_14'] = 0.0

        # ===== Multi-Timeframe Features =====
        resampled = df.resample('15T').agg({
            'close': 'last',
            'volume': 'sum' if 'volume' in df.columns else 'count'
        })
        resampled['RSI_15T'] = ta.rsi(resampled['close'], length=14)
        df = df.join(resampled, rsuffix='_15T', how='left')

        # ===== Feature Validation =====
        required_columns = [
            'close', 'RSI_14', 'STOCH_%K', 'STOCH_%D', 'BB_UPPER', 'BB_MIDDLE',
            'BB_LOWER', 'ATR_14', 'ADX_14', '+DI_14', '-DI_14', 'SMA_50', 'SMA_200',
            'EMA_9', 'EMA_26', 'ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A',
            'ICHIMOKU_SENKOU_B', 'OBV', 'VWAP', 'RSI_15T'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0

        # ===== Feature Smoothing =====
        for col in ['RSI_14', 'STOCH_%K', 'STOCH_%D']:
            df[col] = df[col].rolling(window=3, min_periods=1).mean()

    except Exception as e:
        logger.error(f"Indicator error: {str(e)}")
    return df.ffill().bfill().dropna(how='all', axis=1)

# ---------------- Feature Preparation ----------------
def prepare_features(df: pd.DataFrame, model_type: str):
    try:
        required_columns = [
            'close', 'RSI_14', 'STOCH_%K', 'STOCH_%D', 'BB_UPPER', 'BB_MIDDLE',
            'BB_LOWER', 'ATR_14', 'ADX_14', '+DI_14', '-DI_14', 'SMA_50', 'SMA_200',
            'EMA_9', 'EMA_26', 'ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A',
            'ICHIMOKU_SENKOU_B', 'OBV', 'VWAP', 'RSI_15T'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0

        feature_set = [
            'close', 'SMA_50', 'SMA_200', 'EMA_9', 'EMA_26',
            'ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A',
            'ICHIMOKU_SENKOU_B', 'RSI_14', 'STOCH_%K', 'STOCH_%D',
            'ADX_14', '+DI_14', '-DI_14', 'BB_UPPER', 'BB_MIDDLE',
            'BB_LOWER', 'ATR_14', 'OBV', 'VWAP', 'RSI_15T'
        ]
        
        features = df[feature_set].dropna()
        
        if model_type == "ml":
            return features.tail(1).values.reshape(1, -1)
        elif model_type == "dl":
            sequence = features.tail(100).values
            return sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        elif model_type == "rl":
            rl_features = features.tail(1).copy()
            rl_features['ATR_Ratio'] = rl_features['ATR_14'] / rl_features['close']
            return rl_features.values.reshape(1, -1)

    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}")
        return None

def prepare_features(df: pd.DataFrame, model_type: str):
    try:
        required_columns = [
            'close', 'RSI_14', 'STOCH_%K', 'STOCH_%D', 'BB_UPPER', 'BB_MIDDLE',
            'BB_LOWER', 'ATR_14', 'ADX_14', '+DI_14', '-DI_14', 'SMA_50', 'SMA_200',
            'EMA_9', 'EMA_26', 'ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A',
            'ICHIMOKU_SENKOU_B', 'OBV', 'VWAP', 'RSI_15T'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0

        feature_set = [
            'close', 'SMA_50', 'SMA_200', 'EMA_9', 'EMA_26',
            'ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A',
            'ICHIMOKU_SENKOU_B', 'RSI_14', 'STOCH_%K', 'STOCH_%D',
            'ADX_14', '+DI_14', '-DI_14', 'BB_UPPER', 'BB_MIDDLE',
            'BB_LOWER', 'ATR_14', 'OBV', 'VWAP', 'RSI_15T'
        ]
        
        features = df[feature_set].dropna()
        
        if model_type == "ml":
            return features.tail(1).values.reshape(1, -1)
        elif model_type == "dl":
            sequence = features.tail(100).values
            return sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        elif model_type == "rl":
            rl_features = features.tail(1).copy()
            rl_features['ATR_Ratio'] = rl_features['ATR_14'] / rl_features['close']
            return rl_features.values.reshape(1, -1)

    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}")
        return None

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
                    pred = models[model].predict(ml_data)
                    predictions[model] = float(pred[0])
                except Exception as e:
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

def compute_trade_levels(price: float, atr: float, side: str) -> dict:
    if atr == 0:
        atr = price * 0.01  # Fallback ATR value
    
    trend_multiplier = 1.5 if (atr/price) > 0.01 else 1.2
    volatility_adjusted_atr = atr * trend_multiplier
    
    if side == "BUY":
        return {
            "sl_level": price - volatility_adjusted_atr,
            "tp_levels": [
                price + volatility_adjusted_atr * 1,
                price + volatility_adjusted_atr * 2,
                price + volatility_adjusted_atr * 3
            ]
        }
    else:
        return {
            "sl_level": price + volatility_adjusted_atr,
            "tp_levels": [
                price - volatility_adjusted_atr * 1,
                price - volatility_adjusted_atr * 2,
                price - volatility_adjusted_atr * 3
            ]
        }

def generate_institutional_signal(df: pd.DataFrame, predictions: dict, asset: str) -> dict:
    price = df["close"].iloc[-1]
    atr = df["ATR_14"].iloc[-1] if "ATR_14" in df.columns else 0.0
    adx = df["ADX_14"].iloc[-1] if "ADX_14" in df.columns else 0.0
    di_plus = df["+DI_14"].iloc[-1] if "+DI_14" in df.columns else 0.0
    di_minus = df["-DI_14"].iloc[-1] if "-DI_14" in df.columns else 0.0

    signal = {
        "asset": asset,
        "action": "HOLD",
        "price": price,
        "confidence": 0,
        "timestamp": datetime.utcnow().isoformat(),
        "indicators": {
            "RSI_14": df["RSI_14"].iloc[-1],
            "ATR_14": atr,
            "ADX_14": adx,
            "+DI_14": di_plus,
            "-DI_14": di_minus,
            "ICHIMOKU_SENKOU_A": df["ICHIMOKU_SENKOU_A"].iloc[-1],
            "VWAP": df["VWAP"].iloc[-1] if "VWAP" in df.columns else price
        },
        "predictions": predictions,
        "news_sentiment": "NEUTRAL",
        "tp_levels": [],
        "sl_level": None,
        "trade_mode": None,
        "predicted_change": predictions.get("lstm", 0)
    }

    try:
        # Trend Validation via Ichimoku Cloud
        price_above_cloud = (price > df["ICHIMOKU_SENKOU_A"].iloc[-1] and 
                             price > df["ICHIMOKU_SENKOU_B"].iloc[-1])
        price_below_cloud = (price < df["ICHIMOKU_SENKOU_A"].iloc[-1] and 
                             price < df["ICHIMOKU_SENKOU_B"].iloc[-1])
        
        strong_trend = adx > 25
        di_crossover = di_plus > di_minus
        
        valid_preds = [p for p in predictions.values() if isinstance(p, (int, float)) and not np.isnan(p)]
        if len(valid_preds) < 3:
            logger.info(f"Not enough valid predictions for {asset}. Returning HOLD.")
            return signal

        prediction_avg = np.mean(valid_preds)
        price_diff = prediction_avg - price
        risk_ratio = abs(price_diff) / atr if atr > 0 else 0
        
        # Confidence Calculation (ADX-weighted)
        base_conf = min(100, risk_ratio * 100 * (adx/100 + 0.5))
        consensus_std = np.std(valid_preds)
        consensus_factor = 1 - min(consensus_std / (abs(prediction_avg) + 1e-5), 1)
        confidence = min(100, base_conf * consensus_factor)

        # Signal Logic based on trend, cloud position, and directional crossover
        if price_diff > 0 and price_above_cloud and strong_trend and di_crossover:
            signal.update({
                "action": "BUY",
                "confidence": confidence,
                **compute_trade_levels(price, atr, "BUY")
            })
        elif price_diff < 0 and price_below_cloud and strong_trend and not di_crossover:
            signal.update({
                "action": "SELL",
                "confidence": confidence,
                **compute_trade_levels(price, atr, "SELL")
            })

    except Exception as e:
        logger.error(f"Signal generation error for {asset}: {str(e)}")
    
    return signal

# ---------------- Trading Schedule Management ----------------
def is_forex_trading_hours() -> bool:
    now = datetime.utcnow().time()
    return dt_time(8, 0) <= now <= dt_time(17, 0)

def get_asset_universe() -> Dict[str, List[str]]:
    global trading_mode_override
    default_universe = {
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD"],
        "crypto": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD", "ADA/USD"]
    }
    if trading_mode_override:
        return default_universe.get(trading_mode_override.lower(), default_universe["forex"])
    return default_universe["forex"] if is_forex_trading_hours() else default_universe["crypto"]

# ---------------- Telegram Integration ----------------
try:
    from telegram import Update
    from telegram.ext import Application as TGApplication, CommandHandler, ContextTypes
except ImportError as e:
    logger.error("Ensure you have python-telegram-bot v20+ installed.")
    raise e

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
    try:
        parts = update.message.text.split()
        if len(parts) < 2:
            await update.message.reply_text(
                "Usage: /mode <forex|crypto|auto>\nExample: /mode forex",
                parse_mode="Markdown"
            )
            return
        mode = parts[1].lower()
        if mode not in ["forex", "crypto", "auto"]:
            await update.message.reply_text(
                "Invalid mode. Please choose 'forex', 'crypto', or 'auto'.",
                parse_mode="Markdown"
            )
            return
        async with aiohttp.ClientSession() as session:
            await session.post(f"http://127.0.0.1:8080/switch_mode?mode={mode}")
        await update.message.reply_text(
            f"🔔 [NekoAIBot] Trading mode switched to *{mode.upper()}*.\n— NekoAIBot",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ [NekoAIBot] Error switching mode: {e}", parse_mode="Markdown")

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
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
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": chat_id, "text": response_text, "parse_mode": "Markdown"}
            await session.post(url, json=payload)
    except Exception as e:
        logger.error(f"Telegram response error: {str(e)}")
    return {"status": "ok"}

def run_telegram_listener():
    """Ensures single instance of Telegram listener"""
    if not hasattr(app, 'telegram_running'):
        asyncio.run(start_telegram_listener())
        app.telegram_running = True
    else:
        logger.warning("Telegram listener already running")

# -------- Telegram Integration (Corrected) --------
telegram_app = None  # Singleton instance

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
        await telegram_app.stop()
        await telegram_app.shutdown()
        telegram_app = None
        logger.info("Telegram listener stopped")

# ---------------- Telegram Notifications ----------------
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
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHANNEL_ID, "text": message, "parse_mode": "Markdown", "disable_notification": False}
            await session.post(url, json=payload)
    except Exception as e:
        logger.error(f"Pre-signal error: {str(e)}")

async def send_telegram_alert(signal: dict):
    signal_id = signal.get("id")
    if signal_id in sent_alert_ids:
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
        destinations = [TELEGRAM_CHAT_ID, TELEGRAM_CHANNEL_ID]
        async with aiohttp.ClientSession() as session:
            for chat_id in destinations:
                if not chat_id:
                    continue
                await session.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True},
                    timeout=10
                )
    except Exception as e:
        logger.error(f"Alert system error: {str(e)}")

# ---------------- WebSocket Endpoint ----------------
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

# ---------------- API Endpoints ----------------
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
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": "🔧 System Test Message", "parse_mode": "Markdown"}
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

# ---------------- Premium Trading Worker (Real Implementation) ----------------
async def premium_trading_worker():
    """Global trading worker with proper signal sequencing, API delay, and detailed debug logging"""
    global signal_counter
    signal_cooldown = {}  # Track last signal time per asset

    while signal_generation_active:
        assets = get_asset_universe()
        for asset in assets:
            try:
                # Check cooldown to prevent excessive trading on the same asset
                last_signal = signal_cooldown.get(asset, datetime.min)
                if (datetime.now() - last_signal) < timedelta(minutes=5):
                    logger.info(f"Skipping {asset} due to cooldown.")
                    continue

                # Determine asset type
                asset_type = "crypto" if "/USD" in asset and asset not in ["EUR/USD", "GBP/USD"] else "forex"

                # Fetch market data
                data = await fetch_market_data(asset, asset_type)
                if data.empty:
                    logger.info(f"No data received for {asset}.")
                    continue

                # Compute technical indicators and predictions
                data = compute_professional_indicators(data)
                predictions = generate_institutional_predictions(data)
                signal = generate_institutional_signal(data, predictions, asset)

                # Use incremental counting for signal ID
                with signal_counter_lock:
                    signal_id = str(signal_counter)
                    signal_counter += 1
                signal["id"] = signal_id

                # Log the computed signal for debugging
                logger.info(
                    f"Asset: {asset} | Price: {signal.get('price', 'N/A')} | Action: {signal.get('action')} | "
                    f"Confidence: {signal.get('confidence', 0)} | Signal ID: {signal_id}"
                )

                # Check if the computed signal meets the criteria
                if signal["action"] in ["BUY", "SELL"] and signal["confidence"] >= 60:
                    logger.info(f"Signal qualifies for execution: {signal}")
                    # Phase 1: Pre-Signal Alert
                    await send_pre_signal_message(signal)

                    # Phase 2: Wait for confirmation (30 sec delay)
                    await asyncio.sleep(30)

                    # Phase 3: Final verification before executing trades
                    current_data = await fetch_market_data(asset, asset_type)
                    if not current_data.empty:
                        current_price = current_data["close"].iloc[-1]
                        price_change = abs(current_price - signal["price"]) / signal["price"]

                        logger.info(
                            f"Final verification for {asset}: Current Price = {current_price}, "
                            f"Original Price = {signal['price']}, Price Change = {price_change:.4f}"
                        )

                        if price_change < 0.01:  # Acceptable price change threshold
                            # Phase 4: Execute Trades on global and user accounts
                            await execute_trades(signal)

                            # Phase 5: Send Main Signal Notification
                            await send_telegram_alert(signal)

                            # Update cooldown to avoid rapid re-signaling
                            signal_cooldown[asset] = datetime.now()
                        else:
                            logger.info(f"Price deviation too high for {asset}. Signal aborted.")
                else:
                    logger.info(
                        f"Signal not qualified for {asset}. Action: {signal.get('action')}, "
                        f"Confidence: {signal.get('confidence', 0)}"
                    )

            except Exception as e:
                logger.error(f"Error processing {asset}: {str(e)}")

            # Delay between processing assets to avoid rapid API calls (10-15 sec delay)
            delay = random.randint(10, 15)
            logger.info(f"Waiting {delay} seconds before processing next asset.")
            await asyncio.sleep(delay)

        # Delay at the end of the full asset loop iteration if needed
        await asyncio.sleep(5)
        
# ---------------- Startup & Shutdown Events ----------------
@app.on_event("startup")
@app.on_event("startup")
@app.on_event("startup")
async def startup_event():
    load_all_models()
    
    # Start all services in main event loop
    asyncio.create_task(start_telegram_listener())
    asyncio.create_task(premium_trading_worker())
    asyncio.create_task(monitor_trades())
    asyncio.create_task(schedule_retraining())
    
    logger.info("NekoAIBot Active")

@app.on_event("shutdown")
async def shutdown_event():
    global signal_generation_active
    signal_generation_active = False
    shutdown_flag.set()
    shutdown_mt5_admin()
    logger.info("NekoAIBot Shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
