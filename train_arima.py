# arima_trainer.py
import os
import time
import joblib
import requests
import pandas as pd
from dotenv import load_dotenv
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Configuration
load_dotenv()
API_KEY = os.getenv('TWELVEDATA_API_KEY')
BASE_URL = "https://api.twelvedata.com/time_series"
MODEL_FILE = "all_arima_models.pkl"
RATE_LIMIT = 8  # Requests per minute
REQUEST_DELAY = 60 / RATE_LIMIT  # 7.5 seconds between requests

ASSETS = {
    'forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD'],
    'crypto': ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD', 'ADA/USD']
}

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ARIMATrainer:
    def __init__(self):
        self.models = {}
        self.last_request_time = 0
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def rate_limited_get(self, params):
        """Handle rate limiting with precise timing"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        
        response = requests.get(BASE_URL, params=params)
        self.last_request_time = time.time()
        return response

    def fetch_data(self, symbol, asset_type):
        """Fetch data with robust error handling"""
        params = {
            'symbol': symbol,
            'interval': '1day',
            'outputsize': '5000',
            'apikey': API_KEY,
            'format': 'JSON'
        }
        
        if asset_type == 'crypto':
            params['exchange'] = 'Binance'

        try:
            response = self.rate_limited_get(params)
            data = response.json()
            
            if data.get('status') != 'ok':
                if 'API credits' in data.get('message', ''):
                    print(f"Rate limited on {symbol}, waiting 60 seconds...")
                    time.sleep(60)
                    return self.fetch_data(symbol, asset_type)
                return None
                
            return self.process_data(pd.DataFrame(data['values']))

        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return None

    def process_data(self, df):
        """Clean and validate dataset"""
        return (df.rename(columns={'datetime': 'date'})
                .assign(date=lambda x: pd.to_datetime(x['date']),
                        close=lambda x: pd.to_numeric(x['close']))
                .sort_values('date')
                .set_index('date')
                .asfreq('D')
                .ffill()
                .bfill())

    def train_model(self, symbol, asset_type):
        """Complete training pipeline for one asset"""
        try:
            print(f"Processing {symbol}...")
            df = self.fetch_data(symbol, asset_type)
            
            if df is None or len(df) < 365:
                raise ValueError("Insufficient or invalid data")
                
            model = auto_arima(
                df['close'],
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=3,
                max_q=3,
                information_criterion='aic'
            )
            
            final_model = ARIMA(df['close'], order=model.order).fit()
            
            # Store model with metadata
            safe_symbol = symbol.replace('/', '_')
            self.models[safe_symbol] = {
                'model': final_model,
                'order': model.order,
                'asset_type': asset_type,
                'last_trained': pd.Timestamp.now(),
                'training_range': (str(df.index[0].date()), str(df.index[-1].date()))
            }
            
            print(f"Trained {symbol} (Order: {model.order})")
            return True
            
        except Exception as e:
            print(f"Failed {symbol}: {str(e)}")
            return False

    def save_models(self):
        """Save all models to a single file"""
        joblib.dump(self.models, os.path.join(self.models_dir, MODEL_FILE))
        print(f"\nSuccessfully saved {len(self.models)} models to {MODEL_FILE}")

    def train_all_assets(self):
        """Sequential training with rate limit enforcement"""
        print("Starting training with strict rate limiting...")
        
        # Process Forex first
        for symbol in ASSETS['forex']:
            if self.train_model(symbol, 'forex'):
                time.sleep(REQUEST_DELAY)
        
        # Process Crypto with additional caution
        time.sleep(10)  # Extra buffer between asset classes
        for symbol in ASSETS['crypto']:
            if self.train_model(symbol, 'crypto'):
                time.sleep(REQUEST_DELAY)
        
        # Save all models after completion
        self.save_models()

if __name__ == "__main__":
    trainer = ARIMATrainer()
    trainer.train_all_assets()