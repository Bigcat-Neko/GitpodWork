import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("TWELVEDATA_API_KEY")
load_dotenv()
print(os.getenv("TWELVEDATA_API_KEY"))  # Debugging step

# List of major forex pairs and crypto assets
FOREX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]
CRYPTO_ASSETS = ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", "SOL/USD", "DOGE/USD"]

# Twelve Data API URL
BASE_URL = "https://api.twelvedata.com/time_series"

# Function to fetch real-time data
def fetch_data(symbol, interval="1min"):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 1,  # Only fetch the latest data point
        "apikey": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    # Check if data is valid
    if "values" in data:
        latest_data = data["values"][0]
        return {
            "symbol": symbol,
            "datetime": latest_data["datetime"],
            "open": float(latest_data["open"]),
            "high": float(latest_data["high"]),
            "low": float(latest_data["low"]),
            "close": float(latest_data["close"]),
            "volume": float(latest_data.get("volume", 1))  # Default to 1 if volume is missing
        }
    else:
        print(f"Error fetching data for {symbol}: {data}")
        return None

# Function to collect data for all assets
def collect_market_data():
    all_data = []
    
    for asset in FOREX_PAIRS + CRYPTO_ASSETS:
        print(f"Fetching data for {asset}...")
        data = fetch_data(asset)
        
        if data:
            all_data.append(data)
        
        # Add delay to avoid API rate limiting
        time.sleep(10 + 5 * (os.urandom(1)[0] / 255))  # Random delay between 10-15 sec
    
    return pd.DataFrame(all_data)

# Run data collection
df = collect_market_data()

# Save to CSV file
csv_filename = f"data/market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_filename, index=False)
print(f"Market data saved to {csv_filename}")
