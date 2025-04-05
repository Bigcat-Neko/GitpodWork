import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

# Load API Key
load_dotenv()
TWELVE_DATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

# Define Symbols (Forex uses /, Crypto uses /)
forex_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
crypto_pairs = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD"]

# Save Path
SAVE_PATH = "historical_data/all_data.csv"

# Ensure directory exists
os.makedirs("historical_data", exist_ok=True)

def fetch_data(symbol):
    """Fetch real-time forex or crypto data from Twelve Data."""
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": "1min",
        "apikey": TWELVE_DATA_API_KEY,
        "outputsize": 1,
        "format": "json"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "values" not in data:
        print(f"Error fetching {symbol}: {data.get('message', 'Unknown error')}")
        return None

    latest_data = data["values"][0]
    latest_data["symbol"] = symbol

    return latest_data

# Fetch Data
all_data = []

for symbol in forex_pairs + crypto_pairs:
    print(f"Fetching data for {symbol}...")
    data = fetch_data(symbol)

    if data:
        all_data.append(data)

    time.sleep(15)  # Prevent rate limiting

# Save to CSV
if all_data:
    df = pd.DataFrame(all_data)

    # Ensure all required columns exist
    required_columns = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # Fill missing columns with None

    df.to_csv(SAVE_PATH, index=False)
    print(f"All data saved to {SAVE_PATH}")
else:
    print("No data collected.")
