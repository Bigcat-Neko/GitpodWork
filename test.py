import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv(override=True)
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

def fetch_data(symbol, interval="1h"):
    """
    Fetch historical market data from TwelveData API.

    Parameters:
        symbol (str): Trading pair (e.g., "EURUSD" for Forex, "BTC/USD" for Crypto).
        interval (str): Timeframe ("1min", "5min", "15min", "1h", "4h", "1day").

    Returns:
        DataFrame: Processed historical data for backtesting.
    """
    base_url = "https://api.twelvedata.com/time_series"
    
    params = {
        "symbol": symbol,  
        "interval": interval,
        "apikey": TWELVEDATA_API_KEY,
        "outputsize": "5000",
        "format": "json"
    }

    response = requests.get(base_url, params=params).json()
    
    # Debugging: Print response to check API output
    print(f"Fetching data for {symbol}")
    print(response)

    if "values" not in response:
        raise ValueError(f"Unexpected API response: {response}")

    # Convert API response to DataFrame
    df = pd.DataFrame(response["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float)
    df = df.sort_index()

    return df

# List of Forex & Crypto pairs (without slashes for Forex)
forex_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "USD/CHF", "NZD/USD"]
crypto_pairs = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "ADA/USD", "BNB/USD", "DOGE/USD"]

# Fetch and store data with a 10-second delay to avoid rate limits
market_data = {}

for pair in forex_pairs + crypto_pairs:
    try:
        market_data[pair] = fetch_data(pair, interval="1h")
    except Exception as e:
        print(f"Error fetching {pair}: {e}")

    time.sleep(10)  # 10-second delay to avoid rate limits

# Print sample data
for pair, df in market_data.items():
    print(f"\n{pair} Data Sample:\n", df.head())

# Now, this data can be used for backtesting.
