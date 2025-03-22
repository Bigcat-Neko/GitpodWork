# live_data.py
import os
import requests
from dotenv import load_dotenv
import pandas as pd

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("TWELVEDATA_API_KEY")
if not API_KEY:
    raise Exception("TwelveData API key not found in .env file.")

def fetch_live_data(symbol, interval="1day", outputsize=10):
    """
    Fetch live data for a given symbol using TwelveData.
    Returns a DataFrame.
    """
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if "values" in data:
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        # Convert price columns to numeric types
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        return df.sort_values("datetime")
    else:
        raise Exception(f"Error fetching data for {symbol}: " + str(data))

def fetch_all_live_data():
    """
    Fetch live data for all assets.
    Adjust the lists below to include your desired forex and crypto assets.
    """
    # Example symbols for major forex pairs and popular crypto assets
    forex_symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
    crypto_symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD"]

    all_assets = forex_symbols + crypto_symbols
    live_data_dict = {}
    
    for symbol in all_assets:
        try:
            print(f"Fetching data for {symbol}...")
            df = fetch_live_data(symbol, interval="1day", outputsize=15)
            live_data_dict[symbol] = df
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")
    
    return live_data_dict

# Example usage:
if __name__ == "__main__":
    live_data = fetch_all_live_data()
    for symbol, df in live_data.items():
        print(f"\nLive data for {symbol}:")
        print(df.head())
