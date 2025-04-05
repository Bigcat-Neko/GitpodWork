import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import talib

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# List of Forex-major pairs and crypto pairs
assets = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", 
    "EUR/GBP", "EUR/JPY", "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", 
    "SOL/USD", "DOGE/USD", "DOT/USD", "MATIC/USD", "LTC/USD"
]

# Define function to fetch and process data
def fetch_data(asset):
    print(f"🔄 Fetching live data for {asset}...")
    
    url = f"https://api.twelvedata.com/time_series"
    params = {
        'symbol': asset,
        'interval': '1h',  # 1-hour data for now; you can adjust based on your needs
        'apikey': API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'values' not in data:
        print(f"⚠️ Error fetching {asset}: {data.get('message', 'Unknown error')}")
        return None
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data['values'])
    
    # Ensure the proper column types
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # Calculate technical indicators
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Drop rows with NaN values (due to technical indicators requiring past data)
    df.dropna(inplace=True)
    
    # Save the data
    file_path = f"data/{asset.replace('/', '-')}.csv"
    df.to_csv(file_path, index=False)
    
    print(f"✅ Data saved for {asset}")
    return df

# Main script to fetch data for each asset
def main():
    for asset in assets:
        fetch_data(asset)
        time.sleep(15)  # Wait 15 seconds before fetching the next asset to avoid hitting API rate limit

if __name__ == "__main__":
    main()
