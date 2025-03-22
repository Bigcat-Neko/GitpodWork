import os
import requests
import pandas as pd
import io
import time

# Set your TwelveData API key
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API")  # Ensure your .env contains TWELVE_DATA_API

# Full list of major Forex pairs and Crypto symbols
FOREX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
CRYPTO_SYMBOLS = ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", "SOL/USD", "DOGE/USD"]

def fetch_market_data(symbol, asset_type="forex"):
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "5min",
        "apikey": TWELVE_DATA_API_KEY,
        "format": "CSV"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text), sep=";")
        # Rename columns if needed
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
        # Ensure required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        df["symbol"] = symbol
        return df
    else:
        print(f"❌ Failed to fetch {asset_type} data for {symbol}: {response.text}")
        return None

all_data = []

for pair in FOREX_PAIRS:
    print(f"📡 Fetching Forex Data: {pair}")
    df = fetch_market_data(pair, asset_type="forex")
    if df is not None:
        all_data.append(df)
    time.sleep(10)  # Wait to avoid rate limits

for crypto in CRYPTO_SYMBOLS:
    print(f"📡 Fetching Crypto Data: {crypto}")
    df = fetch_market_data(crypto, asset_type="crypto")
    if df is not None:
        all_data.append(df)
    time.sleep(10)

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("market_data.csv", index=False)
    print("✅ Market data successfully saved as market_data.csv")
else:
    print("❌ No market data was retrieved.")
