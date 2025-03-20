import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

# List of symbols to fetch (you can adjust this list)
symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD", "XRP/USD"]  # Excluding XAU/USD if desired

def fetch_data_for_symbol(symbol, interval="1min", outputsize=500):
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY
    }
    response = requests.get(base_url, params=params, timeout=10)
    data = response.json()
    if "values" in data:
        df = pd.DataFrame(data["values"])
        # Convert numeric columns (close, high, low, volume if exists)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime", inplace=True)
        df["asset"] = symbol  # Add a column to indicate the symbol
        return df
    else:
        raise Exception(f"Error fetching data for {symbol}: {data}")

# Fetch data for all symbols and combine them
all_data = []
for symbol in symbols:
    try:
        df_symbol = fetch_data_for_symbol(symbol)
        all_data.append(df_symbol)
        print(f"Data fetched for {symbol}")
    except Exception as e:
        print(e)

if all_data:
    training_df = pd.concat(all_data, ignore_index=True)
    # Save to CSV
    training_df.to_csv("training_data.csv", index=False)
    print("Training data saved to training_data.csv")
else:
    print("No data was fetched.")
