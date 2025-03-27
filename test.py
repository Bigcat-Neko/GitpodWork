import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("TWELVEDATA_API_KEY")
if not API_KEY:
    raise ValueError("TWELVEDATA_API_KEY not found in environment variables.")

# Define the symbols: Forex main pairs and common crypto pairs
symbols = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF",  # Forex
    "BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD"               # Cryptocurrencies
]

# Base URL for TwelveData's time series endpoint
base_url = "https://api.twelvedata.com/time_series"

# Dictionary to store market data per symbol
market_data = {}

# Loop through each symbol, fetch the data, and wait 10 seconds between calls
for symbol in symbols:
    print(f"Fetching data for {symbol} ...")
    params = {
        "symbol": symbol,
        "interval": "1day",   # Change interval as needed
        "outputsize": 30,     # Number of data points to retrieve; adjust as needed
        "apikey": API_KEY
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if "values" in data:
        market_data[symbol] = data["values"]
        print(f"Data for {symbol} fetched successfully.")
    else:
        # Print error message if data isn't available (e.g., API rate limit reached)
        error_msg = data.get("message", "Unknown error")
        print(f"Error fetching data for {symbol}: {error_msg}")
    
    # Wait for 10 seconds before the next API call
    time.sleep(10)

# Convert the fetched data into a pandas DataFrame.
# Here, we create one DataFrame per symbol and then concatenate them along the columns.
dfs = []
for sym, values in market_data.items():
    df = pd.DataFrame(values)
    df["symbol"] = sym  # add a column to identify the symbol
    dfs.append(df)

if dfs:
    full_df = pd.concat(dfs, ignore_index=True)
    # Save the combined DataFrame to CSV
    output_path = "data/market_features.csv"
    full_df.to_csv(output_path, index=False)
    print(f"Saved market features for all symbols to {output_path}")
else:
    print("No market data fetched.")
