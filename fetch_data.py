import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

# Load your environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("TWELVEDATA_API_KEY")

# Function to fetch historical data from Twelve Data for a given symbol.
def fetch_data(symbol, interval='1day', outputsize=200):
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': interval,
        'outputsize': outputsize,
        'apikey': API_KEY,
        'format': 'JSON'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "values" in data:
        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Convert price columns to float
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        df['asset'] = symbol
        df.sort_values("datetime", inplace=True)
        return df
    else:
        print(f"Error fetching data for {symbol}: {data.get('message', 'Unknown error')}")
        return pd.DataFrame()

# Define lists of symbols for Forex and Crypto.
forex_main = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CHF", "USD/CAD", "NZD/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY"
]

crypto_assets = [
    "BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD",
    "BCH/USD", "ADA/USD", "DOT/USD", "LINK/USD"
]

dfs = []

# Total delay between calls (in seconds) to not exceed the rate limit.
# Adjust this delay based on your API plan's rate limit.
delay_seconds = 10

# Fetch Forex data
for symbol in forex_main:
    print(f"Fetching Forex data for: {symbol}")
    df = fetch_data(symbol, interval="1day", outputsize=200)
    if not df.empty:
        dfs.append(df)
    time.sleep(delay_seconds)  # Delay between calls

# Fetch Crypto data
for symbol in crypto_assets:
    print(f"Fetching Crypto data for: {symbol}")
    df = fetch_data(symbol, interval="1day", outputsize=200)
    if not df.empty:
        dfs.append(df)
    time.sleep(delay_seconds)  # Delay between calls

# Combine all data frames into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Create a new feature - a simple moving average (SMA) of the closing price over 5 periods
combined_df["SMA_5"] = combined_df.groupby("asset")["close"].transform(lambda x: x.rolling(window=5).mean())

# Drop any rows with missing values (which may appear due to the rolling window calculation)
combined_df.dropna(inplace=True)

# Save the combined processed data to a CSV file
combined_df.to_csv("training_data.csv", index=False)

# Print the first few rows to verify the results
print(combined_df.head())
