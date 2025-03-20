import requests
import pandas as pd
import os
from datetime import datetime
from config import FOREX_API_KEY

# Alpha Vantage endpoint for daily Forex data (example: EUR/USD)
ALPHA_VANTAGE_URL = (
    "https://www.alphavantage.co/query"
    "?function=FX_DAILY"
    "&from_symbol=EUR"
    "&to_symbol=USD"
    f"&apikey={FOREX_API_KEY}"
)

def fetch_forex_data():
    response = requests.get(ALPHA_VANTAGE_URL)
    try:
        data = response.json()
    except Exception as e:
        print("Error decoding JSON:", e)
        return

    if "Time Series FX (Daily)" in data:
        ts_data = data["Time Series FX (Daily)"]
        df = pd.DataFrame(ts_data).T  # Transpose so each row is a date
        df.index.name = "date"
        # Rename columns for clarity
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        })
        # Add a timestamp column
        df["fetched_at"] = datetime.utcnow().isoformat()

        csv_file = "data/forex_data.csv"
        # Append to CSV if it exists, otherwise create a new file
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode="a", header=False)
        else:
            df.to_csv(csv_file)
        print("Forex data updated successfully and saved to 'data/forex_data.csv'.")
    else:
        print("Error fetching data from Alpha Vantage:", data)

if __name__ == '__main__':
    fetch_forex_data()
