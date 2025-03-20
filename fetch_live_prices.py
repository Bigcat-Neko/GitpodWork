import os
import time
import requests
import numpy as np

BASE_URL = "https://api.twelvedata.com/time_series"

def get_live_forex_prices(symbol="EUR/USD", interval="1min", outputsize=60):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": os.getenv("TWELVEDATA_API_KEY")
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    data = response.json()
    if "values" in data:
        prices = [float(item["close"]) for item in data["values"]]
        return np.array(prices[::-1]).reshape(-1, 1)  # Reverse order for chronological order
    else:
        raise Exception("Error fetching live forex prices: " + str(data))
