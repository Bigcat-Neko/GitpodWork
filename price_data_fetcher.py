import requests

API_KEY = "35d3a532b868472988a0697df9c1df13"  # Replace with your Twelve Data API key
BASE_URL = "https://api.twelvedata.com/time_series"

def get_live_forex_prices(symbol="EUR/USD", interval="1min", outputsize=60):
    """
    Fetches live Forex prices using Twelve Data API.
    Returns a list of closing prices (oldest first) with length equal to outputsize.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "values" in data:
        # Extract the closing prices from the values list
        prices = [float(item["close"]) for item in data["values"]]
        # Reverse the list so that the oldest price is first
        return prices[::-1]
    else:
        raise Exception("Error fetching live forex prices: " + str(data))

if __name__ == "__main__":
    try:
        prices = get_live_forex_prices()
        print("Live Forex Prices:", prices)
    except Exception as e:
        print("Error:", e)
