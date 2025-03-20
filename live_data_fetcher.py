# live_data_fetcher.py
import requests

# Replace these with your own API keys
ALPHA_VANTAGE_API_KEY = "J3MLNRSIUIKHX9YP"
NEWS_API_KEY = "3a35a4219c3e41b8b9ff607f4582ffd8"

def get_live_forex_prices(pair="EURUSD"):
    """
    Fetches real-time Forex data for the given currency pair using Alpha Vantage.
    Returns a list of closing prices (at least 60 data points).
    """
    url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={pair[:3]}&to_symbol={pair[3:]}&interval=1min&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    # Alpha Vantage returns a nested JSON with keys like "Time Series FX (1min)"
    time_series_key = "Time Series FX (1min)"
    if time_series_key not in data:
        raise Exception("Error fetching Forex data: " + str(data))
    
    # Get the latest 60 entries sorted by timestamp (oldest first)
    sorted_times = sorted(data[time_series_key].keys())
    prices = []
    for t in sorted_times[-60:]:
        price = float(data[time_series_key][t]["4. close"])
        prices.append(price)
    return prices

def get_live_news_headline(query="forex"):
    """
    Fetches the latest news headline related to forex using NewsAPI.
    Returns a headline string.
    """
    url = f"https://newsapi.org/v2/top-headlines?category=business&q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("status") != "ok" or not data.get("articles"):
        raise Exception("Error fetching news: " + str(data))
    # Return the title of the first article
    return data["articles"][0]["title"]

if __name__ == "__main__":
    try:
        prices = get_live_forex_prices("EURUSD")
        print("Live Forex Prices (latest 60):", prices)
        headline = get_live_news_headline("forex")
        print("Live News Headline:", headline)
    except Exception as e:
        print("Error:", e)
