# poll_signals.py
import time
import requests
from live_data_fetcher import get_live_forex_prices, get_live_news_headline

# URL for your FastAPI prediction endpoint
PREDICTION_URL = "http://127.0.0.1:8080/ai/predict"

def poll_and_predict():
    while True:
        try:
            # Fetch live Forex prices and news headline
            prices = get_live_forex_prices("EURUSD")
            headline = get_live_news_headline("forex")
            print("Fetched live data:")
            print("Prices:", prices)
            print("Headline:", headline)
            
            # Build the JSON payload
            payload = {"prices": prices, "headline": headline}
            
            # Call your prediction endpoint
            response = requests.post(PREDICTION_URL, json=payload)
            signal = response.json()
            print("Live Signal Generated:", signal)
        except Exception as e:
            print("Error during polling:", e)
        
        # Wait 60 seconds (adjust as needed)
        time.sleep(60)

if __name__ == "__main__":
    poll_and_predict()
