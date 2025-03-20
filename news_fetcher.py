import requests
import time
import os

# Replace with your actual API key from NewsData.io
NEWSDATA_API_KEY = "pub_73508e810aafdedfeb3530194f3dc81936013"
MAX_RETRIES = 3  # Maximum number of retry attempts
INITIAL_DELAY = 2  # Seconds to wait before the first retry

def get_forex_news():
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": os.getenv("NEWS_API_KEY"),  # Ensure you have this key in your .env file
        "language": "en",
        "category": "business",
        "q": "forex",
        "country": "us"
    }
    retries = 0
    while retries < 3:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success" and data.get("results"):
                return [article["title"] for article in data["results"]]
        elif response.status_code == 429:
            retry_after = response.json().get("parameters", {}).get("retry_after", 2)
            time.sleep(retry_after)
            retries += 1
        else:
            break
    return ["No news available."]
