import requests
import pandas as pd
import os
from datetime import datetime
from config import NEWS_API_KEY  # Ensure this variable is set in your .env and config.py

# NewsAPI endpoint for Forex-related news
NEWS_URL = f"https://newsapi.org/v2/everything?q=forex&apiKey={NEWS_API_KEY}"

def fetch_news_sentiment():
    response = requests.get(NEWS_URL)
    try:
        data = response.json()
    except Exception as e:
        print("Error decoding JSON:", e)
        return
    
    if "articles" in data and data["articles"]:
        # Get the first 10 articles
        articles = data["articles"][:10]
        news_list = []
        for article in articles:
            news_list.append({
                "title": article.get("title"),
                "description": article.get("description"),
                "publishedAt": article.get("publishedAt"),
                "source": article.get("source", {}).get("name")
            })
        df = pd.DataFrame(news_list)
        # Add current timestamp
        df["fetched_at"] = datetime.utcnow().isoformat()
        
        csv_file = "data/forex_news.csv"
        # Append new data if file exists; otherwise, create a new file with header.
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        print("Forex news data updated and appended to 'data/forex_news.csv'.")
    else:
        print("Error fetching news data:", data)

if __name__ == '__main__':
    fetch_news_sentiment()
