import numpy as np
import pickle

# Load your sentiment model (adjust path as needed)
with open("models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

def normalize_sentiment(sentiment_score):
    """
    Adjusts sentiment scoring to prevent extreme negative biases.
    """
    if sentiment_score < -0.5:
        sentiment_score *= 0.7
    elif sentiment_score > 0.5:
        sentiment_score *= 1.2
    return np.clip(sentiment_score, -1, 1)

def analyze_news(news_text):
    """
    Runs sentiment analysis on a news headline and normalizes the score.
    Assumes the sentiment model returns a dict with a 'score' key.
    """
    result = sentiment_model.predict(news_text)
    if isinstance(result, dict):
        score = result.get("score", 0)
    else:
        score = result  # If the model returns a numeric score directly
    adjusted_score = normalize_sentiment(score)
    return adjusted_score

if __name__ == "__main__":
    test_news = "Stock market soars as investors gain confidence."
    sentiment = analyze_news(test_news)
    print(f"🔍 Adjusted Sentiment Score: {sentiment}")
