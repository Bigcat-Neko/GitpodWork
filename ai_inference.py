# ai_inference.py
from fastapi import APIRouter, HTTPException
import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime
import uuid

router = APIRouter()

# Load the trained LSTM model for Forex
try:
    lstm_model = tf.keras.models.load_model("models/trade_signal_model.h5")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"LSTM model load failed: {e}")

# Load the sentiment analysis model
try:
    with open("models/sentiment_model.pkl", "rb") as f:
        sentiment_pipeline = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Sentiment model load failed: {e}")

# Load the anomaly detection model (dummy example)
try:
    with open("models/anomaly_model.pkl", "rb") as f:
        anomaly_model = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Anomaly model load failed: {e}")

# Load the price scaler
try:
    with open("models/price_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Scaler load failed: {e}")

@router.post("/predict")
def predict_signal(data: dict):
    """
    Expects a JSON body with:
      - prices: a list of recent closing prices (>= 60 values)
      - headline: a news headline (or aggregated text) for sentiment analysis
    """
    prices = data.get("prices")
    headline = data.get("headline")
    
    if not prices or len(prices) < 60:
        raise HTTPException(status_code=400, detail="At least 60 price data points are required.")
    if not headline:
        raise HTTPException(status_code=400, detail="A headline is required for sentiment analysis.")
    
    current_day = datetime.utcnow().weekday()  # Monday=0, Sunday=6
    mode = "forex" if current_day < 5 else "other"
    
    signal_id = f"{datetime.utcnow().isoformat()}-{uuid.uuid4().hex[:6]}"
    
    import numpy as np
    if mode == "forex":
        arr = np.array(prices).reshape(-1, 1)
        norm_prices = scaler.transform(arr)
        sequence = norm_prices[-60:].reshape(1, 60, 1)
        lstm_prediction = lstm_model.predict(sequence)[0][0]
        predicted_price = scaler.inverse_transform(np.array([[lstm_prediction]]))[0][0]
    else:
        predicted_price = prices[-1]
    
    sentiment_result = sentiment_pipeline(headline)
    sentiment_label = sentiment_result[0]["label"]
    sentiment_score = sentiment_result[0]["score"]
    
    anomaly_input = np.array(prices[-60:]).reshape(1, -1)
    anomaly_prediction = anomaly_model.predict(anomaly_input)[0]
    anomaly_status = "Anomaly Detected" if anomaly_prediction == 1 else "Normal"
    
    last_price = prices[-1]
    signal = "BUY" if (predicted_price > last_price and sentiment_label.upper() == "POSITIVE") else "SELL"
    
    price_change = abs(predicted_price - last_price)
    confidence = round(sentiment_score * 100 + (price_change / last_price * 100), 2)
    
    return {
        "signal_id": signal_id,
        "mode": mode,
        "predicted_price": predicted_price,
        "signal": signal,
        "confidence": confidence,
        "anomaly": anomaly_status,
        "sentiment": {"label": sentiment_label, "score": sentiment_score},
        "timestamp": datetime.utcnow().isoformat(),
        "entry_price": last_price,
        "info": "Live signal generated based on current market data. For execution, integrate with a broker API."
    }
