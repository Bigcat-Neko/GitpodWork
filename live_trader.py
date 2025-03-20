import time
import requests
import numpy as np
import tensorflow as tf
import telebot
from fetch_live_prices import get_live_forex_prices

# API & Telegram Details
TELEGRAM_BOT_TOKEN = "950251170:AAEHwpGH4SKQIG8KgRS6EoHupBX-lZeknlQ"
TELEGRAM_CHAT_ID = "734698844"

# Load AI Models
lstm_model = tf.keras.models.load_model("models/trade_signal_model.h5")
gru_model = tf.keras.models.load_model("models/gru_model.h5")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

def send_signal_to_telegram(signal):
    """Sends a trade signal to Telegram"""
    message = f"🔹 Live Trade Signal: {signal}"
    bot.send_message(TELEGRAM_CHAT_ID, message)

while True:
    try:
        # Step 1: Get live prices
        live_prices = get_live_forex_prices()
        
        # Step 2: AI Predictions
        lstm_prediction = lstm_model.predict(live_prices.reshape(1, live_prices.shape[0], 1))
        gru_prediction = gru_model.predict(live_prices.reshape(1, live_prices.shape[0], 1))
        
        # Step 3: Decision Logic
        if lstm_prediction > 0.5 and gru_prediction > 0.5:
            send_signal_to_telegram("BUY EUR/USD")
        elif lstm_prediction < 0.5 and gru_prediction < 0.5:
            send_signal_to_telegram("SELL EUR/USD")
        else:
            send_signal_to_telegram("NO CLEAR SIGNAL")

        # Step 4: Wait 1 minute before checking again
        time.sleep(60)  

    except Exception as e:
        print("Error:", e)
