# Import necessary libraries
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import random
import os
from datetime import datetime
from fastapi import FastAPI
import requests

# Initialize FastAPI
app = FastAPI()

# MT5 Connection Configuration
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")

# Connect to MT5
def connect_mt5():
    if not mt5.initialize():
        print("❌ MT5 Initialization failed. Error:", mt5.last_error())
        return False

    authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        print("❌ MT5 Login failed. Error:", mt5.last_error())
        return False

    print("✅ MT5 Connected Successfully!")
    return True

# Function to check if an order can be placed
def validate_trade(symbol, volume):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found!")
        return False
    if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
        print(f"❌ Invalid Volume: {volume}. Allowed: {symbol_info.volume_min} - {symbol_info.volume_max}")
        return False
    return True

# Function to place a trade
def place_trade(symbol, trade_type, volume, sl, tp):
    if not validate_trade(symbol, volume):
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": trade_type,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": "AI Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ MT5 Trade failed! Error: {result.comment}")
        return None

    print(f"✅ Trade placed: {symbol} - {trade_type} @ {result.price_open}")
    return result

# AI Trade Monitoring & Management
def monitor_trades():
    trades = mt5.positions_get()
    for trade in trades:
        symbol = trade.symbol
        profit = trade.profit

        if profit > 10:
            print(f"✅ Closing trade {trade.ticket} for profit: ${profit}")
            mt5.Close(trade.ticket)

        if profit < -10:
            print(f"⚠️ Trade {trade.ticket} in loss: ${profit}. Monitoring closely.")

# Background Task for AI Trading
def ai_trading():
    while True:
        if connect_mt5():
            symbol = "EURUSD"
            trade_type = random.choice([mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL])
            volume = 0.1  # Default lot size
            sl = None  # Auto SL
            tp = None  # Auto TP

            result = place_trade(symbol, trade_type, volume, sl, tp)
            if result:
                print("✅ Trade executed successfully!")

            time.sleep(10)  # Wait before placing the next trade

# Start AI Trading
if __name__ == "__main__":
    ai_trading()
