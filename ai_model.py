# ai_model.py
def generate_trade_signal(market_data: dict) -> str:
    """
    Dummy AI model that returns a trade signal based on market data.
    For this demo, if 'feature1' > 50, return "buy", else "sell".
    """
    if market_data.get("feature1", 0) > 50:
        return "buy"
    else:
        return "sell"
