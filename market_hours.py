def is_forex_market_open():
    from datetime import datetime
    now = datetime.utcnow()
    return now.weekday() < 5 and 0 <= now.hour < 22
