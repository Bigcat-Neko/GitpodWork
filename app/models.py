# app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    trade_type = Column(String, nullable=False)  # e.g., Buy or Sell
    symbol = Column(String, nullable=False)       # e.g., BTC/USD
    price = Column(Float, nullable=False)         # Execution price
    amount = Column(Float, nullable=False)        # Trade size
    confidence = Column(Float, default=0)           # Confidence level (added later)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)  # Trade time
