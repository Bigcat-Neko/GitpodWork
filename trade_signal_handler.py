import queue
import time
from threading import Thread
from app.database import SessionLocal
from app.models import Trade
from utils import send_telegram_message

# Trade processing queue
trade_queue = queue.Queue()

# ✅ Background worker function
def process_trades():
    while True:
        try:
            trade_data = trade_queue.get()  # Get trade from queue
            if trade_data is None:
                continue

            # Simulate trade execution
            print(f"✅ Executing Trade: {trade_data}")

            # Save trade to database
            db = SessionLocal()
            trade = Trade(**trade_data)
            db.add(trade)
            db.commit()
            db.refresh(trade)
            db.close()

            # Send Telegram notification
            message = f"✅ Executed Trade: {trade.trade_type.upper()} {trade.amount} {trade.symbol} at ${trade.price}"
            send_telegram_message(message)

        except Exception as e:
            print(f"⚠️ Error processing trade: {e}")

        time.sleep(2)  # Prevent CPU overload

# ✅ Start the background trade processing thread
worker_thread = Thread(target=process_trades, daemon=True)
worker_thread.start()
