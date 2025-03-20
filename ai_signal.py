import time
import random
from queue import Queue

# We'll reuse the trade_queue from main.py; import it if in a separate file.
# from main import trade_queue  <-- if needed

def simulate_ai_signals(trade_queue: Queue):
    while True:
        # Simulate waiting for some time (e.g., 30 seconds)
        time.sleep(30)

        # Simulate generating a trade signal.
        # You can replace these values with your AI's output later.
        fake_signal = {
            "trade_type": random.choice(["buy", "sell"]),
            "symbol": random.choice(["BTC/USD", "ETH/USD", "XRP/USD"]),
            "price": round(random.uniform(1000, 60000), 2),
            "amount": round(random.uniform(0.5, 5), 2)
        }
        print("\n[AI Simulation] Generated Trade Signal:", fake_signal)

        # Add the fake signal to the trade processing queue
        trade_queue.put(fake_signal)
