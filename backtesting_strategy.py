from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np

# Helper function to calculate a Simple Moving Average (SMA)
def SMA(series, window):
    return pd.Series(series).rolling(window=window).mean().to_numpy()

# Define the strategy with fixed fractional trading simulation
class SmaCross(Strategy):
    short_window = 5
    long_window = 20

    def init(self):
        # Register the SMA indicator using self.I()
        self.sma_short = self.I(SMA, self.data.Close, self.short_window)
        self.sma_long = self.I(SMA, self.data.Close, self.long_window)

    def next(self):
        # Ensure we have enough data points before checking for a crossover
        if len(self.sma_short) < 2 or len(self.sma_long) < 2:
            return

        current_price = self.data.Close[-1]
        # Calculate trade size as 10% of current equity divided by current price.
        # This gives a fraction representing the portion of equity to use.
        size = (self.equity * 0.1) / current_price

        # Ensure the size is a positive fraction (i.e. between 0 and 1)
        if size <= 0:
            return
        if size >= 1:
            # If computed size is 1 or more, override with a default fraction (e.g., 0.5)
            size = 0.5

        # Buy signal: short-term SMA crosses above long-term SMA
        if self.sma_short[-1] > self.sma_long[-1] and self.sma_short[-2] <= self.sma_long[-2]:
            self.buy(size=size)
        # Sell signal: short-term SMA crosses below long-term SMA
        elif self.sma_short[-1] < self.sma_long[-1] and self.sma_short[-2] >= self.sma_long[-2]:
            self.sell(size=size)

# Load your processed data
data = pd.read_csv("processed_data.csv", parse_dates=["datetime"])
data.set_index("datetime", inplace=True)

# Rename columns to match the backtesting package expectations (capitalized OHLC)
data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

# Scale down the prices to simulate fractional trading.
# Adjust the scale_factor as needed (if prices are in thousands, using 1000 is a good starting point)
scale_factor = 1000
data[['Open','High','Low','Close']] = data[['Open','High','Low','Close']] / scale_factor

# Initialize and run the backtest with a low initial cash value (e.g., $10)
bt = Backtest(data, SmaCross, cash=1000000, commission=0.0002)
stats = bt.run()
bt.plot()
print(stats)
