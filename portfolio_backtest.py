import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt

# Step 1: Load your training data.
df = pd.read_csv("training_data.csv", parse_dates=["datetime"])
df.set_index("datetime", inplace=True)

# Step 2: Pivot the data so that each asset gets its own column of closing prices.
price = df.pivot_table(index=df.index, columns="asset", values="close")

# Step 3: Calculate moving averages for each asset.
short_window = 5
long_window = 20
sma_short = price.rolling(window=short_window).mean()
sma_long = price.rolling(window=long_window).mean()

# Step 4: Generate trading signals.
# We create a buy signal when the short moving average is above the long moving average.
# A sell signal is generated when it goes below.
entries = sma_short > sma_long
exits = sma_short < sma_long

# Step 5: Run the portfolio simulation.
# We start with a small cash value (e.g., $10) and let fractional trading handle small amounts.
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=10,
    fees=0.002  # 0.2% commission per trade
)

# Step 6: Show performance metrics and plot the equity curve.
print(portfolio.stats())
portfolio.total_cash().vbt.plot(title="Portfolio Equity Curve")
plt.show()
