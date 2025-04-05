# backtester.py
import backtrader as bt

class ArimaStrategy(bt.Strategy):
    params = (
        ('arima_order', (2,1,2)),
        ('risk_per_trade', 0.01),
    )

    def __init__(self):
        self.arima_signal = self.data.arima_pred
        
    def next(self):
        if self.arima_signal[0] > self.data.close[0] * 1.005:
            self.buy(size=self.position_size())
            
        if self.arima_signal[0] < self.data.close[0] * 0.995:
            self.sell(size=self.position_size())

# Add ARIMA predictions to your data feed