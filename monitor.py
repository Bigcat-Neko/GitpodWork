# monitor.py
from sklearn.metrics import mean_absolute_error

def monitor_performance(actual_prices, predicted_prices):
    error = mean_absolute_error(actual_prices, predicted_prices)
    threshold = actual_prices.mean() * 0.02  # 2% of average price
    
    if error > threshold:
        print(f"Model drift detected (MAE: {error:.4f}), initiating retrain...")
        # Trigger retraining workflow