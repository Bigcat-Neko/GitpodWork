# forecast.py
import joblib
import pandas as pd

def generate_forecasts(model_file="models/all_arima_models.pkl", days=7):
    """Generate forecasts for all assets"""
    models = joblib.load(model_file)
    forecasts = {}
    
    for asset, data in models.items():
        try:
            forecast = data['model'].forecast(steps=days)
            forecasts[asset] = {
                'predictions': forecast.tolist(),
                'order': data['order'],
                'last_trained': data['last_trained']
            }
        except Exception as e:
            print(f"Forecast failed for {asset}: {str(e)}")
    
    return forecasts

# Usage
if __name__ == "__main__":
    forecasts = generate_forecasts()
    print(pd.DataFrame(forecasts).T)