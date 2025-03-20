import requests
import pandas as pd
import os
from datetime import datetime

# Trading Economics guest endpoint for economic calendar data
ECONOMIC_URL = "https://api.tradingeconomics.com/calendar?c=guest:guest"

def fetch_economic_events():
    response = requests.get(ECONOMIC_URL)
    try:
        data = response.json()
    except Exception as e:
        print("Error decoding JSON:", e)
        return

    # Check if the data is a non-empty list
    if isinstance(data, list) and data:
        df = pd.DataFrame(data)
        # Add a timestamp to know when the data was fetched
        df["fetched_at"] = datetime.utcnow().isoformat()
        
        csv_file = "data/economic_events.csv"
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        print("Economic events data updated and appended to 'data/economic_events.csv'.")
    else:
        print("Error fetching economic events:", data)

if __name__ == '__main__':
    fetch_economic_events()
