import pandas as pd

# Load your training data (assumed to have columns: "datetime", "open", "high", "low", "close", "asset")
data = pd.read_csv("training_data.csv", parse_dates=["datetime"])
data.sort_values("datetime", inplace=True)

# Create a new feature - a simple moving average of the closing price over 5 days
data["SMA_5"] = data["close"].rolling(window=5).mean()

# Drop any rows with missing values (which may appear due to the rolling window)
data.dropna(inplace=True)

# Save the processed data to a new CSV
data.to_csv("processed_data.csv", index=False)

# Print the first few rows to verify the results
print(data.head())
