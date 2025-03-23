import os
import pandas as pd

# Ensure the data folder exists and the file is present
csv_path = "data/training.1600000.processed.noemoticon.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}. Please download the Sentiment140 dataset from Kaggle and place it here.")

# The original file has no header, so we manually specify the column names.
columns = ["target", "ids", "date", "flag", "user", "text"]
df = pd.read_csv(csv_path, encoding="latin-1", names=columns)

# Filter out neutral tweets (target == 2) and keep only negative (0) and positive (4)
df = df[df["target"] != 2]

# Map target values: 0 stays 0 (negative), 4 becomes 1 (positive)
df["sentiment"] = df["target"].map({0: 0, 4: 1})

# Keep only the columns we need: text and sentiment
df_clean = df[["text", "sentiment"]].copy()

# Optionally, clean the text (for instance, lowercase it)
df_clean["text"] = df_clean["text"].str.lower()

# Save the cleaned data to a new CSV
output_path = "data/sentiment_data.csv"
df_clean.to_csv(output_path, index=False)
print(f"Cleaned sentiment data saved as {output_path}")
