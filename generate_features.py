import pandas as pd
import ta

def calculate_indicators(df):
    """
    Calculates technical indicators for a given price DataFrame.
    """
    # Calculate indicators – adjust window sizes if needed
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["close"]).stochrsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci()
    df["mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"]).money_flow_index()
    df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"]).chaikin_money_flow()
    df = df.dropna()
    return df

if __name__ == "__main__":
    # Load the raw market data (ensure market_data.csv exists and has the proper columns)
    df = pd.read_csv("market_data.csv", sep=";")  # Adjust separator if needed
    df = calculate_indicators(df)
    df.to_csv("processed_market_data.csv", index=False)
    print("✅ Market data processed and saved as 'processed_market_data.csv'.")
