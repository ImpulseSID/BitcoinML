import os
import pandas as pd


def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Unix Timestamp to pandas datetime (UTC)."""
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df.set_index("Date", inplace=True)
    return df


def resample_ohlcv(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """
    Resample OHLCV (Open, High, Low, Close, Volume) data to given frequency.
    """
    return df.resample(freq).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to the dataframe."""

    # Simple Moving Averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Exponential Moving Average
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26) with Signal line (9)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20 period, 2 std)
    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = sma_20 + (std_20 * 2)
    df["Bollinger_Lower"] = sma_20 - (std_20 * 2)

    return df


def preprocess_bitcoin_data(resample_choice: str = "daily"):
    # Path to raw dataset
    raw_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "bitcoin-historical-data", "btcusd_1-min_data.csv"
    )
    raw_path = os.path.abspath(raw_path)

    # Load dataset
    df = pd.read_csv(raw_path)

    # Keep only relevant columns
    df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]

    # Convert timestamps
    df = convert_timestamp(df)

    # Map user choice to pandas frequency
    freq_map = {
        "daily": "1D",
        "weekly": "1W",
        "monthly": "MS"
    }

    if resample_choice not in freq_map:
        raise ValueError(f"Invalid choice: {resample_choice}. Choose from {list(freq_map.keys())}.")

    # Resample data
    resampled_df = resample_ohlcv(df, freq=freq_map[resample_choice])

    # Add indicators
    resampled_df = add_technical_indicators(resampled_df)

    # Drop NaN rows caused by indicator lookback periods
    resampled_df.dropna(inplace=True)

    # Save processed data
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    out_path = os.path.join(processed_dir, f"bitcoin_{resample_choice}.csv")
    resampled_df.to_csv(out_path)

    print(f"Processed dataset saved to: {out_path}")
    return resampled_df


if __name__ == "__main__":
    while True:
        choice = input("Resample frequency (daily/weekly/monthly): ").strip().lower()
        if choice in ["daily", "weekly", "monthly"]:
            df = preprocess_bitcoin_data(choice)
            print(df.head())
            break
        else:
            print("Invalid input. Please enter 'daily', 'weekly', or 'monthly'.")
