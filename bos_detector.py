import pandas as pd
import numpy as np
from ta.trend import SMAIndicator

def check_bos(df, lookback=3):
    """
    Detect Break of Structure (BOS) in the given DataFrame.
    Parameters:
    - df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and index as datetime
    - lookback: Number of candles to look back for swing highs/lows
    Returns:
    - True if BOS detected, False otherwise
    """
    if len(df) < lookback * 2 + 1:
        return False

    # Ensure column names are lowercase
    df = df.rename(columns=lambda x: x.lower())

    # Check for required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    highs = df['high']
    lows = df['low']
    closes = df['close']

    # Identify swing highs and lows
    is_swing_high = all(highs.iloc[-1] > highs.iloc[-lookback-1:-1]) and all(highs.iloc[-1] > highs.iloc[-lookback-1:-1].shift(-1))
    is_swing_low = all(lows.iloc[-1] < lows.iloc[-lookback-1:-1]) and all(lows.iloc[-1] < lows.iloc[-lookback-1:-1].shift(-1))

    # Detect BOS
    if is_swing_high:
        for i in range(len(df) - lookback - 1, len(df)):
            if closes.iloc[i] > highs.iloc[-1]:
                return True
    elif is_swing_low:
        for i in range(len(df) - lookback - 1, len(df)):
            if closes.iloc[i] < lows.iloc[-1]:
                return True

    return False