import pandas as pd
import numpy as np

def calculate_rsi(data, periods=14):
    """Calculate RSI for the given data."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_rsi_divergence(candles_df, rsi_period=14):
    """
    Check for bullish/bearish RSI divergence.
    candles_df: DataFrame with 'close' column.
    Returns: "bullish", "bearish", or None.
    """
    rsi = calculate_rsi(candles_df['close'], rsi_period)
    price = candles_df['close']

    # Last two pivot points for price and RSI
    price_pivots = price[-5:]  # Last 5 candles for simplicity
    rsi_pivots = rsi[-5:]

    # Bullish divergence: Price makes lower low, RSI makes higher low
    if (price_pivots.iloc[-1] < price_pivots.iloc[-3] and  # Lower low in price
        rsi_pivots.iloc[-1] > rsi_pivots.iloc[-3]):  # Higher low in RSI
        return "bullish"

    # Bearish divergence: Price makes higher high, RSI makes lower high
    if (price_pivots.iloc[-1] > price_pivots.iloc[-3] and  # Higher high in price
        rsi_pivots.iloc[-1] < rsi_pivots.iloc[-3]):  # Lower high in RSI
        return "bearish"

    return None