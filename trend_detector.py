def determine_trend(candles_df, lookback=10):
    """
    Determine the trend using moving averages.
    candles_df: DataFrame with 'close' column.
    Returns: "up", "down", or "neutral".
    """
    short_ma = candles_df['close'].rolling(window=lookback // 2).mean()
    long_ma = candles_df['close'].rolling(window=lookback).mean()

    if short_ma.iloc[-1] > long_ma.iloc[-1]:
        return "up"
    elif short_ma.iloc[-1] < long_ma.iloc[-1]:
        return "down"
    return "neutral"
