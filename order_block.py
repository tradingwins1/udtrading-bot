def check_order_block(candles_df):
    """
    Detect an order block (SMC concept).
    candles_df: DataFrame with 'open', 'high', 'low', 'close'.
    Returns: True if an order block is detected, else False.
    """
    # Simplified: Look for a strong candle followed by a retest
    last_candle = candles_df.iloc[-1]
    prev_candle = candles_df.iloc[-2]

    # Bullish order block: Strong bearish candle followed by a bullish retest
    if (prev_candle['close'] < prev_candle['open'] and  # Strong bearish candle
        last_candle['close'] > last_candle['open'] and  # Bullish retest
        last_candle['low'] <= prev_candle['close']):  # Retests the bearish candle's close
        return True

    # Bearish order block: Strong bullish candle followed by a bearish retest
    if (prev_candle['close'] > prev_candle['open'] and  # Strong bullish candle
        last_candle['close'] < last_candle['open'] and  # Bearish retest
        last_candle['high'] >= prev_candle['close']):  # Retests the bullish candle's close
        return True

    return False