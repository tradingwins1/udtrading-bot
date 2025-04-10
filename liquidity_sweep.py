def check_liquidity_sweep(candles_df):
    """
    Detect a liquidity sweep with reclaim.
    candles_df: DataFrame with 'open', 'high', 'low', 'close'.
    Returns: True if a liquidity sweep with reclaim is detected, else False.
    """
    if len(candles_df) < 3:
        return False

    prev_candle = candles_df.iloc[-2]
    last_candle = candles_df.iloc[-1]

    # Bullish sweep: Price sweeps below a low, then reclaims
    if (prev_candle['low'] < candles_df['low'][-5:-2].min() and  # Sweeps below recent low
        last_candle['close'] > prev_candle['high']):  # Reclaims above the sweep candle
        return True

    # Bearish sweep: Price sweeps above a high, then reclaims
    if (prev_candle['high'] > candles_df['high'][-5:-2].max() and  # Sweeps above recent high
        last_candle['close'] < prev_candle['low']):  # Reclaims below the sweep candle
        return True

    return False