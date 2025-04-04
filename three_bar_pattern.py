
def detect_three_bar_reversal(df, direction='bullish'):
    """
    Detects bullish or bearish 3-bar reversal pattern.
    Returns True if pattern is detected at the latest 3 candles.
    """
    if len(df) < 3:
        return False

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if direction == 'bullish':
        return (
            c1['close'] < c1['open'] and  # red
            c2['low'] < c1['low'] and     # sweep
            c3['close'] > c1['high'] and  # engulf
            c3['close'] > c3['open']      # green
        )
    elif direction == 'bearish':
        return (
            c1['close'] > c1['open'] and
            c2['high'] > c1['high'] and
            c3['close'] < c1['low'] and
            c3['close'] < c3['open']
        )
    return False
