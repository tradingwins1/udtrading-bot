import pandas as pd

def detect_bos(df, swing_lookback=3):
    """
    Detect Break of Structure (BOS) by identifying swing highs/lows and 
    tracking if price breaks them within a forward window.
    Returns a DataFrame with BOS signals.
    """
    bos_signals = []
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    closes = df['close'].tolist()
    dates = df['Datetime'].tolist() if 'Datetime' in df.columns else df.index.to_list()

    for i in range(swing_lookback, len(df) - swing_lookback):
        is_swing_high = all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, swing_lookback + 1))
        if is_swing_high:
            swing_high = highs[i]
            for k in range(i + 1, min(i + swing_lookback + 10, len(df))):  # Increased forward window
                if closes[k] > swing_high:
                    bos_signals.append({
                        'Date': dates[k],
                        'Type': 'BOS High',
                        'Level': swing_high,
                        'Index': k
                    })
                    break

        is_swing_low = all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, swing_lookback + 1))
        if is_swing_low:
            swing_low = lows[i]
            for k in range(i + 1, min(i + swing_lookback + 10, len(df))):  # Increased forward window
                if closes[k] < swing_low:
                    bos_signals.append({
                        'Date': dates[k],
                        'Type': 'BOS Low',
                        'Level': swing_low,
                        'Index': k
                    })
                    break

    return pd.DataFrame(bos_signals)

def check_bos(df, swing_lookback=3):
    bos_signals = detect_bos(df, swing_lookback)
    if bos_signals.empty:
        return False

    recent_datetimes = df['Datetime'].iloc[-2:].values
    for _, signal in bos_signals.iterrows():
        if pd.to_datetime(signal['Date']) in recent_datetimes:
            return True

    return False

def check_bos_with_retest(df, swing_lookback=3, tolerance=0.5):
    """
    Detect if there's a BOS followed by a retest and a bullish or bearish engulfing candle.
    """
    bos_df = detect_bos(df, swing_lookback)
    if bos_df.empty:
        return False

    last_bos = bos_df.iloc[-1]
    bos_level = last_bos['Level']
    bos_type = last_bos['Type']
    bos_index = last_bos['Index']

    # Search for retest + engulf combo after BOS
    for i in range(bos_index + 1, len(df) - 2):  # allow checking next_candle safely
        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]

        if bos_type == 'BOS High' and abs(candle['low'] - bos_level) <= tolerance:
            if (next_candle['close'] > next_candle['open'] and
                next_candle['close'] > candle['close'] and
                next_candle['open'] <= candle['close']):
                return True

        elif bos_type == 'BOS Low' and abs(candle['high'] - bos_level) <= tolerance:
            if (next_candle['close'] < next_candle['open'] and
                next_candle['close'] < candle['close'] and
                next_candle['open'] >= candle['close']):
                return True

    return False