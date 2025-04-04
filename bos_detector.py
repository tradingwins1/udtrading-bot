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
    dates = df.index.to_list()


    for i in range(swing_lookback, len(df) - swing_lookback):
        # --- Detect Swing High ---
        is_swing_high = all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, swing_lookback + 1))
        if is_swing_high:
            swing_high = highs[i]
            for k in range(i + 1, min(i + swing_lookback + 5, len(df))):
                if closes[k] > swing_high:
                    bos_signals.append({
                        'Date': dates[k],
                        'Type': 'BOS High',
                        'Level': swing_high
                    })
                    break

        # --- Detect Swing Low ---
        is_swing_low = all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, swing_lookback + 1))
        if is_swing_low:
            swing_low = lows[i]
            for k in range(i + 1, min(i + swing_lookback + 5, len(df))):
                if closes[k] < swing_low:
                    bos_signals.append({
                        'Date': dates[k],
                        'Type': 'BOS Low',
                        'Level': swing_low
                    })
                    break

    return pd.DataFrame(bos_signals)
