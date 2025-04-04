import pandas as pd

def detect_fvg(df):
    """
    Detects Fair Value Gaps (FVG) using 3-candle imbalance logic:
    A FVG exists if:
    - Previous candle's low > next candle's high (bullish gap)
    - Previous candle's high < next candle's low (bearish gap)

    Returns DataFrame with FVG signal info.
    """
    fvg_signals = []

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        next_candle = df.iloc[i + 1]

        # Bullish FVG: gap between prev low and next high
        if prev['low'] > next_candle['high']:
            fvg_signals.append({
                'Date': curr.name,
                'Type': 'Bullish FVG',
                'GapHigh': prev['low'],
                'GapLow': next_candle['high']
            })

        # Bearish FVG: gap between prev high and next low
        elif prev['high'] < next_candle['low']:
            fvg_signals.append({
                'Date': curr.name,
                'Type': 'Bearish FVG',
                'GapHigh': next_candle['low'],
                'GapLow': prev['high']
            })

    return pd.DataFrame(fvg_signals)
