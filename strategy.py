from datetime import datetime, timedelta
import pytz

# === Red Folder News Filter ===
def get_mock_red_news_events():
    now = datetime.utcnow()
    return [
        now.replace(hour=12, minute=30, second=0, microsecond=0),
        now.replace(hour=14, minute=0, second=0, microsecond=0),
    ]

def is_high_impact_news_nearby(trade_time_utc, window_minutes=15):
    news_events = get_mock_red_news_events()
    for event_time in news_events:
        event_naive = event_time.replace(tzinfo=None)
        trade_naive = trade_time_utc.replace(tzinfo=None)
        if abs((event_naive - trade_naive).total_seconds()) <= window_minutes * 60:
            return True
    return False

# === Wick-to-Body Ratio ===
def wick_to_body_ratio(candle):
    high = candle['high']
    low = candle['low']
    open_price = candle['open']
    close_price = candle['close']
    body = abs(close_price - open_price)
    wick = high - low
    return wick / body if body != 0 else float('inf')

# === Volume Spike Filter ===
def detect_volume_spike(df, i, multiplier=1.5):
    if i < 10:
        return False
    avg_volume = df.iloc[i-10:i]['volume'].mean()
    return df.iloc[i]['volume'] > avg_volume * multiplier

# === Wick-to-Body Validation ===
def is_valid_wick_body_candle(open_price, close_price, high_price, low_price, threshold=0.6):
    body = abs(close_price - open_price)
    total_range = high_price - low_price
    if total_range == 0:
        return False
    return (body / total_range) >= threshold
def is_near_pdh_or_pdl(df, entry_price, threshold=2.0):
    """
    Returns True if entry is within `threshold` points of previous day high or low
    """
    if len(df) < 96:  # Not enough data to calculate PDH/PDL
        return False

    pdh = df['high'].shift(1).rolling(96).max().iloc[-1]
    pdl = df['low'].shift(1).rolling(96).min().iloc[-1]

    return abs(entry_price - pdh) < threshold or abs(entry_price - pdl) < threshold
