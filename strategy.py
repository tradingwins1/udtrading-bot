# === strategy.py (Enhanced with New Confluences) ===
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

# === PDH/PDL Trap Check ===
def is_near_pdh_or_pdl(df, entry_price, threshold=2.0):
    if len(df) < 96:
        return False
    pdh = df['high'].shift(1).rolling(96).max().iloc[-1]
    pdl = df['low'].shift(1).rolling(96).min().iloc[-1]
    return abs(entry_price - pdh) < threshold or abs(entry_price - pdl) < threshold

# === Reclaim Zone Confirmation ===
def reclaim_zone_confirmed(candle, zone):
    return candle['low'] < zone and candle['close'] > zone

# === 84% Rule Check ===
def rule_of_84_entry(breakout_candle, current_candle):
    range_size = breakout_candle['high'] - breakout_candle['low']
    pullback = breakout_candle['high'] - current_candle['low']
    return pullback >= 0.84 * range_size and current_candle['close'] > current_candle['open']

# === 5-Min Range Break + Retest ===
def opening_range_breakout_retest(range_high, range_low, current_candle):
    return (
        current_candle['low'] > range_low and
        current_candle['open'] > range_low and
        current_candle['close'] > range_high
    )

# === Entry Scoring ===
def calculate_entry_score(data):
    score = 0
    if data.get('htf_zone'): score += 2
    if data.get('reclaim_zone'): score += 2
    if data.get('volume_spike'): score += 2
    if data.get('opening_range'): score += 2
    if data.get('rule_84'): score += 2
    return score

# === Strategy Runner Hook ===
def run_strategy(market_data):
    # === Confluence: Avoid near PMH/PML unless reclaimed (Scalping Focus) ===
    near_pmh = abs(market_data['entry'] - market_data['pmh']) < 2.0
    near_pml = abs(market_data['entry'] - market_data['pml']) < 2.0
    reclaimed = market_data.get('reclaim_zone', False)
    if (near_pmh or near_pml) and not reclaimed:
        return None
    if is_high_impact_news_nearby(datetime.utcnow()):
        return None
    score = calculate_entry_score(market_data)
    if score < 6:
        return None
    if not is_valid_wick_body_candle(market_data['open'], market_data['close'], market_data['high'], market_data['low']):
        return None
    # === Dynamic SL/TP for Swing Trades ===
    if market_data['bias'] == 'bullish':
        sl = market_data['pml'] - 5
        tp = market_data['pmh'] + 15
    else:
        sl = market_data['pmh'] + 5
        tp = market_data['pml'] - 15

    # === SleepMode: Overnight Trade Adjustments ===
    if market_data.get('sleep_mode'):
        partial_profit = (tp - market_data['entry']) * 0.5
        sl = market_data['pml'] - 2  # Tighter stop overnight
        tp = market_data['entry'] + partial_profit

    return {
        'action': 'buy' if market_data['bias'] == 'bullish' else 'sell',
        'confidence': score,
        'entry': market_data['entry'],
        'sl': sl,
        'tp': tp
    }

# === Scheduler Integration (Example Usage with PML/PMH) ===
def run_single_asset(df, symbol):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    market_data = {
        'open': last['open'],
        'close': last['close'],
        'high': last['high'],
        'low': last['low'],
        'volume_spike': detect_volume_spike(df, len(df)-1),
        'rule_84': rule_of_84_entry(prev, last),
        'reclaim_zone': reclaim_zone_confirmed(last, zone=last['low']),
        'opening_range': opening_range_breakout_retest(
            range_high=last['high'] - 20,
            range_low=last['low'] + 20,
            current_candle=last
        ),
        'htf_zone': True,
        'bias': 'bullish',
        'entry': last['close'],
        'sl': last['low'] - 25,
        'tp': last['close'] + 50,
        'pml': df['low'].min(),
        'pmh': df['high'].max(),
        'sleep_mode': True  # Flag for Sleep-Friendly Logic
    }

    signal = run_strategy(market_data)
    if signal:
        print(f"[{symbol}] Entry confirmed →", signal)
        with open(f"logs/{symbol}_signals.log", "a") as log:
            log.write(f"{datetime.utcnow().isoformat()} | {symbol} | {signal}\n")
        return signal
    return None
