import pandas as pd
def calculate_avg_range(candles_df, lookback=10):
    ranges = candles_df['high'] - candles_df['low']
    return ranges[-lookback:].mean()

def calculate_atr(candles_df, period=14):
    high_low = candles_df['high'] - candles_df['low']
    high_close = abs(candles_df['high'] - candles_df['close'].shift())
    low_close = abs(candles_df['low'] - candles_df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range[-period:].mean()

def check_3_bar_pattern(candles_df):
    if len(candles_df) < 3:
        return None
    bar1, bar2, bar3 = candles_df.iloc[-3], candles_df.iloc[-2], candles_df.iloc[-1]

    if (bar1['close'] < bar1['open'] and
        bar2['close'] < bar2['open'] and
        bar3['close'] > bar3['open'] and
        bar3['close'] > bar2['high'] and
        (bar3['high'] - bar3['low']) > (bar2['high'] - bar2['low'])):
        return "bullish"

    if (bar1['close'] > bar1['open'] and
        bar2['close'] > bar2['open'] and
        bar3['close'] < bar3['open'] and
        bar3['close'] < bar2['low'] and
        (bar3['high'] - bar3['low']) > (bar2['high'] - bar2['low'])):
        return "bearish"

    return None

def check_pop_and_fade(candles_df, trend):
    if trend != "down" or len(candles_df) < 3:
        return False
    last_candle = candles_df.iloc[-1]
    pop_candle = candles_df.iloc[-2]
    volume_spike = pop_candle['volume'] > candles_df['volume'][:-2].mean() * 1.2
    pop_bullish = pop_candle['close'] > pop_candle['open']
    fade_detected = last_candle['close'] < pop_candle['close']
    return volume_spike and pop_bullish and fade_detected

def check_choch(candles_df, trend, lookback=20, required_swings=3, volume_filter=False, htf_trend=None, volume_multiplier=1.2):
    if len(candles_df) < lookback + 1:
        return False, None

    last_candle = candles_df.iloc[-1]
    lookback_df = candles_df.iloc[-(lookback + 1):-1]

    def find_swing_highs(df):
        swing_highs = []
        for i in range(2, len(df) - 2):
            if df['high'].iloc[i] > df['high'].iloc[i - 2] and df['high'].iloc[i] > df['high'].iloc[i + 2]:
                swing_highs.append((df['Datetime'].iloc[i], df['high'].iloc[i]))
        return sorted(swing_highs, key=lambda x: x[0], reverse=True)

    def find_swing_lows(df):
        swing_lows = []
        for i in range(2, len(df) - 2):
            if df['low'].iloc[i] < df['low'].iloc[i - 2] and df['low'].iloc[i] < df['low'].iloc[i + 2]:
                swing_lows.append((df['Datetime'].iloc[i], df['low'].iloc[i]))
        return sorted(swing_lows, key=lambda x: x[0], reverse=True)

    if trend == "down":
        swing_highs = find_swing_highs(lookback_df)
        print("Swing Highs:", swing_highs)
        if len(swing_highs) < required_swings:
            return False, None
        nth_swing_high = swing_highs[required_swings - 1][1]

        if last_candle['close'] > nth_swing_high:
            if volume_filter:
                recent_vol = candles_df['volume'].iloc[-1]
                avg_vol = candles_df['volume'].iloc[-10:-1].mean()
                if recent_vol < avg_vol * volume_multiplier:
                    print("❌ Volume filter failed")
                    return False, None

            if htf_trend and htf_trend != "up":
                print("❌ HTF trend mismatch for bullish CHOCH")
                return False, None

            print(f"CHOCH triggered: close {last_candle['close']} > swing high {nth_swing_high}")
            return True, "bullish"

        return False, None

    elif trend == "up":
        swing_lows = find_swing_lows(lookback_df)
        print("Swing Lows:", swing_lows)
        if len(swing_lows) < required_swings:
            return False, None
        nth_swing_low = swing_lows[required_swings - 1][1]

        if last_candle['close'] < nth_swing_low:
            if volume_filter:
                recent_vol = candles_df['volume'].iloc[-1]
                avg_vol = candles_df['volume'].iloc[-10:-1].mean()
                if recent_vol < avg_vol * volume_multiplier:
                    print("❌ Volume filter failed")
                    return False, None

            if htf_trend and htf_trend != "down":
                print("❌ HTF trend mismatch for bearish CHOCH")
                return False, None

            print(f"CHOCH triggered: close {last_candle['close']} < swing low {nth_swing_low}")
            return True, "bearish"

        return False, None

    return False, None