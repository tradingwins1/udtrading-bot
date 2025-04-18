import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, SMAIndicator, MACD
from gap_logic import detect_gap_fill_reversal
import logging
import pytz

logger = logging.getLogger(__name__)

def detect_ug_signals(tsla_5min_data, tsla_1min_data, key_levels, lookback=3, volume_factor=1.05, wick_body_ratio=0.3, fib_84_level=0.84):
    """
    Detect UG Trading Bot signals with prioritized confluences on 5-minute TSLA data, using 1-minute data for entry.
    """
    signals = []
    logger.debug("Starting signal detection for 5min data with shape: %s, 1min data with shape: %s", tsla_5min_data.shape, tsla_1min_data.shape)
    
    df_5min = tsla_5min_data.copy()
    df_1min = tsla_1min_data.copy()
    if not all(col in df_5min.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        logger.error("5min DataFrame missing required OHLCV columns")
        raise ValueError("5min DataFrame missing required OHLCV columns")
    if not all(col in df_1min.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        logger.error("1min DataFrame missing required OHLCV columns")
        raise ValueError("1min DataFrame missing required OHLCV columns")
    
    try:
        df_5min['ATR'] = AverageTrueRange(df_5min['High'], df_5min['Low'], df_5min['Close'], window=14).average_true_range()
        df_5min['VWAP'] = VolumeWeightedAveragePrice(df_5min['High'], df_5min['Low'], df_5min['Close'], df_5min['Volume'], window=14).volume_weighted_average_price()
        df_5min['RSI'] = RSIIndicator(df_5min['Close'], window=14).rsi()
        df_5min['SMA20'] = SMAIndicator(df_5min['Close'], window=20).sma_indicator()
        df_5min['SMA50'] = SMAIndicator(df_5min['Close'], window=50).sma_indicator()
        df_5min['SMA200'] = SMAIndicator(df_5min['Close'], window=200).sma_indicator()
        df_5min['MACD'] = MACD(df_5min['Close']).macd()
        df_5min['MACD_Signal'] = MACD(df_5min['Close']).macd_signal()
        h1_data = df_5min.resample('1H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        h1_data['ADX'] = ADXIndicator(h1_data['High'], h1_data['Low'], h1_data['Close'], window=14).adx()
        df_5min['ADX_1H'] = h1_data['ADX'].reindex(df_5min.index, method='ffill')

        df_1min['ATR'] = AverageTrueRange(df_1min['High'], df_1min['Low'], df_1min['Close'], window=14).average_true_range()
        df_1min['RSI'] = RSIIndicator(df_1min['Close'], window=14).rsi()
        logger.debug("Indicators calculated successfully")
    except Exception as e:
        logger.error("Error calculating indicators: %s", e)
        raise

    if df_5min.index.tz is None:
        df_5min.index = df_5min.index.tz_localize('US/Eastern')
    else:
        df_5min.index = df_5min.index.tz_convert('US/Eastern')
    if df_1min.index.tz is None:
        df_1min.index = df_1min.index.tz_localize('US/Eastern')
    else:
        df_1min.index = df_1min.index.tz_convert('US/Eastern')
    logger.debug("Timestamps localized to US/Eastern: 5min %s, 1min %s", df_5min.index[:5].tolist(), df_1min.index[:5].tolist())

    df_5min['avg_volume'] = df_5min['Volume'].shift(1).rolling(14).mean()
    df_5min['is_volume_spike'] = df_5min['Volume'] > df_5min['avg_volume'] * volume_factor
    df_1min['avg_volume'] = df_1min['Volume'].shift(1).rolling(14).mean()
    df_1min['is_volume_spike'] = df_1min['Volume'] > df_1min['avg_volume'] * volume_factor

    df_5min['body'] = np.abs(df_5min['Close'] - df_5min['Open'])
    df_5min['wick_high'] = df_5min['High'] - np.maximum(df_5min['Open'], df_5min['Close'])
    df_5min['wick_low'] = np.minimum(df_5min['Open'], df_5min['Close']) - df_5min['Low']
    df_5min['is_sweep_high'] = (df_5min['wick_high'] > df_5min['body'] * wick_body_ratio)
    df_5min['is_sweep_low'] = (df_5min['wick_low'] > df_5min['body'] * wick_body_ratio)

    df_1min['body'] = np.abs(df_1min['Close'] - df_1min['Open'])
    df_1min['wick_high'] = df_1min['High'] - np.maximum(df_1min['Open'], df_1min['Close'])
    df_1min['wick_low'] = np.minimum(df_1min['Open'], df_1min['Close']) - df_1min['Low']
    df_1min['is_sweep_high'] = (df_1min['wick_high'] > df_1min['body'] * wick_body_ratio)
    df_1min['is_sweep_low'] = (df_1min['wick_low'] > df_1min['body'] * wick_body_ratio)

    df_5min['time'] = df_5min.index.time
    df_5min['is_ny_open'] = (df_5min['time'] >= pd.Timestamp('09:30').time()) & (df_5min['time'] <= pd.Timestamp('16:00').time())
    df_1min['time'] = df_1min.index.time
    df_1min['is_ny_open'] = (df_1min['time'] >= pd.Timestamp('09:30').time()) & (df_1min['time'] <= pd.Timestamp('16:00').time())

    daily_groups_5min = df_5min.groupby(df_5min.index.date)
    daily_groups_1min = df_1min.groupby(df_1min.index.date)
    for day in daily_groups_5min.groups.keys():
        # Skip if choppy conditions (ADX < 25 on 1H)
        day_data_5min = daily_groups_5min.get_group(day)
        adx = day_data_5min['ADX_1H'].iloc[-1] if not day_data_5min['ADX_1H'].empty else 0
        if adx < 25:
            logger.debug("Skipping day %s due to choppy conditions: ADX=%.2f", day, adx)
            continue

        # Skip during news events
        from red_news_filter import is_red_folder_event_today
        if is_red_folder_event_today():
            logger.debug("Skipping day %s due to news event", day)
            continue

        # Mark 5-minute and 1-minute ranges (9:30 AM to 9:34 AM)
        day_data_1min = daily_groups_1min.get_group(day)
        opening_range_5min = day_data_5min.between_time('09:30', '09:34')
        range_high_5min = opening_range_5min['High'].max() if not opening_range_5min.empty else np.nan
        range_low_5min = opening_range_5min['Low'].min() if not opening_range_5min.empty else np.nan

        opening_range_1min = day_data_1min.between_time('09:30', '09:34')
        range_high_1min = opening_range_1min['High'].max() if not opening_range_1min.empty else np.nan
        range_low_1min = opening_range_1min['Low'].min() if not opening_range_1min.empty else np.nan

        logger.debug("Day %s: 5min Opening range high=%.2f, low=%.2f; 1min Opening range high=%.2f, low=%.2f", 
                     day, range_high_5min, range_low_5min, range_high_1min, range_low_1min)

        # Wait until 9:34 AM
        trading_data_5min = day_data_5min[day_data_5min.index.time >= pd.Timestamp('09:34').time()]
        trading_data_1min = day_data_1min[day_data_1min.index.time >= pd.Timestamp('09:34').time()]
        for i in range(lookback, len(trading_data_5min) - lookback):
            try:
                if not trading_data_5min['is_ny_open'].iloc[i]:
                    continue

                bar_5min = trading_data_5min.iloc[i]
                timestamp = bar_5min.name
                day = timestamp.date()

                # Get key levels for the day
                levels = key_levels.get(day, {})
                pml = levels.get('PML', np.nan)
                pmh = levels.get('PMH', np.nan)
                pdl = levels.get('PDL', np.nan)
                pdh = levels.get('PDH', np.nan)
                kpl = round(bar_5min['Close'] / 0.5) * 0.5

                # 3 Bar Pattern on 5min
                if i >= 2:
                    bar_1_bearish = trading_data_5min['Close'].iloc[i-2] < trading_data_5min['Open'].iloc[i-2]
                    bar_2_bullish = trading_data_5min['Close'].iloc[i-1] > trading_data_5min['Open'].iloc[i-1]
                    bar_3_bullish = bar_5min['Close'] > bar_5min['Open']
                    is_3_bar_pattern = bar_1_bearish and bar_2_bullish and bar_3_bullish
                else:
                    is_3_bar_pattern = False
                logger.debug("3 Bar Pattern at bar %d: %s", i, is_3_bar_pattern)

                # RSI Confirmation
                rsi_overbought = bar_5min['RSI'] > 70
                rsi_oversold = bar_5min['RSI'] < 30

                # MACD Confirmation
                macd_bullish = bar_5min['MACD'] > bar_5min['MACD_Signal'] and bar_5min['MACD'] > 0
                macd_bearish = bar_5min['MACD'] < bar_5min['MACD_Signal'] and bar_5min['MACD'] < 0

                # Trend Check
                is_downtrend = bar_5min['SMA50'] < bar_5min['SMA200']
                is_uptrend = bar_5min['SMA50'] > bar_5min['SMA200']

                # Check proximity for cases
                proximity_threshold = 0.01 * bar_5min['Close']  # 1% of current price
                cases = {
                    'Case 1': True,  # Direct Break and Retest
                    'Case 2': pml and abs(pml - range_low_5min) <= proximity_threshold,
                    'Case 3': pml and abs(pml - range_high_5min) <= proximity_threshold,
                    'Case 4': pmh and abs(pmh - range_low_5min) <= proximity_threshold,
                    'Case 5': pmh and abs(pmh - range_high_5min) <= proximity_threshold,
                    'Case 6': pdl and abs(pdl - range_low_5min) <= proximity_threshold,
                    'Case 7': pdl and abs(pdl - range_high_5min) <= proximity_threshold,
                    'Case 8': pdh and abs(pdh - range_low_5min) <= proximity_threshold,
                    'Case 9': pdh and abs(pdh - range_high_5min) <= proximity_threshold
                }

                # Priority 1: Break and Retest (App 1)
                if i > 0:
                    # Check break on 1-minute data
                    break_detected_1min = False
                    break_direction_1min = None
                    break_candle_1min = None
                    window_1min = trading_data_1min[trading_data_1min.index <= timestamp][-5:]  # Last 5 minutes
                    for j in range(len(window_1min)):
                        bar_1min = window_1min.iloc[j]
                        break_candle_body_1min = abs(bar_1min['Close'] - bar_1min['Open'])
                        displacement_condition_up_1min = (bar_1min['Close'] > range_high_1min and break_candle_body_1min > 2 * bar_1min['ATR'])
                        displacement_condition_down_1min = (bar_1min['Close'] < range_low_1min and break_candle_body_1min > 2 * bar_1min['ATR'])
                        if displacement_condition_up_1min:
                            break_detected_1min = True
                            break_direction_1min = 'long'
                            break_candle_1min = bar_1min
                            break
                        if displacement_condition_down_1min:
                            break_detected_1min = True
                            break_direction_1min = 'short'
                            break_candle_1min = bar_1min
                            break

                    if break_detected_1min:
                        # Confirm retest on either 1-minute or 5-minute data
                        retest_confirmed = False
                        retest_timestamp = None
                        strong_bullish_1min = strong_bearish_1min = wick_rejection_up_1min = wick_rejection_down_1min = False
                        strong_bullish_5min = strong_bearish_5min = wick_rejection_up_5min = wick_rejection_down_5min = False

                        # Check 1-minute retest
                        for j in range(len(window_1min)):
                            bar_1min = window_1min.iloc[j]
                            if break_direction_1min == 'long':
                                retest_condition_1min = (bar_1min['Low'] <= range_high_1min and bar_1min['High'] >= range_high_1min)
                                strong_bullish_1min = bar_1min['Close'] > bar_1min['Open'] and (bar_1min['Close'] - bar_1min['Open']) > bar_1min['ATR']
                                wick_rejection_up_1min = bar_1min['wick_low'] > bar_1min['body'] * wick_body_ratio
                                if retest_condition_1min and (strong_bullish_1min or wick_rejection_up_1min):
                                    retest_confirmed = True
                                    retest_timestamp = bar_1min.name
                                    break
                            else:
                                retest_condition_1min = (bar_1min['High'] >= range_low_1min and bar_1min['Low'] <= range_low_1min)
                                strong_bearish_1min = bar_1min['Close'] < bar_1min['Open'] and (bar_1min['Open'] - bar_1min['Close']) > bar_1min['ATR']
                                wick_rejection_down_1min = bar_1min['wick_high'] > bar_1min['body'] * wick_body_ratio
                                if retest_condition_1min and (strong_bearish_1min or wick_rejection_down_1min):
                                    retest_confirmed = True
                                    retest_timestamp = bar_1min.name
                                    break

                        # Check 5-minute retest if 1-minute not confirmed
                        if not retest_confirmed:
                            window_5min = trading_data_5min[trading_data_5min.index <= timestamp][-2:]  # Last 2 bars
                            for j in range(len(window_5min)):
                                bar_5min = window_5min.iloc[j]
                                if break_direction_1min == 'long':
                                    retest_condition_5min = (bar_5min['Low'] <= range_high_5min and bar_5min['High'] >= range_high_5min)
                                    strong_bullish_5min = bar_5min['Close'] > bar_5min['Open'] and (bar_5min['Close'] - bar_5min['Open']) > bar_5min['ATR']
                                    wick_rejection_up_5min = bar_5min['wick_low'] > bar_5min['body'] * wick_body_ratio
                                    if retest_condition_5min and (strong_bullish_5min or wick_rejection_up_5min):
                                        retest_confirmed = True
                                        retest_timestamp = bar_5min.name
                                        break
                                else:
                                    retest_condition_5min = (bar_5min['High'] >= range_low_5min and bar_5min['Low'] <= range_low_5min)
                                    strong_bearish_5min = bar_5min['Close'] < bar_5min['Open'] and (bar_5min['Open'] - bar_5min['Close']) > bar_5min['ATR']
                                    wick_rejection_down_5min = bar_5min['wick_high'] > bar_5min['body'] * wick_body_ratio
                                    if retest_condition_5min and (strong_bearish_5min or wick_rejection_down_5min):
                                        retest_confirmed = True
                                        retest_timestamp = bar_5min.name
                                        break

                        if retest_confirmed:
                            selected_signal = {
                                'timestamp': retest_timestamp,
                                'Type': f'Break and Retest ({break_direction_1min.capitalize()})',
                                'Level': range_high_1min if break_direction_1min == 'long' else range_low_1min,
                                'Volume': break_candle_1min['Volume'],
                                'VWAP': break_candle_1min['VWAP'] if 'VWAP' in break_candle_1min else np.nan,
                                'RSI': break_candle_1min['RSI'],
                                'Direction': break_direction_1min,
                                'Confluences': {
                                    'Break and Retest with Displacement': True,
                                    'Order Block Continuation': False,
                                    'Previous Day Gap Fill': False,
                                    'Pop and Fade': False,
                                    '3 Bar Pattern': is_3_bar_pattern,
                                    'RSI Oversold': rsi_oversold,
                                    'RSI Overbought': rsi_overbought,
                                    'MACD Bullish': macd_bullish,
                                    'MACD Bearish': macd_bearish,
                                    'Uptrend': is_uptrend,
                                    'Downtrend': is_downtrend
                                },
                                'PDH': pdh,
                                'PDL': pdl,
                                'PMH': pmh,
                                'PML': pml,
                                'BOS_Level': range_high_1min if break_direction_1min == 'long' else range_low_1min
                            }
                            logger.debug("Selected Break and Retest (%s): timestamp=%s", break_direction_1min.capitalize(), retest_timestamp)
                            signals.append(selected_signal)
            except Exception as e:
                logger.error("Error in signal detection loop at bar %d: %s", i, e)
                continue

    bos_signals_df = pd.DataFrame(signals)
    if not bos_signals_df.empty:
        bos_signals_df.set_index('timestamp', inplace=True)
        if bos_signals_df.index.duplicated().any():
            logger.warning("Duplicates found in bos_signals_df: %s", bos_signals_df.index[bos_signals_df.index.duplicated()].tolist())
            bos_signals_df = bos_signals_df[~bos_signals_df.index.duplicated(keep='first')]
    logger.debug("BOS signals added: shape=%s", bos_signals_df.shape if not bos_signals_df.empty else (0, 0))

    gap_signals = detect_gap_fill_reversal(df_5min)
    if not bos_signals_df.empty and not gap_signals.empty:
        signals_df = pd.concat([bos_signals_df, gap_signals])
        signals_df = signals_df[~signals_df.index.duplicated(keep='first')]
    elif not bos_signals_df.empty:
        signals_df = bos_signals_df
    elif not gap_signals.empty:
        signals_df = gap_signals
    else:
        signals_df = pd.DataFrame()

    if not signals_df.empty:
        signals_df = signals_df.sort_index()
    logger.info("Merged signals: total=%d", len(signals_df))

    return signals_df