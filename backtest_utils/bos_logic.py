import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from gap_logic import detect_gap_fill_reversal
import logging
import pytz

logger = logging.getLogger(__name__)

def detect_ug_signals(df, lookback=3, volume_factor=1.2, wick_body_ratio=1.0, fib_84_level=0.84):
    """
    Detect UG Trading Bot signals with prioritized confluences on 5-minute TSLA data.
    Generates only one signal per bar to avoid duplicates.
    """
    signals = []
    logger.debug("Starting signal detection for data with shape: %s", df.shape)
    
    df = df.copy()
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        logger.error("DataFrame missing required OHLCV columns")
        raise ValueError("DataFrame missing required OHLCV columns")
    
    try:
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['VWAP'] = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=14).volume_weighted_average_price()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        logger.debug("Indicators calculated successfully")
    except Exception as e:
        logger.error("Error calculating indicators: %s", e)
        raise

    # Localize timestamps directly as US/Eastern (EST)
    if df.index.tz is None:
        df.index = df.index.tz_localize('US/Eastern')
    else:
        df.index = df.index.tz_convert('US/Eastern')
    logger.debug("Timestamps localized to US/Eastern: %s", df.index[:5].tolist())

    # Volume spike
    df['avg_volume'] = df['Volume'].shift(1).rolling(14).mean()
    df['is_volume_spike'] = df['Volume'] > df['avg_volume'] * volume_factor
    logger.debug("Volume spikes identified: %s", df['is_volume_spike'].sum())

    # Wick/body ratio
    df['body'] = np.abs(df['Close'] - df['Open'])
    df['wick_high'] = df['High'] - np.maximum(df['Open'], df['Close'])
    df['wick_low'] = np.minimum(df['Open'], df['Close']) - df['Low']
    df['is_sweep_high'] = (df['wick_high'] > df['body'] * wick_body_ratio)
    df['is_sweep_low'] = (df['wick_low'] > df['body'] * wick_body_ratio)
    logger.debug("Sweep lows identified: %s", df['is_sweep_low'].sum())

    # NY Market Open (9:30-11:30 AM EST)
    df['time'] = df.index.time
    logger.debug("Sample timestamps: %s", df.index[:5].tolist())
    logger.debug("Sample times: %s", df['time'][:5].tolist())
    df['is_ny_open'] = (df['time'] >= pd.Timestamp('09:30').time()) & (df['time'] <= pd.Timestamp('11:30').time())
    logger.debug("NY Open periods identified: %s", df['is_ny_open'].sum())

    # Log total bars before filtering
    total_bars = len(df)
    logger.debug("Total bars before filtering: %s", total_bars)

    # Filter for NY Open
    ny_open_bars = df[df['is_ny_open']]
    logger.debug("Bars after NY Open filter: %s", len(ny_open_bars))

    # Generate Gap Fill Signals from gap_logic.py
    gap_signals = detect_gap_fill_reversal(df, gap_threshold=0.01)
    gap_signals_df = pd.DataFrame()
    if not gap_signals.empty:
        gap_signals_df = gap_signals
        logger.debug("Gap signals added: shape=%s", gap_signals_df.shape)
    else:
        logger.warning("No gap fill signals generated.")

    # BOS and Liquidity Sweep Signals (one signal per bar)
    for i in range(lookback, len(df) - lookback):
        try:
            if not df['is_ny_open'].iloc[i]:
                continue

            # 3 Bar Pattern (simplified: 1 bearish candle followed by 1 bullish candle)
            if i >= 1:
                bar_1_bearish = df['Close'].iloc[i-1] < df['Open'].iloc[i-1]
                bar_2_bullish = df['Close'].iloc[i] > df['Open'].iloc[i]
                is_3_bar_pattern = bar_1_bearish and bar_2_bullish
            else:
                is_3_bar_pattern = False
            logger.debug("3 Bar Pattern at bar %s: %s (bar_1_bearish=%s, bar_2_bullish=%s)", 
                         i, is_3_bar_pattern, bar_1_bearish, bar_2_bullish)

            # Previous Day Gap Fill Confluence
            has_gap_fill = df.index[i] in gap_signals_df.index if not gap_signals_df.empty else False
            logger.debug("Previous Day Gap Fill at bar %s: has_gap_fill=%s", i, has_gap_fill)

            # Initialize signal for this bar
            selected_signal = None

            # Priority 1: Liquidity Sweep Low (Pop and Fade Out)
            if df['is_sweep_low'].iloc[i]:
                for k in range(i + 1, min(len(df), i + 5)):
                    if df['Close'].iloc[k] > df['Low'].iloc[i]:
                        selected_signal = {
                            'timestamp': df.index[k],
                            'Type': 'Liquidity Sweep Low at KPL',
                            'Level': df['Low'].iloc[i],
                            'Volume': df['Volume'].iloc[k],
                            'VWAP': df['VWAP'].iloc[k],
                            'RSI': df['RSI'].iloc[k],
                            'Direction': 'long',
                            'Confluences': {
                                'Break and Retest with Displacement': False,
                                'Order Block Continuation': False,
                                'Previous Day Gap Fill': has_gap_fill,
                                'Pop and Fade Out': True,
                                '3 Bar Pattern': is_3_bar_pattern
                            },
                            'PDH': np.nan,
                            'PDL': np.nan,
                            'PMH': np.nan,
                            'PML': np.nan,
                            'BOS_Level': df['Low'].iloc[i]
                        }
                        logger.debug("Selected Liquidity Sweep Low at bar %s, timestamp=%s", k, df.index[k])
                        break

            # Priority 2: BOS Low with Break and Retest (only if no liquidity sweep)
            if not selected_signal and i > 0:
                swing_low = df['Low'].iloc[i]
                for k in range(i + 1, min(len(df), i + lookback + 10)):
                    break_candle_body = abs(df['Close'].iloc[k] - df['Open'].iloc[k])
                    displacement_condition = (df['Close'].iloc[k] < swing_low and 
                                             break_candle_body > 2 * df['ATR'].iloc[k])
                    if displacement_condition:
                        ob_high = df['High'].iloc[i-1] if df['Close'].iloc[i-1] > df['Open'].iloc[i-1] else swing_low
                        ob_low = df['Low'].iloc[i-1] if df['Close'].iloc[i-1] > df['Open'].iloc[i-1] else swing_low
                        highs_slice = df['High'].iloc[i:k]
                        if len(highs_slice) > 0:
                            retrace_level = swing_low + (highs_slice.max() - swing_low) * fib_84_level
                            for j in range(k, min(len(df), k + 5)):
                                retest_condition = (df['High'].iloc[j] >= retrace_level and 
                                                    df['High'].iloc[j] <= ob_high and 
                                                    df['Low'].iloc[j] >= ob_low)
                                if retest_condition:
                                    selected_signal = {
                                        'timestamp': df.index[j],
                                        'Type': 'BOS Low Retest',
                                        'Level': swing_low,
                                        'Volume': df['Volume'].iloc[j],
                                        'VWAP': df['VWAP'].iloc[j],
                                        'RSI': df['RSI'].iloc[j],
                                        'Direction': 'short',
                                        'Confluences': {
                                            'Break and Retest with Displacement': True,
                                            'Order Block Continuation': True,
                                            'Previous Day Gap Fill': has_gap_fill,
                                            'Pop and Fade Out': False,
                                            '3 Bar Pattern': is_3_bar_pattern
                                        },
                                        'PDH': np.nan,
                                        'PDL': np.nan,
                                        'PMH': np.nan,
                                        'PML': np.nan,
                                        'BOS_Level': swing_low
                                    }
                                    logger.debug("Selected BOS Low Retest at bar %s, timestamp=%s", j, df.index[j])
                                    break
                            else:
                                selected_signal = {
                                    'timestamp': df.index[k],
                                    'Type': 'BOS Low',
                                    'Level': swing_low,
                                    'Volume': df['Volume'].iloc[k],
                                    'VWAP': df['VWAP'].iloc[k],
                                    'RSI': df['RSI'].iloc[k],
                                    'Direction': 'short',
                                    'Confluences': {
                                        'Break and Retest with Displacement': True,
                                        'Order Block Continuation': False,
                                        'Previous Day Gap Fill': has_gap_fill,
                                        'Pop and Fade Out': False,
                                        '3 Bar Pattern': is_3_bar_pattern
                                    },
                                    'PDH': np.nan,
                                    'PDL': np.nan,
                                    'PMH': np.nan,
                                    'PML': np.nan,
                                    'BOS_Level': swing_low
                                }
                                logger.debug("Selected BOS Low at bar %s, timestamp=%s", k, df.index[k])
                            break

            if selected_signal:
                signals.append(selected_signal)

        except Exception as e:
            logger.error("Error processing bar %s: %s", i, e)
            continue

    bos_signals_df = pd.DataFrame(signals)
    if not bos_signals_df.empty:
        bos_signals_df.set_index('timestamp', inplace=True)
        # Ensure no duplicates in bos_signals_df
        if bos_signals_df.index.duplicated().any():
            logger.warning("Duplicates found in bos_signals_df: %s", bos_signals_df.index[bos_signals_df.index.duplicated()].tolist())
            bos_signals_df = bos_signals_df[~bos_signals_df.index.duplicated(keep='first')]
    logger.debug("BOS signals added: shape=%s", bos_signals_df.shape if not bos_signals_df.empty else (0, 0))

    # Merge BOS and Gap Signals
    if not bos_signals_df.empty and not gap_signals_df.empty:
        signals_df = pd.concat([bos_signals_df, gap_signals_df])
        # Final deduplication
        signals_df = signals_df[~signals_df.index.duplicated(keep='first')]
    elif not bos_signals_df.empty:
        signals_df = bos_signals_df
    elif not gap_signals_df.empty:
        signals_df = gap_signals_df
    else:
        signals_df = pd.DataFrame()

    if not signals_df.empty:
        signals_df = signals_df.sort_index()
    logger.info("Merged signals: total=%s", len(signals_df))

    return signals_df