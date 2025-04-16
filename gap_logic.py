import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_gap_fill_reversal(data, gap_threshold=0.01, lookahead_bars=18):
    """
    Detects gap fill reversal setups.
    Args:
        data (DataFrame): Price data with datetime index.
        gap_threshold (float): Minimum gap % to qualify (default 1%).
        lookahead_bars (int): Bars to look ahead for reversal (1 bar = 5 min).
    Returns:
        DataFrame: Signals DataFrame with entry points in BOS format.
    """
    logger.debug("Starting gap fill reversal detection for data with shape: %s", data.shape)
    signals = []
    df = data.copy()
    
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        logger.error("DataFrame missing required OHLCV columns")
        raise ValueError("DataFrame missing required OHLCV columns")

    try:
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['gap_type'] = np.where(df['gap_pct'] > gap_threshold, 'up',
                                 np.where(df['gap_pct'] < -gap_threshold, 'down', None))
        logger.debug("Gap ups identified: %s, Gap downs: %s", 
                     (df['gap_type'] == 'up').sum(), (df['gap_type'] == 'down').sum())
    except Exception as e:
        logger.error("Error calculating gap metrics: %s", e)
        raise

    for i in range(1, len(df) - lookahead_bars):
        try:
            row = df.iloc[i]
            if row['gap_type'] is None:
                continue

            window = df.iloc[i:i+lookahead_bars]
            if row['gap_type'] == 'up':
                reversal = window['Low'].min() <= row['prev_close']
                direction = 'short'
            else:
                reversal = window['High'].max() >= row['prev_close']
                direction = 'long'

            if reversal:
                signal = {
                    'timestamp': row.name,
                    'Type': 'Gap Fill Reversal',
                    'Direction': direction,
                    'Confluences': {'Previous Day Gap Fill': True},
                    'PDH': np.nan,
                    'PDL': np.nan,
                    'PMH': np.nan,
                    'PML': np.nan,
                    'BOS_Level': row['prev_close'],
                    'Volume': row['Volume'],
                    'VWAP': row['VWAP'] if 'VWAP' in df.columns else np.nan,
                    'RSI': row['RSI'] if 'RSI' in df.columns else np.nan
                }
                signals.append(signal)
                logger.debug("Detected Gap Fill Reversal at %s: direction=%s, level=%s", 
                             row.name, direction, row['prev_close'])
        except Exception as e:
            logger.error("Error processing bar %s: %s", i, e)
            continue

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df.set_index('timestamp', inplace=True)
    else:
        logger.warning("No gap fill reversal signals generated.")
    
    logger.debug("Gap fill reversal signals: shape=%s", signals_df.shape if not signals_df.empty else (0, 0))
    return signals_df