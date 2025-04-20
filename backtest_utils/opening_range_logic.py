import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from ta.trend import SMAIndicator

logger = logging.getLogger(__name__)

def detect_opening_range_signals(tsla_5min_data, tsla_1min_data, qqq_5min_data, key_levels):
    """
    Detect 1-min and 5-min opening range break (ORB) signals around PMH/PML, aligned with QQQ trend.
    Args:
        tsla_5min_data (pd.DataFrame): 5-min TSLA data with OHLC, RSI, ATR, SMA200.
        tsla_1min_data (pd.DataFrame): 1-min TSLA data with OHLC, ATR.
        qqq_5min_data (pd.DataFrame): 5-min QQQ data for trend alignment.
        key_levels (dict): Dictionary of daily PMH, PML, PDL, PDH levels.
    Returns:
        pd.DataFrame: Signals with timestamp, Type, Direction, Confluences.
    """
    signals = []
    
    # Align data indices
    tsla_1min_data = tsla_1min_data.copy()
    tsla_5min_data = tsla_5min_data.copy()
    qqq_5min_data = qqq_5min_data.copy()
    
    # Calculate QQQ trend (1H timeframe)
    h1_data = qqq_5min_data.resample('1h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    h1_data['SMA200'] = SMAIndicator(h1_data['Close'], window=200).sma_indicator()
    qqq_trend = 'up' if h1_data['Close'].iloc[-1] > h1_data['SMA200'].iloc[-1] else 'down'
    
    # Define NY session open (9:30 AM EST)
    ny_open = time(9, 30)
    
    # Group data by day for ORB calculation
    daily_groups_1min = tsla_1min_data.groupby(tsla_1min_data.index.date)
    daily_groups_5min = tsla_5min_data.groupby(tsla_5min_data.index.date)
    
    for day, day_data_1min in daily_groups_1min:
        day_data_5min = daily_groups_5min.get_group(day) if day in daily_groups_5min.groups else pd.DataFrame()
        if day_data_5min.empty:
            continue
        
        # Get PMH, PML, PDL, PDH for the day
        levels = key_levels.get(day, {})
        pmh, pml, pdl, pdh = levels.get('PMH'), levels.get('PML'), levels.get('PDL'), levels.get('PDH')
        if not all([pmh, pml, pdl, pdh]):
            continue
        
        # Determine TSLA trend bias using 5-min SMA200
        trend = 'uptrend' if day_data_5min['SMA200'].iloc[-1] < day_data_5min['Close'].iloc[-1] else 'downtrend'
        
        # --- 1-min ORB ---
        # Find first 1-min candle at or after 9:30 AM
        first_1min_candle = day_data_1min[day_data_1min.index.time >= ny_open].head(1)
        if first_1min_candle.empty:
            continue
        
        orb_1min_high = first_1min_candle['High'].iloc[0]
        orb_1min_low = first_1min_candle['Low'].iloc[0]
        orb_1min_timestamp = first_1min_candle.index[0]
        
        # Look for break of 1-min ORB within the next 10 minutes
        look_forward_1min = day_data_1min[
            (day_data_1min.index > orb_1min_timestamp) &
            (day_data_1min.index <= orb_1min_timestamp + pd.Timedelta(minutes=10))
        ]
        
        for idx, row in look_forward_1min.iterrows():
            confluences = {
                'ORB 1-min Break': True,
                'Liquidity Sweep': False,
                'Uptrend': trend == 'uptrend',
                'Downtrend': trend == 'downtrend',
                'QQQ Aligned': False
            }
            
            # Check for liquidity sweep (price briefly breaks PMH/PML then reverses)
            if row['High'] > pmh:
                prior_candles = look_forward_1min[look_forward_1min.index < idx].tail(3)
                if prior_candles['Low'].min() < pmh:
                    confluences['Liquidity Sweep'] = True
            
            if row['Low'] < pml:
                prior_candles = look_forward_1min[look_forward_1min.index < idx].tail(3)
                if prior_candles['High'].max() > pml:
                    confluences['Liquidity Sweep'] = True
            
            # Validate liquidity sweep with QQQ trend
            if confluences['Liquidity Sweep']:
                if (row['High'] > pmh and qqq_trend == 'down') or (row['Low'] < pml and qqq_trend == 'up'):
                    confluences['Liquidity Sweep'] = False  # Likely a fakeout
            
            # Long: Break above 1-min ORB high, near PMH, aligned with QQQ
            if (row['High'] > orb_1min_high and 
                abs(row['Close'] - pmh) <= row['ATR'] * 0.5 and 
                qqq_trend == 'up'):
                confluences['QQQ Aligned'] = True
                signals.append({
                    'timestamp': idx,
                    'Type': 'Opening Range Break (Long)',
                    'Direction': 'long',
                    'Confluences': confluences,
                    'PDH': pdh,
                    'PDL': pdl,
                    'PMH': pmh,
                    'PML': pml,
                    'BOS_Level': orb_1min_high
                })
            
            # Short: Break below 1-min ORB low, near PML, aligned with QQQ
            elif (row['Low'] < orb_1min_low and 
                  abs(row['Close'] - pml) <= row['ATR'] * 0.5 and 
                  qqq_trend == 'down'):
                confluences['QQQ Aligned'] = True
                signals.append({
                    'timestamp': idx,
                    'Type': 'Opening Range Break (Short)',
                    'Direction': 'short',
                    'Confluences': confluences,
                    'PDH': pdh,
                    'PDL': pdl,
                    'PMH': pmh,
                    'PML': pml,
                    'BOS_Level': orb_1min_low
                })
        
        # --- 5-min ORB ---
        # Find first 5-min candle at or after 9:30 AM
        first_5min_candle = day_data_5min[day_data_5min.index.time >= ny_open].head(1)
        if first_5min_candle.empty:
            continue
        
        orb_5min_high = first_5min_candle['High'].iloc[0]
        orb_5min_low = first_5min_candle['Low'].iloc[0]
        orb_5min_timestamp = first_5min_candle.index[0]
        
        # Look for break of 5-min ORB within the next 30 minutes
        look_forward_5min = day_data_5min[
            (day_data_5min.index > orb_5min_timestamp) &
            (day_data_5min.index <= orb_5min_timestamp + pd.Timedelta(minutes=30))
        ]
        
        for idx, row in look_forward_5min.iterrows():
            confluences = {
                'ORB 5-min Break': True,
                'Liquidity Sweep': False,
                'Uptrend': trend == 'uptrend',
                'Downtrend': trend == 'downtrend',
                'QQQ Aligned': False
            }
            
            # Check for liquidity sweep
            if row['High'] > pmh:
                prior_candles = look_forward_5min[look_forward_5min.index < idx].tail(3)
                if prior_candles['Low'].min() < pmh:
                    confluences['Liquidity Sweep'] = True
            
            if row['Low'] < pml:
                prior_candles = look_forward_5min[look_forward_5min.index < idx].tail(3)
                if prior_candles['High'].max() > pml:
                    confluences['Liquidity Sweep'] = True
            
            # Validate liquidity sweep with QQQ trend
            if confluences['Liquidity Sweep']:
                if (row['High'] > pmh and qqq_trend == 'down') or (row['Low'] < pml and qqq_trend == 'up'):
                    confluences['Liquidity Sweep'] = False
            
            # Long: Break above 5-min ORB high, near PMH, aligned with QQQ
            if (row['High'] > orb_5min_high and 
                abs(row['Close'] - pmh) <= row['ATR'] * 0.5 and 
                qqq_trend == 'up'):
                confluences['QQQ Aligned'] = True
                signals.append({
                    'timestamp': idx,
                    'Type': 'Opening Range Break (Long)',
                    'Direction': 'long',
                    'Confluences': confluences,
                    'PDH': pdh,
                    'PDL': pdl,
                    'PMH': pmh,
                    'PML': pml,
                    'BOS_Level': orb_5min_high
                })
            
            # Short: Break below 5-min ORB low, near PML, aligned with QQQ
            elif (row['Low'] < orb_5min_low and 
                  abs(row['Close'] - pml) <= row['ATR'] * 0.5 and 
                  qqq_trend == 'down'):
                confluences['QQQ Aligned'] = True
                signals.append({
                    'timestamp': idx,
                    'Type': 'Opening Range Break (Short)',
                    'Direction': 'short',
                    'Confluences': confluences,
                    'PDH': pdh,
                    'PDL': pdl,
                    'PMH': pmh,
                    'PML': pml,
                    'BOS_Level': orb_5min_low
                })
    
    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df.set_index('timestamp', inplace=True)
        logger.info("ORB signals generated: %d, timestamps: %s", len(signals_df), signals_df.index.tolist()[:5])
    else:
        logger.warning("No ORB signals generated.")
    
    return signals_df