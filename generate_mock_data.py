# generate_mock_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_columns(df):
    """Normalize column names to ['Open', 'High', 'Low', 'Close', 'Volume']."""
    # Log the original column names for debugging
    logger.info(f"Original columns in DataFrame: {df.columns.tolist()}")

    # Define possible column name variations (case-insensitive)
    column_map = {
        'open': ['open', 'OPEN', 'O', 'price_open', 'Open'],
        'high': ['high', 'HIGH', 'H', 'price_high', 'High'],
        'low': ['low', 'LOW', 'L', 'price_low', 'Low'],
        'close': ['close', 'CLOSE', 'C', 'price_close', 'Close'],
        'volume': ['volume', 'VOLUME', 'V', 'vol', 'Volume']
    }

    # Create a mapping from input columns to standard names
    rename_dict = {}
    for standard_col, aliases in column_map.items():
        for alias in aliases:
            for col in df.columns:
                if col.lower() == alias.lower():
                    rename_dict[col] = standard_col.capitalize()
                    break

    # Check if all required columns are found
    missing_cols = [col for col in column_map.keys() if col.capitalize() not in rename_dict.values()]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Rename columns
    df = df.rename(columns=rename_dict)
    logger.info(f"Normalized columns: {df.columns.tolist()}")
    return df

def interpolate_5min_to_1min(df_5min, symbol):
    """Interpolate 5-minute data to 1-minute data."""
    try:
        # Normalize columns
        df_5min = normalize_columns(df_5min)

        # Ensure index is datetime and timezone-aware
        df_5min.index = pd.to_datetime(df_5min.index)
        if df_5min.index.tz is None:
            df_5min.index = df_5min.index.tz_localize('US/Eastern')
        else:
            df_5min.index = df_5min.index.tz_convert('US/Eastern')
        df_5min = df_5min.sort_index()

        # Create 1-minute index with the same timezone
        start_date = df_5min.index.min()
        end_date = df_5min.index.max()
        minute_index = pd.date_range(start=start_date, end=end_date, freq='1min', tz='US/Eastern')
        
        # Interpolate OHLCV
        df_1min = pd.DataFrame(index=minute_index)
        df_1min = df_1min.join(df_5min[['Open', 'High', 'Low', 'Close', 'Volume']])
        
        # Forward-fill and interpolate
        df_1min[['Open', 'Close']] = df_1min[['Open', 'Close']].interpolate(method='linear')
        df_1min['High'] = df_1min[['Open', 'Close']].max(axis=1)
        df_1min['Low'] = df_1min[['Open', 'Close']].min(axis=1)
        df_1min['Volume'] = df_1min['Volume'].interpolate(method='linear') / 5  # Distribute volume across 5 minutes
        
        # Fill any remaining NaNs
        df_1min = df_1min.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure column names match IBKR format
        df_1min.index.name = 'timestamp'
        
        # Save to CSV
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{symbol}_3M_1min_mock.csv"
        df_1min.to_csv(output_file)
        logger.info(f"Generated 1-minute mock data for {symbol} saved to {output_file}")
        return df_1min
    except Exception as e:
        logger.error(f"Error interpolating 1-minute data for {symbol}: {e}")
        return None

def generate_qqq_mock_data(tsla_5min_data):
    """Generate mock QQQ data based on TSLA with adjusted volatility."""
    try:
        # Normalize columns before processing
        qqq_5min = tsla_5min_data.copy()
        qqq_5min = normalize_columns(qqq_5min)
        
        # Adjust prices (QQQ is less volatile, assume ~50% of TSLA's price movements)
        qqq_5min['Open'] *= 0.5
        qqq_5min['High'] *= 0.5
        qqq_5min['Low'] *= 0.5
        qqq_5min['Close'] *= 0.5
        qqq_5min['Volume'] *= 0.7  # Lower volume for QQQ
        
        # Add some noise to simulate different price dynamics
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, len(qqq_5min))
        qqq_5min['Close'] += qqq_5min['Close'] * noise
        qqq_5min['Open'] = qqq_5min['Close'].shift(1).fillna(qqq_5min['Open'])
        qqq_5min['High'] = qqq_5min[['Open', 'Close']].max(axis=1)
        qqq_5min['Low'] = qqq_5min[['Open', 'Close']].min(axis=1)
        
        # Save to CSV
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/QQQ_3M_5min_mock.csv"
        qqq_5min.to_csv(output_file)
        logger.info(f"Generated 5-minute mock data for QQQ saved to {output_file}")
        
        # Interpolate to 1-minute
        qqq_1min = interpolate_5min_to_1min(qqq_5min, "QQQ")
        if qqq_1min is None:
            logger.error("Failed to interpolate QQQ 1-minute data")
            return None, None
        return qqq_5min, qqq_1min
    except Exception as e:
        logger.error(f"Error generating QQQ mock data: {e}")
        return None, None

def main():
    # Load TSLA 5-minute mock data
    try:
        tsla_5min_data = pd.read_csv("data/TSLA_3M_5min_mock.csv", index_col=0, parse_dates=True)
        logger.info("Loaded TSLA 5-minute mock data")
        # Normalize columns immediately after loading
        tsla_5min_data = normalize_columns(tsla_5min_data)
    except FileNotFoundError:
        logger.error("TSLA_3M_5min_mock.csv not found in data/ directory")
        return
    except Exception as e:
        logger.error(f"Error loading TSLA_3M_5min_mock.csv: {e}")
        return

    # Interpolate TSLA 1-minute data
    tsla_1min_data = interpolate_5min_to_1min(tsla_5min_data, "TSLA")
    if tsla_1min_data is None:
        logger.error("Failed to generate TSLA 1-minute mock data")
        return
    logger.info("Successfully generated TSLA 1-minute mock data")

    # Generate QQQ mock data
    qqq_5min_data, qqq_1min_data = generate_qqq_mock_data(tsla_5min_data)
    if qqq_5min_data is None or qqq_1min_data is None:
        logger.error("Failed to generate QQQ mock data")
        return

    logger.info("All mock data generated successfully")

if __name__ == "__main__":
    main()