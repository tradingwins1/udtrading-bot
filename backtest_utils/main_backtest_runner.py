import pandas as pd
from strategy_wrapper import UGBacktestStrategy
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv_with_datetime(file_path):
    """Load CSV with flexible datetime column handling."""
    try:
        # Read the first row to inspect headers
        df = pd.read_csv(file_path, nrows=0)
        columns = df.columns.tolist()
        logger.info(f"Columns in {file_path}: {columns}")

        # Try common datetime column names
        datetime_cols = ['Datetime', 'DateTime', 'date', 'timestamp', 'time']
        datetime_col = None
        for col in datetime_cols:
            if col in columns:
                datetime_col = col
                break

        if datetime_col:
            # Explicitly parse the datetime column
            df = pd.read_csv(file_path, index_col=datetime_col, parse_dates=[datetime_col])
        else:
            # Fall back to first column as index
            logger.warning(f"No recognized datetime column in {file_path}. Using first column as index.")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Ensure index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Index in {file_path} is not a DatetimeIndex. Attempting to convert.")
            # Log a sample of index values, including position 98910 if possible
            sample_indices = df.index[:5].tolist()
            if len(df.index) > 98910:
                sample_indices.append(df.index[98910])
            logger.info(f"Sample index values before conversion in {file_path}: {sample_indices}")
            # Convert with utc=True to handle timezone-aware datetimes
            df.index = pd.to_datetime(df.index, utc=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error(f"Failed to convert index in {file_path} to DatetimeIndex.")
                raise ValueError(f"Index in {file_path} could not be converted to DatetimeIndex")

        # Log first few index values for debugging
        logger.info(f"First few index values in {file_path}: {df.index[:5].tolist()}")

        # Ensure index is timezone-aware (US/Eastern)
        if df.index.tz is None:
            df.index = df.index.tz_localize('US/Eastern')
        else:
            df.index = df.index.tz_convert('US/Eastern')

        # Normalize column names to expected format
        column_map = {
            'open': ['open', 'OPEN', 'O'],
            'high': ['high', 'HIGH', 'H'],
            'low': ['low', 'LOW', 'L'],
            'close': ['close', 'CLOSE', 'C'],
            'volume': ['volume', 'VOLUME', 'V']
        }
        rename_dict = {}
        for standard, aliases in column_map.items():
            for alias in aliases:
                if alias in df.columns:
                    rename_dict[alias] = standard.capitalize()
                    break

        df = df.rename(columns=rename_dict)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in {file_path}: {required_columns}")
            raise ValueError(f"Missing required columns in {file_path}")

        logger.info(f"Loaded {file_path} with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsla_5min", help="Path to TSLA 5-minute data file")
    parser.add_argument("--tsla_1min", help="Path to TSLA 1-minute data file")
    parser.add_argument("--qqq_5min", help="Path to QQQ 5-minute data file")
    parser.add_argument("--qqq_1min", help="Path to QQQ 1-minute data file")
    args = parser.parse_args()

    # Load TSLA data
    tsla_5min_data = load_csv_with_datetime(args.tsla_5min if args.tsla_5min else "data/TSLA_3M_5min_mock.csv")
    tsla_1min_data = load_csv_with_datetime(args.tsla_1min if args.tsla_1min else "data/TSLA_3M_1min_mock.csv")

    # Load QQQ data
    qqq_5min_data = load_csv_with_datetime(args.qqq_5min if args.qqq_5min else "data/QQQ_3M_5min_mock.csv")
    qqq_1min_data = load_csv_with_datetime(args.qqq_1min if args.qqq_1min else "data/QQQ_3M_1min_mock.csv")

    # Run backtest
    strategy = UGBacktestStrategy(tsla_5min_data, qqq_5min_data, tsla_1min_data, qqq_1min_data)
    results, trades = strategy.run()

    # Save results
    trades.to_csv("trades_output.csv")
    print("Backtest completed. Trades saved to trades_output.csv")

if __name__ == "__main__":
    main()