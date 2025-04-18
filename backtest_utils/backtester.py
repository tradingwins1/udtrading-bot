import pandas as pd
import logging
from strategy_wrapper import UGBacktestStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(tsla_5min_path, qqq_5min_path, tsla_1min_path, qqq_1min_path, initial_capital=10000):
    """
    Run backtest with TSLA and QQQ data on both 5-minute and 1-minute timeframes.
    
    Args:
        tsla_5min_path (str): Path to TSLA 5-minute data CSV
        qqq_5min_path (str): Path to QQQ 5-minute data CSV
        tsla_1min_path (str): Path to TSLA 1-minute data CSV
        qqq_1min_path (str): Path to QQQ 1-minute data CSV
        initial_capital (float): Starting capital for the backtest
    
    Returns:
        tuple: (results_df, trades_df) containing equity curve and trade details
    """
    try:
        # Load data
        logger.info("Loading data files...")
        tsla_5min_data = pd.read_csv(tsla_5min_path, index_col='Datetime', parse_dates=True)
        qqq_5min_data = pd.read_csv(qqq_5min_path, index_col='Datetime', parse_dates=True)
        tsla_1min_data = pd.read_csv(tsla_1min_path, index_col='Datetime', parse_dates=True)
        qqq_1min_data = pd.read_csv(qqq_1min_path, index_col='Datetime', parse_dates=True)

        # Validate data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for df, name in [(tsla_5min_data, 'TSLA 5min'), (qqq_5min_data, 'QQQ 5min'), 
                        (tsla_1min_data, 'TSLA 1min'), (qqq_1min_data, 'QQQ 1min')]:
            if not all(col in df.columns for col in required_columns):
                logger.error(f"{name} missing required columns: {required_columns}")
                raise ValueError(f"{name} missing required columns")

        # Ensure timezone consistency
        for df in [tsla_5min_data, qqq_5min_data, tsla_1min_data, qqq_1min_data]:
            if df.index.tz is None:
                df.index = df.index.tz_localize('US/Eastern')
            else:
                df.index = df.index.tz_convert('US/Eastern')

        logger.info("Data loaded successfully. TSLA 5min shape: %s, QQQ 5min shape: %s, TSLA 1min shape: %s, QQQ 1min shape: %s",
                    tsla_5min_data.shape, qqq_5min_data.shape, tsla_1min_data.shape, qqq_1min_data.shape)

        # Initialize and run strategy
        strategy = UGBacktestStrategy(
            tsla_5min_data=tsla_5min_data,
            qqq_5min_data=qqq_5min_data,
            tsla_1min_data=tsla_1min_data,
            qqq_1min_data=qqq_1min_data,
            initial_capital=initial_capital
        )

        logger.info("Starting backtest...")
        results, trades = strategy.run()
        logger.info("Backtest completed. Total trades: %d", len(trades))

        return results, trades

    except Exception as e:
        logger.error("Error in backtest: %s", e)
        raise

if __name__ == "__main__":
    # Default paths to mock data
    tsla_5min_path = "data/TSLA_3M_5min_mock.csv"
    qqq_5min_path = "data/QQQ_3M_5min_mock.csv"
    tsla_1min_path = "data/TSLA_3M_1min_mock.csv"
    qqq_1min_path = "data/QQQ_3M_1min_mock.csv"

    results, trades = run_backtest(tsla_5min_path, qqq_5min_path, tsla_1min_path, qqq_1min_path)
    trades.to_csv("trades_output.csv")
    print("Backtest completed. Trades saved to trades_output.csv")