import logging
import pandas as pd
from strategy_wrapper import UGBacktestStrategy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define data paths
    tsla_5min_path = 'data/TSLA_3M_5min_mock.csv'
    tsla_1min_path = 'data/TSLA_3M_1min_mock.csv'
    qqq_5min_path = 'data/QQQ_3M_5min_mock.csv'
    qqq_1min_path = 'data/QQQ_3M_1min_mock.csv'

    # Instantiate the strategy
    strategy = UGBacktestStrategy()

    # Load the data
    strategy.load_data(tsla_5min_path, tsla_1min_path, qqq_5min_path, qqq_1min_path)

    # Initialize models
    strategy.initialize_models()

    # Run the backtest
    strategy.run()

    # Save trades
    strategy.save_trades('trades_output.csv')

if __name__ == "__main__":
    main()