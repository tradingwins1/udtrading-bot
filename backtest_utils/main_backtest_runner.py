import logging
import os
import yaml
from strategy_wrapper import UGBacktestStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Loaded strategy_wrapper from: %s", os.path.abspath("strategy_wrapper.py"))

    strategy = UGBacktestStrategy(config_path=config_path)

    data_dir = config.get('data_dir', 'data')
    tsla_5min_path = os.path.join(data_dir, config.get('tsla_5min_file', 'TSLA_3M_5min_mock.csv'))
    tsla_1min_path = os.path.join(data_dir, config.get('tsla_1min_file', 'TSLA_3M_1min_mock.csv'))
    qqq_5min_path = os.path.join(data_dir, config.get('qqq_5min_file', 'QQQ_3M_5min_mock.csv'))
    qqq_1min_path = os.path.join(data_dir, config.get('qqq_1min_file', 'QQQ_3M_1min_mock.csv'))

    for path in [tsla_5min_path, tsla_1min_path, qqq_5min_path, qqq_1min_path]:
        if not os.path.exists(path):
            logger.error("Data file not found: %s", path)
            raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading data from: %s", data_dir)
    data = strategy.load_data(
        tsla_5min_path,
        tsla_1min_path,
        qqq_5min_path,
        qqq_1min_path
    )

    # Run backtest on training data
    logger.info("Running backtest on training data")
    strategy.run(start_bar=0, end_bar=len(strategy.train_data)-1, data_subset='train')

    # Run backtest on testing data
    logger.info("Running backtest on testing data")
    strategy.run(start_bar=0, end_bar=len(strategy.test_data)-1, data_subset='test')

    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    strategy.save_trades(os.path.join(output_dir, config.get('trades_output_file', 'trades_output.csv')))

if __name__ == "__main__":
    main()