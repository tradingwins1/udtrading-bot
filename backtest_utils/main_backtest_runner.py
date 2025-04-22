# main_backtest_runner.py
import logging
import os
from strategy_wrapper import UGBacktestStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loaded strategy_wrapper from: %s", os.path.abspath("strategy_wrapper.py"))

    strategy = UGBacktestStrategy()

    logger.info("load_data in dir: %s", os.path.exists("data"))
    strategy.load_data(
        "data/TSLA_3M_5min_mock.csv",
        "data/TSLA_3M_1min_mock.csv",
        "data/QQQ_3M_5min_mock.csv",
        "data/QQQ_3M_1min_mock.csv"
    )

    logger.info("Methods: %s", [method for method in dir(strategy) if not method.startswith('_')])
    strategy.run()
    strategy.save_trades("trades_output.csv")

if __name__ == "__main__":
    main()