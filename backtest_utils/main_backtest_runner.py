# main_backtest_runner.py
# -----------------------------------------------------
# Run this script to execute a full local backtest using:
# Enhanced UGBacktestStrategy (R:R = 1:2, 75% win rate targeting)
# -----------------------------------------------------

import argparse
import pandas as pd
import os
from backtester import BacktestEngine
import logging

logger = logging.getLogger(__name__)

def run_backtest(mock_file, initial_capital=10000, confidence_threshold=6.0):
    logger.info("Running backtest with mock_file=%s, initial_capital=%s, confidence_threshold=%s", 
                mock_file, initial_capital, confidence_threshold)
    
    if not os.path.exists(mock_file):
        logger.error("Data file not found: %s", mock_file)
        print(f"❌ Data file not found: {mock_file}")
        return

    print(f"📂 Running backtest on: {mock_file}")
    logger.info("Starting backtest on file: %s", mock_file)
    engine = BacktestEngine(initial_capital=initial_capital, use_cached=False, mock_data_file=os.path.basename(mock_file))
    results, trades = engine.run()
    engine.analyze()
    engine.plot()

    # Save results to CSV
    trades.to_csv("trades_output.csv")
    logger.info("Trade log saved to trades_output.csv")
    print("✅ Trade log saved to trades_output.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UG Backtester (Enhanced)")
    parser.add_argument("--mock", type=str, required=True, help="Path to mock CSV file (e.g., TSLA_1M_5min_mock.csv)")
    parser.add_argument("--initial_capital", type=float, default=10000, help="Initial capital to start with")
    parser.add_argument("--confidence_threshold", type=float, default=6.0, help="Minimum confidence score to allow trade entry")

    args = parser.parse_args()

    run_backtest(
        mock_file=args.mock,
        initial_capital=args.initial_capital,
        confidence_threshold=args.confidence_threshold
    )