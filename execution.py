# execution.py
import pandas as pd
from trailing_stop import simulate_trailing_stop
from trade_logger import log_trade


def execute_trade_with_trailing_sl(trades_df):
    results = []

    for _, trade in trades_df.iterrows():
        price_stream = [trade['TakeProfit']]  # Simulated single TP hit for simplicity
        result = simulate_trailing_stop(trade, price_stream)
        results.append(result)
        log_trade(result)

    return pd.DataFrame(results)
