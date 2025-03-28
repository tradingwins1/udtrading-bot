import pandas as pd
import os
from datetime import datetime

LOG_FILE = "trade_log.csv"

def log_trade(trade):
    """
    Appends a trade row to the trade_log.csv file.
    """
    trade_data = {
        "Date": trade.get("Date", datetime.utcnow().isoformat()),
        "Signal": trade.get("Signal"),
        "EntryPrice": trade.get("EntryPrice"),
        "StopLoss": trade.get("StopLoss"),
        "TakeProfit": trade.get("TakeProfit"),
        "ExitPrice": trade.get("ExitPrice", None),
        "ExitType": trade.get("ExitType", None),
        "PnL": trade.get("PnL", None)
    }

    df = pd.DataFrame([trade_data])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    print("✅ Trade logged to trade_log.csv")
