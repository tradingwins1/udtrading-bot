
import pandas as pd
import json

# Load config
with open('config.json') as f:
    config = json.load(f)

risk_cfg = config["strategy"]

def apply_risk_management(signals_df):
    managed_trades = []

    for _, row in signals_df.iterrows():
        entry = row['EntryPrice']
        sl = row['StopLoss']
        tp = row['TakeProfit']
        risk = risk_cfg["risk_per_trade"]

        stop_range = abs(entry - sl)
        position_size = max(1, round(risk / stop_range)) if stop_range > 0 else 1

        managed_trades.append({
            'Date': row['Date'],
            'Signal': row['Signal'],
            'EntryPrice': entry,
            'StopLoss': sl,
            'TakeProfit': tp,
            'RiskPerTrade': risk,
            'PositionSize': position_size
        })

    return pd.DataFrame(managed_trades)
