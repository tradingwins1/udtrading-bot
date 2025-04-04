import sqlite3
from datetime import datetime

DB_NAME = "trade_logs.db"

def check_trade_exit(asset, current_price):
    """
    Scans open trades for a given asset and checks if SL or TP is hit.
    If hit, updates the trade record.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, entry_price, stop_loss, take_profit, direction
        FROM trades
        WHERE asset = ? AND result IS NULL
    """, (asset.upper(),))
    open_trades = cursor.fetchall()

    for trade in open_trades:
        trade_id, entry, sl, tp, direction = trade
        result = None
        exit_price = None

        if direction.upper() == "LONG":
            if current_price <= sl:
                result = "LOSS"
                exit_price = sl
            elif current_price >= tp:
                result = "WIN"
                exit_price = tp

        elif direction.upper() == "SHORT":
            if current_price >= sl:
                result = "LOSS"
                exit_price = sl
            elif current_price <= tp:
                result = "WIN"
                exit_price = tp

        if result:
            print(f"🔁 Trade #{trade_id} | {asset} | {direction} | {result} at {current_price}")
            cursor.execute("""
                UPDATE trades
                SET exit_price = ?, result = ?, timestamp = ?
                WHERE id = ?
            """, (exit_price, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), trade_id))

    conn.commit()
    conn.close()
