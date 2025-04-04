import sqlite3
import pandas as pd

DB_NAME = "trade_logs.db"

def load_trades():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    return df

def get_stats():
    df = load_trades()
    if df.empty:
        return None

    stats = df.groupby('setup_type').agg({
        'result': lambda x: (x == 'WIN').mean() * 100,
        'rr_ratio': 'mean'
    }).rename(columns={'result': 'win_rate_%', 'rr_ratio': 'avg_rr'}).reset_index()

    return stats
if __name__ == "__main__":
    stats = get_stats()
    if stats is not None:
        print(stats)
    else:
        print("No trade data found.")



