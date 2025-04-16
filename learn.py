import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "trade_logs.db"

def init_db():
    """Initialize trade_logs.db with updated schema."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_time TEXT,
            exit_time TEXT,
            setup_type TEXT,
            signal_type TEXT,
            result TEXT,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            size INTEGER,
            rsi_entry REAL,
            atr_entry REAL,
            confidence_score REAL,
            holding_period REAL,
            reason TEXT,
            rr_ratio REAL
        )
    ''')
    conn.commit()
    conn.close()

def load_trades():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    return df

def update_trade_log(trade_data, candles_df):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO trades (
            entry_time, exit_time, setup_type, signal_type, result,
            entry_price, exit_price, pnl, size, rsi_entry,
            atr_entry, confidence_score, holding_period, reason, rr_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trade_data['entry_time'].isoformat(),
        trade_data['exit_time'].isoformat(),
        trade_data['setup_type'],
        trade_data['signal_type'],
        trade_data['result'],
        trade_data['entry_price'],
        trade_data['exit_price'],
        trade_data['pnl'],
        trade_data['size'],
        trade_data['rsi_entry'],
        trade_data['atr_entry'],
        trade_data['confidence_score'],
        trade_data['holding_period'],
        trade_data['reason'],
        trade_data['rr_ratio']
    ))
    conn.commit()
    conn.close()

def get_stats(candles_df=None):
    df = load_trades()
    if df.empty:
        return None

    stats = df.groupby('setup_type').agg({
        'result': lambda x: (x == 'win').mean() * 100,
        'rr_ratio': 'mean',
        'confidence_score': 'mean',
        'holding_period': 'mean'
    }).rename(columns={
        'result': 'win_rate_%',
        'rr_ratio': 'avg_rr',
        'confidence_score': 'avg_confidence',
        'holding_period': 'avg_holding_period_hours'
    }).reset_index()
    
    stats['trade_count'] = df.groupby('setup_type').size().reindex(stats['setup_type'], fill_value=0).values
    return stats

if __name__ == "__main__":
    init_db()  # Ensure DB is initialized
    stats = get_stats()
    if stats is not None:
        print(stats)
    else:
        print("No trade data found.")