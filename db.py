import sqlite3
from datetime import datetime

DB_NAME = "trade_logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            asset TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            rr_ratio REAL,
            result TEXT,
            setup_type TEXT,
            confidence_score REAL,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(trade):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO trades (
            timestamp, asset, direction, entry_price, exit_price, stop_loss,
            take_profit, rr_ratio, result, setup_type, confidence_score, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trade['asset'],
        trade['direction'],
        trade['entry_price'],
        trade['exit_price'],
        trade['stop_loss'],
        trade['take_profit'],
        trade['rr_ratio'],
        trade['result'],
        trade['setup_type'],
        trade.get('confidence_score', None),
        trade.get('notes', '')
    ))
    conn.commit()
    conn.close()
