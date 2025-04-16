# db.py
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_NAME = "/home/ibuser/bot/trade_logs.db"

def init_db():
    logger.info("Initializing database at %s", DB_NAME)
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                target REAL,
                qty REAL,
                status TEXT,
                type TEXT,
                confidence REAL,
                session TEXT
            )
        ''')
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error("Failed to initialize database: %s", e)
    finally:
        conn.close()

def log_trade(trade):
    logger.info("Logging trade for %s: %s", trade['asset'], trade['direction'])
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (
                timestamp, symbol, side, entry, sl, tp, target, qty, status, type, confidence, session
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            trade['asset'],
            trade['direction'],
            trade['entry_price'],
            trade['stop_loss'],
            trade['take_profit'],
            trade.get('exit_price', 0.0),
            trade['rr_ratio'],
            trade['result'],
            trade['setup_type'],
            trade.get('confidence_score', 0.0),
            trade.get('notes', '')
        ))
        conn.commit()
        logger.info("Trade logged successfully")
    except sqlite3.Error as e:
        logger.error("Failed to log trade: %s", e)
    finally:
        conn.close()