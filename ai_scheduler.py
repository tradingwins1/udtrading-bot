import schedule
import asyncio
import time
from datetime import datetime
from ibkr_client import IBKRTrader
from red_news_filter import is_red_folder_event_today
from live_tracker import add_live_trade, track_live_trades
from price_feed import get_latest_price
from execution import check_entry
import pytz
import os
from dotenv import load_dotenv
from db import init_db, log_trade
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

init_db()
load_dotenv()
trader = IBKRTrader()

trade_counter = {
    "scalp": 0,
    "swing": 0,
    "last_date": datetime.now().date()
}

def reset_trade_counter():
    today = datetime.now().date()
    if trade_counter["last_date"] != today:
        trade_counter["scalp"] = 0
        trade_counter["swing"] = 0
        trade_counter["last_date"] = today

def map_symbol(symbol):
    symbol_map = {
        "USDJPY": "JPY=X",
        "EURUSD": "EURUSD=X",
        "XAUUSD": "XAUUSD=X"
    }
    return symbol_map.get(symbol.upper(), symbol)

async def track_open_trades():
    all_assets = ["TSLA", "AAPL", "NVDA", "AMD", "USDJPY", "EURUSD", "XAUUSD"]
    for asset in all_assets:
        mapped_symbol = map_symbol(asset)
        latest_price = get_latest_price(mapped_symbol)
        if latest_price:
            track_live_trades()

# Define assets and timeframes for each session
scalping_assets = [
    ("TSLA", "stock", "5m"),
    ("AAPL", "stock", "5m"),
    ("NVDA", "stock", "5m"),
    ("AMD", "stock", "5m"),
    ("BTCUSD", "crypto", "5m")
]

swing_assets = [
    ("USDJPY", "forex", "1h"),
    ("EURUSD", "forex", "1h"),
    ("XAUUSD", "forex", "4h"),
    ("MNQ", "futures", "15m")
]

async def run_session(assets, session_name):
    logger.info(f"Running {session_name} session")
    reset_trade_counter()
    if is_red_folder_event_today():
        logger.info("Red folder news detected, skipping session")
        return

    for symbol, asset_type, timeframe in assets:
        # Skip assets not relevant to the session
        if asset_type == "crypto" and session_name != "london_ny":
            continue
        if asset_type == "stock" and session_name != "ny":
            continue

        # Fetch candles (placeholder: implement in data_feed.py)
        from bot.data_feed import fetch_candles
        candles_df = await fetch_candles(symbol, timeframe)
        if candles_df is None or candles_df.empty:
            logger.info(f"No data for {symbol}")
            continue

        # Check entry
        contract = trader.create_contract(symbol, asset_type)  # Placeholder: Implement in IBKRTrader
        entry = await check_entry(trader, candles_df, contract)
        if entry:
            add_live_trade(symbol, entry['entry_price'], entry['stop_loss'], entry['take_profit'], entry['direction'])
            log_trade({
                "asset": symbol,
                "direction": entry['direction'],
                "entry_price": entry['entry_price'],
                "exit_price": None,
                "stop_loss": entry['stop_loss'],
                "take_profit": entry['take_profit'],
                "rr_ratio": 2.0,
                "result": "Pending",
                "setup_type": "scalp" if asset_type in ["stock", "crypto"] else "swing",
                "confidence_score": entry['score'],
                "notes": f"Session: {session_name}"
            })
            if asset_type in ["stock", "crypto"]:
                trade_counter["scalp"] += 1
            else:
                trade_counter["swing"] += 1

    await track_open_trades()

    # After session, assess performance and update strategy
    from bot.bot_trainer import assess_trade_performance, update_strategy
    adjustments = assess_trade_performance()
    if adjustments:
        update_strategy(adjustments)
        from bot.scorer import update_weights
        from bot.smc_strategy import update_strategy_params
        update_weights(adjustments)
        update_strategy_params(adjustments)
        logger.info(f"Applied strategy adjustments: {adjustments}")

async def main():
    await trader.connect()
    # Asian Session (00:00-08:00 UTC): Forex (4H), Futures (15-min)
    schedule.every().day.at("00:00").do(
        lambda: asyncio.create_task(run_session(swing_assets, "asian"))
    )
    # London-NY Overlap (12:00-20:00 UTC): Forex (1H/4H), Futures (15-min), Crypto (5-min)
    schedule.every().day.at("12:00").do(
        lambda: asyncio.create_task(run_session(swing_assets + scalping_assets, "london_ny"))
    )
    # NY Open (13:30-20:00 UTC): Options (5-min), Futures (15-min), Crypto (5-min)
    schedule.every().day.at("13:30").do(
        lambda: asyncio.create_task(run_session(scalping_assets, "ny"))
    )
    logger.info("AI Scheduler running...")
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())