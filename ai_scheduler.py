<<<<<<< HEAD
import schedule
import asyncio
=======
# ai_scheduler.py
import schedule
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
import time
from datetime import datetime
from data_feed import run_single_asset
from red_news_filter import is_red_folder_event_today
from tracker import check_trade_exit
from price_feed import get_latest_price
import pytz
import os
from dotenv import load_dotenv
from db import init_db
<<<<<<< HEAD
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
=======
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3

init_db()
load_dotenv()

<<<<<<< HEAD
=======
# Track trades per day
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
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

def is_scalping_window():
    now = datetime.now(pytz.timezone("US/Central"))
    if now.hour == 8 and now.minute < 30:
<<<<<<< HEAD
        logger.info("Scalping automation only starts at 8:30 AM CST")
        return False
    elif now.hour == 10 and now.minute > 30:
        logger.info("Scalping automation ends after 10:30 AM CST")
        return False
    elif now.hour < 8 or now.hour > 10:
        logger.info("Scalping window is from 8:30 AM to 10:30 AM CST only")
=======
        print("⏳ Scalping automation only starts at 8:30 AM CST")
        return False
    elif now.hour == 10 and now.minute > 30:
        print("⏳ Scalping automation ends after 10:30 AM CST")
        return False
    elif now.hour < 8 or now.hour > 10:
        print("⏳ Scalping window is from 8:30 AM to 10:30 AM CST only")
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
        return False
    return True

def is_swing_window():
    now = datetime.now(pytz.timezone("US/Central"))
    if now.hour < 7:
<<<<<<< HEAD
        logger.info("Swing scanning only starts at 7:00 AM CST")
        return False
    elif now.hour > 11:
        logger.info("Swing scanning ends after 11:00 AM CST")
        return False
    return True

=======
        print("⏳ Swing scanning only starts at 7:00 AM CST")
        return False
    elif now.hour > 11:
        print("⏳ Swing scanning ends after 11:00 AM CST")
        return False
    return True

# ✅ Symbol mapping for Yahoo Finance
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
def map_symbol(symbol):
    symbol_map = {
        "USDJPY": "JPY=X",
        "EURUSD": "EURUSD=X",
        "XAUUSD": "XAUUSD=X"
    }
    return symbol_map.get(symbol.upper(), symbol)

<<<<<<< HEAD
async def track_open_trades():
=======
# ✅ Monitor SL/TP status
def track_open_trades():
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
    all_assets = ["TSLA", "AAPL", "NVDA", "AMD", "USDJPY", "EURUSD", "XAUUSD"]
    for asset in all_assets:
        mapped_symbol = map_symbol(asset)
        latest_price = get_latest_price(mapped_symbol)
        if latest_price:
            check_trade_exit(asset, latest_price)
        else:
<<<<<<< HEAD
            logger.warning(f"Could not fetch price for {asset}")

=======
            print(f"⚠️ Could not fetch price for {asset}")

# Scalping assets
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
scalping_assets = [
    ("TSLA", "stock"),
    ("AAPL", "stock"),
    ("NVDA", "stock"),
    ("AMD", "stock")
]

<<<<<<< HEAD
async def run_scalping():
    logger.info("Running scalping automation")
    reset_trade_counter()
    if not is_scalping_window():
        logger.info("Outside scalping hours")
        return
    if is_red_folder_event_today():
        logger.info("Red folder news detected, skipping scalping today")
        return
    for symbol, asset_type in scalping_assets:
        if trade_counter["scalp"] >= 2:
            logger.info("Max scalps reached for today")
            return
        await asyncio.to_thread(run_single_asset, symbol, asset_type)  # Run sync function in thread
        trade_counter["scalp"] += 1
    await track_open_trades()

=======
def run_scalping():
    print("🔍 Running scalping automation")
    reset_trade_counter()
    if not is_scalping_window():
        print("⏰ Outside scalping hours")
        return
    if is_red_folder_event_today():
        print("🚫 Red folder news detected, skipping scalping today")
        return
    for symbol, asset_type in scalping_assets:
        if trade_counter["scalp"] >= 2:
            print("✅ Max scalps reached for today")
            return
        run_single_asset(symbol, asset_type)
        trade_counter["scalp"] += 1
    track_open_trades()

# Swing assets
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
swing_assets = [
    ("USDJPY", "forex"),
    ("EURUSD", "forex"),
    ("XAUUSD", "forex")
]

<<<<<<< HEAD
async def run_swing():
    logger.info("Running swing setup")
    reset_trade_counter()
    if not is_swing_window():
        logger.info("Outside swing hours")
        return
    if is_red_folder_event_today():
        logger.info("Red folder news detected, skipping swing today")
        return
    for symbol, asset_type in swing_assets:
        if trade_counter["swing"] >= 2:
            logger.info("Swing trade already executed today")
            return
        await asyncio.to_thread(run_single_asset, symbol, asset_type)  # Run sync function in thread
        trade_counter["swing"] += 1
    await track_open_trades()

async def run_scheduler():
    schedule.every(5).minutes.do(lambda: asyncio.run(run_scalping()))
    schedule.every(1).hours.do(lambda: asyncio.run(run_swing()))
    logger.info("AI Scheduler running...")
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run_scheduler())
=======
def run_swing():
    print("🔍 Running swing setup")
    reset_trade_counter()
    if not is_swing_window():
        print("⏰ Outside swing hours")
        return
    if is_red_folder_event_today():
        print("🚫 Red folder news detected, skipping swing today")
        return
    for symbol, asset_type in swing_assets:
        if trade_counter["swing"] >= 2:
            print("✅ Swing trade already executed today")
            return
        run_single_asset(symbol, asset_type)
        trade_counter["swing"] += 1
    track_open_trades()

# Main scheduler loop
schedule.every(5).minutes.do(run_scalping)
schedule.every(1).hours.do(run_swing)

print("📅 AI Scheduler running...")
while True:
    schedule.run_pending()
    time.sleep(1)
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
