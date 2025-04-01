# ai_scheduler.py
import schedule
import time
from datetime import datetime
from data_feed import run_single_asset
from red_news_filter import is_red_folder_event_today
from tracker import check_trade_exit  # ✅ NEW
from price_feed import get_latest_price  # ✅ Assumes you have this module
import pytz
import os
from dotenv import load_dotenv

from db import init_db   # ✅ Add this line
init_db()                # ✅ Run it once at startup

load_dotenv()

# Track trades per day
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
        print("⏳ Scalping automation only starts at 8:30 AM CST")
        return False
    elif now.hour == 10 and now.minute > 30:
        print("⏳ Scalping automation ends after 10:30 AM CST")
        return False
    elif now.hour < 8 or now.hour > 10:
        print("⏳ Scalping window is from 8:30 AM to 10:30 AM CST only")
        return False
    return True

def is_swing_window():
    now = datetime.now(pytz.timezone("US/Central"))
    if now.hour < 7:
        print("⏳ Swing scanning only starts at 7:00 AM CST")
        return False
    elif now.hour > 11:
        print("⏳ Swing scanning ends after 11:00 AM CST")
        return False
    return True

# ✅ New: Monitor SL/TP for tracked assets
def track_open_trades():
    all_assets = ["TSLA", "AAPL", "NVDA", "AMD", "USDJPY", "EURUSD", "XAUUSD"]
    for asset in all_assets:
        latest_price = get_latest_price(asset)
        if latest_price:
            check_trade_exit(asset, latest_price)

# Main scalping job
scalping_assets = [
    ("TSLA", "stock"),
    ("AAPL", "stock"),
    ("NVDA", "stock"),
    ("AMD", "stock")
]

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
    track_open_trades()  # ✅ Monitor SL/TP

# Swing trading job
swing_assets = [
    ("USDJPY", "forex"),
    ("EURUSD", "forex"),
    ("XAUUSD", "forex")
]

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
    track_open_trades()  # ✅ Monitor SL/TP

# Scheduler Loop
schedule.every(5).minutes.do(run_scalping)
schedule.every(1).hours.do(run_swing)

print("📅 AI Scheduler running...")
while True:
    schedule.run_pending()
    time.sleep(1)
