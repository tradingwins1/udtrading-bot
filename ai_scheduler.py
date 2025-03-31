# ai_scheduler.py
import schedule
import time
from datetime import datetime
from data_feed import run_single_asset
from discord_alert import send_alert
from red_news_filter import is_red_folder_event_today

# Configurable settings
ALLOWED_SCALPING_WINDOW = ("08:30", "10:30")  # CST
MAX_SCALPS_PER_DAY = 2
scalp_counter = 0

# Assets to trade
scalping_assets = [
    {"symbol": "TSLA", "type": "stock"},
    {"symbol": "AAPL", "type": "stock"},
    {"symbol": "NVDA", "type": "stock"},
    {"symbol": "AMD", "type": "stock"},
    {"symbol": "ETHUSD", "type": "crypto"},
]
swing_assets = [
    {"symbol": "USDJPY", "type": "forex"},
    {"symbol": "EURUSD", "type": "forex"},
    {"symbol": "XAUUSD", "type": "forex"},
]

def is_time_in_range(start, end):
    now = datetime.now().strftime("%H:%M")
    return start <= now <= end

def run_scalping():
    global scalp_counter
    if scalp_counter >= MAX_SCALPS_PER_DAY:
        print("✅ Daily scalp limit reached.")
        return

    if not is_time_in_range(*ALLOWED_SCALPING_WINDOW):
        print("⏱️ Not within scalping hours.")
        return

    if is_red_folder_event_today():
        print("🚫 High-impact news day. Scalping disabled.")
        send_alert("🚨 Scalping skipped due to red-folder news.")
        return

    for asset in scalping_assets:
        if scalp_counter >= MAX_SCALPS_PER_DAY:
            break
        run_single_asset(asset["symbol"], asset["type"])
        scalp_counter += 1


def run_swing():
    if is_red_folder_event_today():
        print("🚫 High-impact news day. Swing trade skipped.")
        send_alert("🚨 Swing trade skipped due to red-folder news.")
        return

    for asset in swing_assets:
        run_single_asset(asset["symbol"], asset["type"])

# Daily schedule (CST logic assumed to be local timezone)
schedule.every().day.at("08:30").do(run_scalping)
schedule.every().day.at("14:00").do(run_swing)  # Midday 2 PM for swing

def stop_scheduler():
    print("🛑 Scheduler stopped.")
    exit()

if __name__ == "__main__":
    print("📅 AI Scheduler running...")
    while True:
        schedule.run_pending()
        time.sleep(30)
