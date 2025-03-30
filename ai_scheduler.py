# ✅ Auto Scheduler + Trade Tracker (Step 2 of Final Bot Setup)
# This script will run every N minutes (or at custom times)
# It will: fetch data, run strategy, place trades, and log outcomes.

import os
import time
import schedule
from datetime import datetime
from dotenv import load_dotenv
from data_feed import run_single_asset

load_dotenv()

# Define your scheduled asset runs here
def run_forex_swing():
    print("\n🕙 Running scheduled swing check for USDJPY")
    run_single_asset("USDJPY", "forex")

def run_stock_scalping():
    print("\n⚡ Running scheduled scalping check for TSLA")
    run_single_asset("TSLA", "stock")

# Add schedules here (can be expanded based on strategy)
schedule.every(1).hours.do(run_forex_swing)
schedule.every(15).minutes.do(run_stock_scalping)

print("✅ AI Scheduler started at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f"❌ Scheduler encountered error: {e}")
        # Optional: Alert to Discord if webhook available
        webhook_url = os.getenv("DISCORD_WEBHOOK_SCALPING")
        if webhook_url:
            import requests
            requests.post(webhook_url, json={"content": f"❌ Scheduler error at {datetime.now()} — {e}"})
        time.sleep(60)  # Wait before retrying
