# scheduler.py
import subprocess
import time
from config import get_config
# Load config
config = get_config()

print("⏱️ Starting full multi-asset run...")
for asset in config["assets"]:
    symbol = asset["symbol"]
    asset_type = asset["asset_type"]
    print(f"🚀 Running bot for {symbol} ({asset_type})...")
    subprocess.run(["python", "data_feed.py", symbol, "--asset_type", asset_type])
    time.sleep(10)  # Optional: Delay between executions to prevent rate-limiting
