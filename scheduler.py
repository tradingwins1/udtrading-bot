
import json
import subprocess
import time

with open("config.json") as f:
    config = json.load(f)

print("⏱️ Starting full multi-asset run...")
for asset in config["assets"]:
    symbol = asset["symbol"]
    asset_type = asset["asset_type"]
    print(f"🚀 Running bot for {symbol} ({asset_type})...")
    subprocess.run(["python", "data_feed.py", symbol, "--asset_type", asset_type])
    time.sleep(10)  # Optional: Delay between executions to prevent rate-limiting
