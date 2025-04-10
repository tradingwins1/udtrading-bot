# discord_alert.py
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def send_alert(symbol, side, entry, sl, tp, timeframe, confidence=8, alert_type="scalp", reason=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if alert_type == "swing":
        webhook_url = os.getenv("DISCORD_WEBHOOK_SWING")
        title = f"📈 Swing Trade Alert [{side.upper()}]"
    else:
        webhook_url = os.getenv("DISCORD_WEBHOOK_SCALPING")
        title = f"⚡ Scalping Alert [{side.upper()}]"

    if not webhook_url:
        print("❌ Failed to send Discord alert: Discord webhook not set in environment variables.")
        return

    content = f"**{title}**\n\n"
    content += f"**Asset:** `{symbol}`\n"
    content += f"**Direction:** `{side.upper()}`\n"
    content += f"**Entry:** `{entry}` | **SL:** `{sl}` | **TP:** `{tp}`\n"
    content += f"**Timeframe:** `{timeframe}`\n"
    content += f"**Confidence:** `{confidence}/10`\n"
    if reason:
        content += f"**Reason:** {reason}\n"
    content += f"**Triggered:** `{now}`"

    try:
        response = requests.post(webhook_url, json={"content": content})
        if response.status_code == 204:
            print(f"📣 Discord alert sent! ({alert_type.upper()}) ✅")
        else:
            print(f"❌ Discord alert failed! Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        print(f"❌ Exception while sending Discord alert: {e}")