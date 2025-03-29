import requests
import os

def send_alert(message, side, entry, sl, tp):
    try:
        webhook_url = os.getenv("DISCORD_WEBHOOK_SCALP")
        if not webhook_url:
            raise ValueError("Discord webhook not set in environment variables.")

        payload = {
            "content": f"**{side.upper()} ALERT**\n"
                       f"{message}\n"
                       f"Entry: `{entry}` | SL: `{sl}` | TP: `{tp}`"
        }

        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        print(f"📣 Discord alert sent! ({response.status_code})")

    except Exception as e:
        print(f"❌ Failed to send Discord alert: {e}")